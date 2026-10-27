"""Staged experiment orchestration. This module never opens test answers."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from .io_utils import append_journal, canonical_hash, file_hash, read_journal, tree_hash, write_json
from .selection import SimilarityIndex, choose_common_n, random_examples


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_predictions(path, rows):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Sample_id", "Answer"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ["Sample_id", "Answer"]} for r in rows)
    temp.replace(path)


def code_fingerprint():
    return canonical_hash({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def verify_data(prepared):
    prepared = Path(prepared)
    manifest = read_json(prepared / "manifest.json")
    for name, expected in manifest["artifact_sha256"].items():
        if file_hash(prepared / name) != expected:
            raise ValueError(f"Prepared data changed: {name}")
    return manifest


def environment_snapshot(output):
    from .modeling import dependency_versions
    import torch
    import bitsandbytes
    import peft
    import sentence_transformers
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. In Colab choose Runtime > Change runtime type > GPU")
    result = {"python": sys.version, "platform": platform.platform(),
              "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
              "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "bf16_supported": torch.cuda.is_bf16_supported(including_emulation=False),
              "versions": dependency_versions(), "status": "imports_verified_gpu_smoke_pending"}
    # A successful import alone is not a successful quantized inference/training test.
    write_json(output, result)
    Path(output).with_suffix(".pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def preflight(prepared, config_path, output, gpu=False):
    from .modeling import ModelConfig, load_tokenizer
    from .prompts import audit_input_lengths
    from .training import encode_assistant_example
    manifest = verify_data(prepared)
    config = read_json(config_path)
    if "inference_recovery" in config:
        from .recovery import verify_inference_recovery
        verify_inference_recovery(config_path, prepared)
    tokenizer = load_tokenizer(ModelConfig(**config["model"]))
    result = {"config_sha256": file_hash(config_path), "code_sha256": code_fingerprint(),
              "data_manifest_sha256": file_hash(Path(prepared) / "manifest.json"),
              "model_revision": config["model"]["revision"], "splits": {}}
    for split in ["train", "validation", "test"]:
        rows = read_rows(Path(prepared) / f"{split}.csv")
        result["splits"][split] = audit_input_lengths(rows, tokenizer, config["input_budget"])
        print(split, json.dumps({k: v for k, v in result["splits"][split].items() if k != "oversize"}))
    lengths, failures, mask_examples = [], [], []
    for split in ["train", "validation"]:
        for row in read_rows(Path(prepared) / f"{split}.csv"):
            try:
                encoded = encode_assistant_example(row, tokenizer, config["training"]["max_length"])
                lengths.append(len(encoded["input_ids"]))
                if row["Task"] not in {r["Task"] for r in mask_examples}:
                    supervised = [x for x in encoded["labels"] if x != -100]
                    mask_examples.append({"Sample_id": row["Sample_id"], "Task": row["Task"],
                                          "answer": row["Answer"],
                                          "supervised_text": tokenizer.decode(supervised),
                                          "input_tokens": len(encoded["input_ids"]),
                                          "supervised_tokens": len(supervised)})
            except ValueError as e:
                failures.append({"Sample_id": row["Sample_id"], "error": str(e)})
    result["training_sequences"] = {"checked": len(lengths) + len(failures),
                                     "max_tokens": max(lengths, default=0),
                                     "failures": failures, "mask_examples": mask_examples}
    result["status"] = "passed" if not failures and not any(
        x["oversize_count"] for x in result["splits"].values()) else "blocked"
    write_json(output, result)
    print(json.dumps({"status": result["status"], "training_sequences": result["training_sequences"]},
                     ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Length/mask preflight blocked. Inspect report, explicitly revise shared policy, rerun. No rows dropped.")
    if gpu:
        import torch
        from .modeling import dependency_versions
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required")
        dtype = "bfloat16" if torch.cuda.is_bf16_supported(including_emulation=False) else "float16"
        if config["model"]["compute_dtype"] not in ["auto", dtype]:
            raise RuntimeError("GPU differs from configured precision; keep controls consistent")
        config["model"]["compute_dtype"] = dtype
        config["data_manifest_sha256"] = result["data_manifest_sha256"]
        config["source_code_sha256"] = result["code_sha256"]
        config["runtime_versions"] = dependency_versions()
        write_json(Path(output).parent / "run_config.json", config, immutable=True)
    return result


def inference_config(config):
    from .prompts import PROMPT_VERSION
    return {"model": config["model"], "input_budget": config["input_budget"],
            "max_new_tokens": config["max_new_tokens"], "quantization": "nf4-double",
            "enable_thinking": False, "do_sample": False, "num_beams": 1,
            "prompt_version": PROMPT_VERSION}


def predict(prepared, config_path, output_dir, method="baseline", split="validation_tune",
            n=0, limit=None, loaded=None, index=None):
    from .modeling import ModelConfig, dependency_versions, generate_answer, load_model
    from .prompts import build_prompt
    if method not in ["baseline", "random", "similar", "sft"]:
        raise ValueError("Unknown method")
    if split not in ["validation_tune", "validation", "test"]:
        raise ValueError("Only validation/test prediction targets are allowed")
    manifest = verify_data(prepared)
    config_path, prepared, output_dir = Path(config_path), Path(prepared), Path(output_dir)
    config = read_json(config_path)
    if "inference_recovery" in config:
        from .recovery import verify_inference_recovery
        verify_inference_recovery(config_path, prepared)
    if "runtime_versions" in config and config["runtime_versions"] != dependency_versions():
        raise ValueError("Dependency versions changed; restore saved runtime before resuming")
    if config["model"]["compute_dtype"] == "auto":
        raise ValueError("Run GPU preflight to resolve and record compute precision first")
    if split == "test":
        if not config.get("experiment_configuration_frozen") or limit is not None:
            raise ValueError("All test runs require frozen config and full test coverage")
        if code_fingerprint() != config["source_code_sha256"]:
            raise ValueError("Code changed after configuration freeze")
        if file_hash(prepared / "manifest.json") != config["data_manifest_sha256"]:
            raise ValueError("Data changed after configuration freeze")
        n = config["selected_n"] if method in ["random", "similar"] else 0
    if method in ["baseline", "sft"] and n != 0:
        raise ValueError("Baseline/SFT must have zero demonstrations")
    rows = read_rows(prepared / f"{split}.csv")
    expected_key = {"validation_tune": "tune_ids", "validation": "validation_ids", "test": "test_ids"}[split]
    if {r["Sample_id"] for r in rows} != set(manifest[expected_key]):
        raise ValueError("Target IDs differ from split manifest")
    if limit is not None:
        # smoke contains all three task types, never sampled from test
        rows = [r for task in ["qa", "rag", "math"]
                for r in [x for x in rows if x["Task"] == task][:limit]]
    ids = {r["Sample_id"] for r in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate target IDs")
    pool = read_rows(prepared / "train.csv")
    if {r["Sample_id"] for r in pool} != set(manifest["train_ids"]):
        raise ValueError("Example pool must equal saved training subset")
    if ids & set(manifest["train_ids"]):
        raise ValueError("Inference and training IDs overlap")
    adapter = None
    if method == "sft":
        adapter = (config_path.parent / config["adapter_path"]).resolve()
        if tree_hash(adapter) != config["adapter_sha256"]:
            raise ValueError("Adapter changed after freezing")
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"config_sha256": file_hash(config_path), "input_sha256": file_hash(prepared / f"{split}.csv"),
                "pool_sha256": file_hash(prepared / "train.csv"), "code_sha256": code_fingerprint(),
                "method": method, "n": n, "split": split, "limit_per_task": limit,
                "expected_ids": [r["Sample_id"] for r in rows], "inference_config": inference_config(config)}
    write_json(output_dir / "identity.json", identity, immutable=True)
    journal = output_dir / "raw_predictions.jsonl"
    completed = read_journal(journal, ids)
    if len(completed) < len(rows):
        loaded = loaded or load_model(ModelConfig(**config["model"]), adapter_path=str(adapter) if adapter else None)
        if loaded.metadata.get("config") != config["model"]:
            raise ValueError("Loaded model configuration differs from the run")
        loaded_adapter = loaded.metadata.get("adapter_path")
        if (adapter is None and loaded_adapter is not None) or (adapter is not None and
            (loaded_adapter is None or Path(loaded_adapter).resolve() != adapter)):
            raise ValueError("Loaded adapter does not match the requested method")
        if method == "similar":
            index = index or SimilarityIndex(pool, config["embedding"], output_dir.parent / "embedding_cache")
        for row in rows:
            if row["Sample_id"] in completed:
                continue
            query = {k: row[k] for k in ["Sample_id", "Task", "Question", "Context"]}
            selected_at = time.perf_counter()
            if method == "random":
                demos = random_examples(pool, query, n, config["seed"])
            elif method == "similar":
                demos, _ = index.select(query, n)
            else:
                demos = []
            selection_seconds = time.perf_counter() - selected_at
            if any(d["Sample_id"] not in manifest["train_ids"] for d in demos):
                raise ValueError("Non-training demonstration detected")
            prompt_started = time.perf_counter()
            prompt = build_prompt(query, loaded.tokenizer, demos, config["input_budget"])
            prompt_seconds = time.perf_counter() - prompt_started
            record = {"Sample_id": row["Sample_id"], "Task": row["Task"],
                      **generate_answer(loaded, prompt, config["max_new_tokens"]),
                      "requested_shots": n, "actual_shots": prompt.actual_shots,
                      "demo_ids": prompt.demo_ids, "prompt_sha256": canonical_hash(prompt.text),
                      "selection_seconds": selection_seconds,
                      "prompt_seconds": prompt_seconds,
                      "runtime": loaded.metadata}
            append_journal(journal, record)
            completed[row["Sample_id"]] = record
            if len(completed) % 10 == 0 or len(completed) == len(rows):
                print(f"{method} {split}: {len(completed)}/{len(rows)}", flush=True)
    records = [completed[r["Sample_id"]] for r in rows]
    write_predictions(output_dir / "predictions.csv", records)
    elapsed = sum(r["elapsed_seconds"] for r in records)
    stats = {"count": len(records), "inference_seconds": elapsed,
             "inference_timing_scope": "model.generate only; selection and prompt construction reported separately",
             "examples_per_second": len(records) / elapsed if elapsed else None,
             "generated_tokens_per_second": sum(r["generated_tokens"] for r in records) / elapsed if elapsed else None,
             "mean_input_tokens": sum(r["input_tokens"] for r in records) / len(records),
             "mean_generated_tokens": sum(r["generated_tokens"] for r in records) / len(records),
             "mean_actual_shots": sum(r["actual_shots"] for r in records) / len(records),
             "selection_seconds": sum(r["selection_seconds"] for r in records),
             "prompt_seconds": sum(r["prompt_seconds"] for r in records),
             "peak_memory_allocated_bytes": max(r["peak_memory_allocated_bytes"] for r in records),
             "hit_token_limit_count": sum(r["hit_token_limit"] for r in records)}
    stats["actual_shots_by_task"] = {}
    for task in ["qa", "rag", "math"]:
        task_rows = [r for r in records if r["Task"] == task]
        if task_rows:
            stats["actual_shots_by_task"][task] = {
                "count": len(task_rows), "mean": sum(r["actual_shots"] for r in task_rows) / len(task_rows),
                "histogram": {str(n): sum(r["actual_shots"] == n for r in task_rows)
                              for n in sorted({r["actual_shots"] for r in task_rows})}}
    stats_path = output_dir / "statistics.json"
    if index is not None:
        stats["index"] = {"preparation_seconds_this_process": index.preparation_seconds,
                          "original_build_seconds": index.original_index_seconds,
                          "cache_hit": index.cache_hit}
    elif stats_path.exists() and "index" in read_json(stats_path):
        stats["index"] = read_json(stats_path)["index"]
    write_json(stats_path, stats)
    run_manifest = {**identity, "completed": True, "count": len(records),
                    "test_sha256": file_hash(prepared / "test.csv"),
                    "prediction_sha256": file_hash(output_dir / "predictions.csv"),
                    "raw_sha256": file_hash(journal)}
    write_json(output_dir / "run_manifest.json", run_manifest)
    if split != "test":
        from .metrics import score_records
        report = score_records(rows, records)
        write_json(output_dir / "evaluation.json", report)
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return stats


def tune(prepared, config_path, output):
    from .modeling import load_model
    config = read_json(config_path)
    if "inference_recovery" in config:
        from .recovery import verify_inference_recovery
        verify_inference_recovery(config_path, prepared)
    loaded = load_model(config["model"])
    pool = read_rows(Path(prepared) / "train.csv")
    index = SimilarityIndex(pool, config["embedding"], Path(output) / "embedding_cache")
    scores, evidence = {}, {}
    for n in config["n_candidates"]:
        scores[n] = {}
        for method in ["random", "similar"]:
            folder = Path(output) / f"{method}_{n}"
            predict(prepared, config_path, folder, method=method, n=n, loaded=loaded, index=index if method == "similar" else None)
            scores[n][method] = read_json(folder / "evaluation.json")["metrics"]["macro_score"]
            evidence[f"{method}_{n}"] = {"evaluation_sha256": file_hash(folder / "evaluation.json"),
                                        "run_manifest_sha256": file_hash(folder / "run_manifest.json"),
                                        "prediction_sha256": file_hash(folder / "predictions.csv")}
    result = {"selected_n": choose_common_n(scores), "scores": scores, "evidence": evidence,
              "selection_rule": "mean(random macro, similar macro); ties within 1e-12 choose smaller n",
              "config_sha256": file_hash(config_path), "code_sha256": code_fingerprint(),
              "data_manifest_sha256": file_hash(Path(prepared) / "manifest.json"),
              "validation_ids": [r["Sample_id"] for r in read_rows(Path(prepared) / "validation_tune.csv")]}
    write_json(Path(output) / "selection.json", result, immutable=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def freeze(prepared, config_path, selection_path, adapter_path, output):
    import os
    config = read_json(config_path)
    selection = read_json(selection_path)
    verify_data(prepared)
    recovery = None
    if "inference_recovery" in config:
        from .recovery import verify_inference_recovery
        recovery = verify_inference_recovery(config_path, prepared)
        if Path(output).resolve().parent != Path(config_path).resolve().parent:
            raise ValueError("Recovery freeze must stay beside its run_config.json to preserve relative provenance")
    if selection["config_sha256"] != file_hash(config_path) or selection["code_sha256"] != code_fingerprint():
        raise ValueError("Selection configuration/code differs from current experiment")
    if selection["data_manifest_sha256"] != file_hash(Path(prepared) / "manifest.json"):
        raise ValueError("Data differs from validation selection")
    manifest = read_json(Path(prepared) / "manifest.json")
    if set(selection["validation_ids"]) != set(manifest["tune_ids"]) or len(selection["validation_ids"]) != 180:
        raise ValueError("Selection must use exactly the fixed 180 tuning samples")
    if set(map(int, selection["scores"])) != set(config["n_candidates"]):
        raise ValueError("Missing n-shot candidate results")
    if selection["selected_n"] != choose_common_n(selection["scores"]):
        raise ValueError("Selected n contradicts saved validation scores")
    from .metrics import score_records
    references = read_rows(Path(prepared) / "validation_tune.csv")
    for n in config["n_candidates"]:
        for method in ["random", "similar"]:
            key = f"{method}_{n}"
            folder = Path(selection_path).parent / key
            for name, artifact in [("evaluation", "evaluation.json"), ("run_manifest", "run_manifest.json"),
                                   ("prediction", "predictions.csv")]:
                if file_hash(folder / artifact) != selection["evidence"][key][f"{name}_sha256"]:
                    raise ValueError("Validation selection evidence changed")
            run = read_json(folder / "run_manifest.json")
            if not run["completed"] or run["config_sha256"] != file_hash(config_path) or run["n"] != n or run["method"] != method:
                raise ValueError("Validation run metadata differs")
            actual = score_records(references, read_rows(folder / "predictions.csv"))["metrics"]["macro_score"]
            if actual != selection["scores"][str(n)][method]:
                raise ValueError("Validation score does not match complete predictions")
    adapter_path, output = Path(adapter_path).resolve(), Path(output).resolve()
    if not (adapter_path / "adapter_config.json").exists() or not (adapter_path / "adapter_model.safetensors").exists():
        raise ValueError("A real completed training adapter is required")
    training_report = adapter_path.parent / "training_summary.json"
    if not training_report.exists():
        raise ValueError("Missing completed training evidence")
    evidence = read_json(training_report)
    if evidence.get("smoke") or not evidence.get("completed"):
        raise ValueError("Smoke/unfinished training cannot be frozen as final SFT")
    evidence_config = Path(recovery["parent_config_path"]) if recovery else Path(config_path)
    evidence_code = recovery["parent_code_sha256"] if recovery else code_fingerprint()
    if recovery and (adapter_path != Path(recovery["adapter_path"]) or training_report != Path(recovery["training_summary_path"])):
        raise ValueError("Recovery freeze must use the verified unchanged SFT copy")
    if evidence.get("config_sha256") != file_hash(evidence_config):
        raise ValueError("Training configuration differs from validation configuration")
    for name in ["train", "validation"]:
        if evidence.get(f"{name}_csv_sha256") != file_hash(Path(prepared) / f"{name}.csv"):
            raise ValueError("Training data differs from prepared split")
    if evidence.get("code_sha256") != evidence_code:
        raise ValueError("Training code differs from current code")
    if evidence.get("data_manifest_sha256") != file_hash(Path(prepared) / "manifest.json"):
        raise ValueError("Training data manifest differs")
    if evidence.get("dependencies") != config.get("runtime_versions"):
        raise ValueError("Training dependencies differ from validation runtime")
    if evidence.get("adapter_sha256") != tree_hash(adapter_path):
        raise ValueError("Adapter changed after training completed")
    config.update({"experiment_configuration_frozen": True, "selected_n": selection["selected_n"],
                   "selection_sha256": file_hash(selection_path),
                   "adapter_path": os.path.relpath(adapter_path, output.parent).replace("\\", "/"),
                   "adapter_sha256": tree_hash(adapter_path),
                   "source_code_sha256": code_fingerprint(),
                   "data_manifest_sha256": file_hash(Path(prepared) / "manifest.json"),
                   "training_summary_sha256": file_hash(training_report)})
    config["dependency_status"] = "validated_by_actual_validation_inference_and_completed_sft"
    write_json(output, config, immutable=True)
    print(f"Configuration frozen: {output}")


def bundle(prepared, config_path, runs_dir, output):
    import os
    output = Path(output).resolve()
    relative = lambda p: os.path.relpath(Path(p).resolve(), output.parent).replace("\\", "/")
    result = {"schema_version": 1, "config_path": relative(config_path),
              "config_sha256": file_hash(config_path), "test_path": relative(Path(prepared) / "test.csv"),
              "test_sha256": file_hash(Path(prepared) / "test.csv"),
              "data_manifest_path": relative(Path(prepared) / "manifest.json"),
              "data_manifest_sha256": file_hash(Path(prepared) / "manifest.json"), "runs": {}}
    for method in ["baseline", "random", "similar", "sft"]:
        run = Path(runs_dir) / method
        prediction, manifest = run / "predictions.csv", run / "run_manifest.json"
        result["runs"][method] = {"prediction_path": relative(prediction), "prediction_sha256": file_hash(prediction),
                                 "run_manifest_path": relative(manifest), "run_manifest_sha256": file_hash(manifest)}
    write_json(output, result, immutable=True)
    print(f"Four-run bundle saved: {output}; test answers have not been opened")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["environment", "preflight", "smoke", "predict", "tune", "freeze", "bundle"])
    parser.add_argument("--prepared", default="data/prepared")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--split", default="validation_tune")
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--selection")
    parser.add_argument("--adapter")
    parser.add_argument("--runs-dir")
    args = parser.parse_args()
    if args.stage == "environment":
        environment_snapshot(args.output)
    elif args.stage == "preflight":
        preflight(args.prepared, args.config, args.output, args.gpu)
    elif args.stage in ["smoke", "predict"]:
        predict(args.prepared, args.config, args.output, args.method, args.split, args.n,
                limit=2 if args.stage == "smoke" else None)
    elif args.stage == "tune":
        tune(args.prepared, args.config, args.output)
    elif args.stage == "freeze":
        freeze(args.prepared, args.config, args.selection, args.adapter, args.output)
    elif args.stage == "bundle":
        bundle(args.prepared, args.config, args.runs_dir, args.output)


if __name__ == "__main__":
    main()
