"""Validation evaluation and the sole, gated reader of test_answer.csv.

The final bundle hashes a frozen configuration, public test inputs, four
complete prediction CSVs and their run manifests. All checks finish before
the archive's answer member is opened. Hashes establish artifact consistency,
not a cryptographic assertion about a human's experiment chronology.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from .metrics import TASKS, index_records, require_same_ids, score_records

METHODS = ("baseline", "random", "similar", "sft")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON artifact: {path}")
    with path.open(encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _csv_records(handle: Any, label: str, required: set[str], exact_columns: bool = False) -> list[dict[str, str]]:
    reader = csv.DictReader(handle)
    fields = reader.fieldnames
    if fields is None or len(set(fields)) != len(fields) or not required.issubset(fields):
        raise ValueError(f"{label}: missing or duplicate CSV columns; required={sorted(required)!r}.")
    if exact_columns and set(fields) != required:
        raise ValueError(f"{label}: expected exactly these columns: {sorted(required)!r}.")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{label}: malformed CSV row {reader.line_num}.")
        rows.append(row)
    index_records(rows, label)
    return rows


def read_csv(path: str | Path, required: set[str] | None = None, exact_columns: bool = False) -> list[dict[str, str]]:
    """Read a public/prepared CSV; the answer file has a separate gated reader."""
    csv_path = Path(path)
    if csv_path.name.lower() == "test_answer.csv":
        raise ValueError("test_answer.csv may only be read through evaluate_final after all gates pass.")
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return _csv_records(handle, str(csv_path), required or {"Sample_id", "Answer"}, exact_columns)


def _write_json(path: Path, result: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _id_list(manifest: dict[str, Any], key: str) -> list[str]:
    ids = manifest.get(key)
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError(f"Split manifest requires a nonempty {key} list.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Split manifest contains duplicate IDs in {key}.")
    return ids


def evaluate_validation(
    references_path: str | Path,
    predictions_path: str | Path,
    split_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    references_path = Path(references_path)
    allowed = {"validation.csv": "validation_ids", "validation_tune.csv": "tune_ids"}
    if references_path.name not in allowed:
        raise ValueError("Validation references must be prepared validation.csv or validation_tune.csv.")
    manifest = _read_json(Path(split_manifest_path))
    validation_ids = _id_list(manifest, "validation_ids")
    train_ids = _id_list(manifest, "train_ids")
    test_ids = _id_list(manifest, "test_ids")
    tune_ids = _id_list(manifest, "tune_ids")
    if set(validation_ids) & (set(train_ids) | set(test_ids)) or set(train_ids) & set(test_ids):
        raise ValueError("Split manifest has overlapping train, validation or test IDs.")
    if not set(tune_ids).issubset(validation_ids):
        raise ValueError("Tuning IDs must be a subset of validation IDs.")
    references = read_csv(references_path, {"Sample_id", "Task", "Answer"})
    require_same_ids(manifest[allowed[references_path.name]], (row["Sample_id"] for row in references), "validation references")
    artifact_hash = manifest.get("artifact_sha256", {}).get(references_path.name)
    if artifact_hash is not None and sha256_file(references_path) != artifact_hash:
        raise ValueError("Validation reference content does not match the split manifest hash.")
    predictions = read_csv(predictions_path, {"Sample_id", "Answer"}, exact_columns=True)
    result = score_records(references, predictions)
    result["provenance"] = {
        "evaluation": "validation", "references_sha256": sha256_file(references_path),
        "predictions_sha256": sha256_file(predictions_path),
        "split_manifest_sha256": sha256_file(split_manifest_path),
    }
    _write_json(Path(output_path), result)
    return result


def _bundle_artifact(bundle_dir: Path, entry: dict[str, Any], path_key: str, hash_key: str) -> Path:
    value, expected_hash = entry.get(path_key), entry.get(hash_key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Final bundle requires {path_key}.")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise ValueError(f"Final bundle requires a lowercase SHA256 in {hash_key}.")
    path = Path(value)
    path = (bundle_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if path.name.lower() == "test_answer.csv":
        raise ValueError("Test answers cannot be used as pre-evaluation bundle artifacts.")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"Hash mismatch for {path_key}: {path}")
    return path


def _check_final_bundle(bundle_path: Path) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Complete every public-artifact check without opening the answer archive."""
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Final bundle schema_version must be 1.")
    bundle_dir = bundle_path.resolve().parent
    config_path = _bundle_artifact(bundle_dir, bundle, "config_path", "config_sha256")
    config = _read_json(config_path)
    if config.get("experiment_configuration_frozen") is not True:
        raise ValueError("Configuration must be explicitly frozen before final evaluation.")
    if any(key not in config for key in ("model", "input_budget", "max_new_tokens", "selected_n", "source_code_sha256", "data_manifest_sha256")):
        raise ValueError("Frozen configuration is missing shared inference settings.")
    if type(config["selected_n"]) is not int or config["selected_n"] not in (1, 3, 5):
        raise ValueError("Frozen selected_n must be one of the evaluated 1, 3, 5 candidates.")
    expected_inference = {
        "model": config["model"], "input_budget": config["input_budget"],
        "max_new_tokens": config["max_new_tokens"], "quantization": "nf4-double",
        "enable_thinking": False, "do_sample": False, "num_beams": 1,
        "prompt_version": "course-task-context-v1",
    }
    test_path = _bundle_artifact(bundle_dir, bundle, "test_path", "test_sha256")
    if test_path.name != "test.csv":
        raise ValueError("Final bundle test_path must reference public test.csv.")
    test_rows = read_csv(test_path, {"Sample_id", "Task", "Question", "Context"})
    if any("Answer" in row or "Accepted_answers" in row for row in test_rows):
        raise ValueError("Public test.csv must not contain answer columns.")
    if any(row["Task"] not in TASKS for row in test_rows):
        raise ValueError("Public test.csv contains an unsupported task.")
    data_manifest_path = _bundle_artifact(bundle_dir, bundle, "data_manifest_path", "data_manifest_sha256")
    if bundle["data_manifest_sha256"] != config["data_manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from the frozen configuration.")
    data_manifest = _read_json(data_manifest_path)
    source_test_hash = data_manifest.get("source_sha256", {}).get("test.csv")
    if not isinstance(source_test_hash, str) or len(source_test_hash) != 64 or any(char not in "0123456789abcdef" for char in source_test_hash):
        raise ValueError("Prepared manifest must record the original archive test.csv SHA256.")
    test_ids = _id_list(data_manifest, "test_ids")
    if data_manifest.get("artifact_sha256", {}).get("test.csv") != bundle["test_sha256"]:
        raise ValueError("Public test.csv differs from the frozen prepared data manifest.")
    require_same_ids(test_ids, (row["Sample_id"] for row in test_rows), "frozen public test.csv")
    runs = bundle.get("runs")
    if not isinstance(runs, dict) or set(runs) != set(METHODS):
        raise ValueError(f"Final evaluation requires exactly four runs: {METHODS!r}.")
    predictions: dict[str, list[dict[str, str]]] = {}
    for method in METHODS:
        entry = runs[method]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid run entry for {method}.")
        prediction_path = _bundle_artifact(bundle_dir, entry, "prediction_path", "prediction_sha256")
        manifest_path = _bundle_artifact(bundle_dir, entry, "run_manifest_path", "run_manifest_sha256")
        manifest = _read_json(manifest_path)
        required_manifest = {
            "method": method, "config_sha256": bundle["config_sha256"],
            "test_sha256": bundle["test_sha256"],
            "prediction_sha256": entry["prediction_sha256"], "completed": True,
            "code_sha256": config["source_code_sha256"], "split": "test",
            "limit_per_task": None, "input_sha256": bundle["test_sha256"],
            "n": config["selected_n"] if method in ("random", "similar") else 0,
            "count": len(test_rows),
        }
        for key, expected in required_manifest.items():
            if key not in manifest or manifest[key] != expected or (key == "completed" and manifest[key] is not True):
                raise ValueError(f"{method}: run manifest {key} disagrees with the frozen bundle.")
        require_same_ids(test_ids, _id_list(manifest, "expected_ids"), f"{method} planned prediction IDs")
        if manifest.get("inference_config") != expected_inference:
            raise ValueError(f"{method}: inference_config disagrees with the frozen shared settings.")
        rows = read_csv(prediction_path, {"Sample_id", "Answer"}, exact_columns=True)
        require_same_ids((row["Sample_id"] for row in test_rows), (row["Sample_id"] for row in rows), f"{method} predictions")
        predictions[method] = rows
    return bundle, test_rows, predictions, data_manifest


def evaluate_final(frozen_path: str | Path, answer_archive: str | Path, output: str | Path) -> dict[str, Any]:
    """Evaluate four frozen, complete runs; only this function opens test answers.

    ``frozen_path`` is final_bundle.json, whose config_path points to the
    earlier frozen configuration. Relative paths resolve from the bundle.
    ``answer_archive`` is the original course zip. No answer extraction occurs.
    """
    bundle_path = Path(frozen_path)
    bundle, test_rows, predictions, data_manifest = _check_final_bundle(bundle_path)

    # Privacy/leakage boundary: do not move any answer access above the gates.
    with zipfile.ZipFile(answer_archive) as archive:
        public_members = [name for name in archive.namelist() if Path(name).name == "test.csv" and not name.endswith("/")]
        if len(public_members) != 1:
            raise ValueError("Answer archive must contain exactly one public test.csv member.")
        public_digest = hashlib.sha256()
        with archive.open(public_members[0]) as public_handle:
            for chunk in iter(lambda: public_handle.read(1024 * 1024), b""):
                public_digest.update(chunk)
        if public_digest.hexdigest() != data_manifest["source_sha256"]["test.csv"]:
            raise ValueError("Answer archive public test.csv does not match the original prepared dataset source.")
        members = [name for name in archive.namelist() if Path(name).name == "test_answer.csv" and not name.endswith("/")]
        if len(members) != 1:
            raise ValueError("Answer archive must contain exactly one test_answer.csv member.")
        with archive.open(members[0]) as binary_handle:
            with io.TextIOWrapper(binary_handle, encoding="utf-8-sig", newline="") as handle:
                answer_rows = _csv_records(handle, "final test answers", {"Sample_id", "Answer"})
    answers = index_records(answer_rows, "final test answers")
    require_same_ids((row["Sample_id"] for row in test_rows), answers, "final test answers")
    references = []
    for row in test_rows:
        answer = answers[row["Sample_id"]]
        if "Task" in answer and answer["Task"] != row["Task"]:
            raise ValueError(f"Test task mismatch for {row['Sample_id']!r}.")
        references.append({**row, **answer})
    result = {
        "provenance": {
            "evaluation": "final_test", "final_bundle_sha256": sha256_file(bundle_path),
            "config_sha256": bundle["config_sha256"], "test_sha256": bundle["test_sha256"],
            "data_manifest_sha256": bundle["data_manifest_sha256"],
            "archive_public_test_sha256": data_manifest["source_sha256"]["test.csv"],
            "prediction_sha256": {method: bundle["runs"][method]["prediction_sha256"] for method in METHODS},
            "warning": "Test results explain the frozen experiments; do not reuse them for tuning.",
        },
        "runs": {method: score_records(references, predictions[method]) for method in METHODS},
    }
    _write_json(Path(output), result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("validation", help="Score only the prepared validation split.")
    validation.add_argument("--references", required=True)
    validation.add_argument("--predictions", required=True)
    validation.add_argument("--split-manifest", required=True)
    validation.add_argument("--output", required=True)
    final = commands.add_parser("final", help="Score four complete frozen test runs.")
    final.add_argument("--frozen", required=True, help="Path to final_bundle.json.")
    final.add_argument("--answer-archive", required=True)
    final.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "validation":
        result = evaluate_validation(args.references, args.predictions, args.split_manifest, args.output)
        print(json.dumps(result["metrics"], indent=2, ensure_ascii=False, allow_nan=False))
    else:
        result = evaluate_final(args.frozen, args.answer_archive, args.output)
        print(json.dumps({method: data["metrics"] for method, data in result["runs"].items()}, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
