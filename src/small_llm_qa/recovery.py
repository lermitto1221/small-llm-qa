"""Narrow inference-only repair with immutable, verified SFT provenance.

This bridge is specific to the recorded AMP run. It never trains, rewrites
training evidence, or opens held-out answers. Only generation may change in
modeling.py; every other training dependency must remain byte-identical.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import uuid

from .io_utils import canonical_hash, file_hash, tree_hash, write_json
from .modeling import dependency_versions

PARENT_CODE_SHA256 = "9f5f5834b5efa6fbe23984815722707ee38604b19366adb69aabce1283a75f1e"
PARENT_SUMMARY_SHA256 = "571a35564fa7dfa4777ad1922bae4e33f9aaea323ef0bc67a73eb01fe587f783"
PARENT_ADAPTER_SHA256 = "a21a6ce4c49256a7a6bd134ec0f0a18defafd3c4d67a078f9f5ef7bf67418a3e"
RECOVERY_ID = "greedy_inference_v1"
FROZEN_FIELDS = {"experiment_configuration_frozen", "selected_n", "selection_sha256", "adapter_path",
                 "adapter_sha256", "training_summary_sha256", "dependency_status"}


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _regular_tree(path):
    path = Path(path)
    if not path.is_dir() or path.is_symlink() or any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError(f"A regular directory without links is required: {path}")
    return path


def _source_files(directory):
    directory = _regular_tree(directory)
    files = {path.name: path.read_bytes() for path in sorted(directory.glob("*.py"))}
    if "__init__.py" not in files or "training.py" not in files or "modeling.py" not in files:
        raise ValueError("Incomplete source package")
    return files


def _source_hash(directory):
    import hashlib
    return canonical_hash({name: hashlib.sha256(value).hexdigest() for name, value in _source_files(directory).items()})


def _modeling_without_generation(content):
    tree = ast.parse(content.decode("utf-8"))
    functions = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and item.name == "generate_answer"]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ValueError("Expected exactly one synchronous generate_answer function")
    tree.body.remove(functions[0])
    return ast.dump(tree, include_attributes=False)


def _verify_source(parent_source, current_source):
    old, new = _source_files(parent_source), _source_files(current_source)
    if _source_hash(parent_source) != PARENT_CODE_SHA256:
        raise ValueError("Parent source does not match the recorded completed SFT run")
    if set(new) != set(old) | {"recovery.py"} or "recovery.py" in old:
        raise ValueError("Recovery may add only recovery.py to the original package")
    for name in old:
        if name == "modeling.py":
            if _modeling_without_generation(old[name]) != _modeling_without_generation(new[name]):
                raise ValueError("Model loading or training code changed outside generate_answer")
        elif name != "workflow.py" and old[name] != new[name]:
            raise ValueError(f"Training/shared dependency changed: {name}")
    return _source_hash(current_source)


def _reference(base, metadata, name, *, directory=False):
    value = metadata.get(name + "_path")
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"Recovery {name}_path must be a relative path")
    path = (base / value).resolve()
    if directory:
        _regular_tree(path)
        actual = _source_hash(path) if name == "parent_source" else tree_hash(path)
    else:
        if path.suffix != ".json" or path.is_symlink():
            raise ValueError(f"Expected regular JSON evidence: {path}")
        actual = file_hash(path)
    if actual != metadata.get(name + "_sha256"):
        raise ValueError(f"Recovery evidence changed: {name}")
    return path


def _verify(config, config_path, prepared, current_source, *, require_copies=True):
    base, prepared = Path(config_path).resolve().parent, Path(prepared).resolve()
    metadata = config.get("inference_recovery")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("recovery_id") != RECOVERY_ID:
        raise ValueError("Missing or unsupported inference recovery provenance")
    if metadata.get("training_reused") is not True or metadata.get("training_rerun") is not False:
        raise ValueError("Recovery must reuse completed training without rerunning it")
    paths = {name: _reference(base, metadata, name, directory=name in {"parent_source", "parent_adapter"})
             for name in ("parent_config", "parent_frozen", "parent_summary", "parent_training_manifest", "parent_source", "parent_adapter")}
    original = _json(paths["parent_config"])
    frozen = _json(paths["parent_frozen"])
    summary = _json(paths["parent_summary"])
    training_manifest = _json(paths["parent_training_manifest"])
    if metadata["parent_source_sha256"] != PARENT_CODE_SHA256 or metadata["parent_summary_sha256"] != PARENT_SUMMARY_SHA256 or metadata["parent_adapter_sha256"] != PARENT_ADAPTER_SHA256:
        raise ValueError("Recovery evidence does not match the recorded original run")
    parent_runs = paths["parent_config"].parent
    expected_paths = {"parent_frozen": parent_runs / "frozen_config.json",
                      "parent_summary": parent_runs / "sft" / "training_summary.json",
                      "parent_training_manifest": parent_runs / "sft" / "training_manifest.json",
                      "parent_adapter": parent_runs / "sft" / "adapter"}
    if paths["parent_config"].name != "run_config.json" or any(paths[name] != path.resolve() for name, path in expected_paths.items()):
        raise ValueError("Parent training artifacts do not belong to the same original run")
    if base == parent_runs or parent_runs in base.parents or base in parent_runs.parents:
        raise ValueError("Recovery must use a separate run directory")
    if original.get("experiment_configuration_frozen") is not False or "inference_recovery" in original:
        raise ValueError("Parent must be the original unfrozen run configuration")
    if frozen.get("experiment_configuration_frozen") is not True:
        raise ValueError("Original frozen configuration is required")
    if {key: value for key, value in frozen.items() if key not in FROZEN_FIELDS} != {key: value for key, value in original.items() if key not in FROZEN_FIELDS}:
        raise ValueError("Parent frozen configuration changed experiment settings")
    if frozen.get("training_summary_sha256") != PARENT_SUMMARY_SHA256 or frozen.get("adapter_sha256") != PARENT_ADAPTER_SHA256:
        raise ValueError("Parent frozen configuration does not authenticate the training evidence")
    if (paths["parent_frozen"].parent / frozen.get("adapter_path", "")).resolve() != paths["parent_adapter"]:
        raise ValueError("Parent frozen adapter path disagrees with training evidence")
    code_hash = _verify_source(paths["parent_source"], current_source)
    if original.get("source_code_sha256") != PARENT_CODE_SHA256 or config.get("source_code_sha256") != code_hash:
        raise ValueError("Original/current source hash differs from its configuration")
    expected = {**original, "source_code_sha256": code_hash, "inference_recovery": metadata}
    frozen_current = config.get("experiment_configuration_frozen") is True
    if frozen_current:
        if {key: value for key, value in config.items() if key not in FROZEN_FIELDS} != {key: value for key, value in expected.items() if key not in FROZEN_FIELDS}:
            raise ValueError("Recovery changed settings beyond the inference repair")
        if config.get("training_summary_sha256") != PARENT_SUMMARY_SHA256 or config.get("adapter_sha256") != PARENT_ADAPTER_SHA256:
            raise ValueError("Recovery frozen training evidence differs")
    elif config != expected:
        raise ValueError("Recovery config may differ only in source hash and inference_recovery")
    data_manifest_path = prepared / "manifest.json"
    data_hash = file_hash(data_manifest_path)
    manifest = _json(data_manifest_path)
    if original.get("data_manifest_sha256") != data_hash:
        raise ValueError("Prepared split manifest differs from the original run")
    for name, digest in manifest["artifact_sha256"].items():
        path = (prepared / name).resolve()
        if prepared not in path.parents or path.name.casefold() == "test_answer.csv":
            raise ValueError("Prepared manifest contains a forbidden artifact path")
        if file_hash(path) != digest:
            raise ValueError(f"Prepared artifact changed: {name}")
    versions = original.get("runtime_versions")
    if not isinstance(versions, dict) or not versions or versions != dependency_versions():
        raise ValueError("Current dependency versions differ from original SFT")
    if summary.get("completed") is not True or summary.get("smoke") is not False or summary.get("status") != "complete":
        raise ValueError("Only completed full SFT can be reused")
    shared = {"config_sha256": file_hash(paths["parent_config"]), "code_sha256": PARENT_CODE_SHA256,
              "data_manifest_sha256": data_hash, "dependencies": versions}
    if any(summary.get(key) != value or training_manifest.get(key) != value for key, value in shared.items()):
        raise ValueError("Original training provenance differs from original config/source/data/dependencies")
    if training_manifest.get("smoke") is not False or training_manifest.get("model") != original["model"] or training_manifest.get("training") != original["training"]:
        raise ValueError("Original training model/settings do not match the parent configuration")
    for split in ("train", "validation"):
        digest = file_hash(prepared / f"{split}.csv")
        if summary.get(f"{split}_csv_sha256") != digest or training_manifest.get(f"{split}_sha256") != digest:
            raise ValueError("Original training CSV hashes differ from the prepared split")
        if summary.get(f"{split}_rows") != len(manifest[f"{split}_ids"]):
            raise ValueError("Original training did not cover its complete split")
    if summary.get("adapter_sha256") != PARENT_ADAPTER_SHA256:
        raise ValueError("Completed adapter differs from the original summary")
    if any(not (paths["parent_adapter"] / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")):
        raise ValueError("Original adapter is incomplete")
    copied_summary, copied_adapter = base / "sft" / "training_summary.json", base / "sft" / "adapter"
    if require_copies:
        if file_hash(copied_summary) != PARENT_SUMMARY_SHA256 or tree_hash(_regular_tree(copied_adapter)) != PARENT_ADAPTER_SHA256:
            raise ValueError("Copied SFT evidence differs from the immutable original")
        if frozen_current and (base / config.get("adapter_path", "")).resolve() != copied_adapter.resolve():
            raise ValueError("Recovery frozen adapter must be the verified copy")
    return {"parent_config_path": str(paths["parent_config"]), "parent_source_path": str(paths["parent_source"]),
            "original_training_summary_path": str(paths["parent_summary"]),
            "training_summary_path": str(copied_summary), "training_summary_sha256": PARENT_SUMMARY_SHA256,
            "adapter_path": str(copied_adapter), "adapter_sha256": PARENT_ADAPTER_SHA256,
            "parent_code_sha256": PARENT_CODE_SHA256, "current_code_sha256": code_hash,
            "training_reused": True, "training_rerun": False}


def verify_inference_recovery(config_path, prepared, current_source=None):
    """Verify the bridge before any inference; return original/copied evidence paths."""
    return _verify(_json(config_path), config_path, prepared,
                   Path(current_source) if current_source is not None else Path(__file__).parent)


def prepare_inference_recovery(parent_runs, new_runs, prepared, parent_source, current_source=None):
    """Copy unchanged SFT artifacts and write a separate immutable inference config."""
    parent_runs, new_runs = Path(parent_runs).resolve(), Path(new_runs).resolve()
    current_source = Path(current_source) if current_source is not None else Path(__file__).parent
    parent_config = parent_runs / "run_config.json"
    parent = _json(parent_config)
    references = {"parent_config": parent_config, "parent_frozen": parent_runs / "frozen_config.json",
                  "parent_summary": parent_runs / "sft" / "training_summary.json",
                  "parent_training_manifest": parent_runs / "sft" / "training_manifest.json",
                  "parent_source": Path(parent_source).resolve(), "parent_adapter": parent_runs / "sft" / "adapter"}
    metadata = {"schema_version": 1, "recovery_id": RECOVERY_ID,
                "reason": "Recompute every inference result with explicit greedy generate overrides; reuse unchanged completed SFT.",
                "training_reused": True, "training_rerun": False}
    for name, path in references.items():
        metadata[name + "_path"] = os.path.relpath(path, new_runs).replace("\\", "/")
        metadata[name + "_sha256"] = (_source_hash(path) if name == "parent_source" else
                                      tree_hash(_regular_tree(path)) if name == "parent_adapter" else file_hash(path))
    config = {**parent, "source_code_sha256": _source_hash(current_source), "inference_recovery": metadata}
    config_path = new_runs / "run_config.json"
    evidence = _verify(config, config_path, prepared, current_source, require_copies=False)
    if config_path.exists() and _json(config_path) != config:
        raise ValueError("Existing recovery configuration differs; use a fresh run directory")
    new_runs.mkdir(parents=True, exist_ok=True)
    copied_adapter, copied_summary = Path(evidence["adapter_path"]), Path(evidence["training_summary_path"])
    copied_adapter.parent.mkdir(exist_ok=True)
    if copied_adapter.exists():
        if tree_hash(_regular_tree(copied_adapter)) != PARENT_ADAPTER_SHA256:
            raise ValueError("Existing copied adapter differs; nothing was overwritten")
    else:
        # A Drive interruption leaves only an isolated staging directory, so a
        # retry cannot mistake a partial copy for the immutable final adapter.
        staged = copied_adapter.with_name(".adapter-copy-" + uuid.uuid4().hex)
        shutil.copytree(references["parent_adapter"], staged)
        if tree_hash(_regular_tree(staged)) != PARENT_ADAPTER_SHA256:
            raise ValueError("Staged adapter copy failed integrity verification")
        staged.replace(copied_adapter)
    if copied_summary.exists():
        if file_hash(copied_summary) != PARENT_SUMMARY_SHA256:
            raise ValueError("Existing copied training summary differs; nothing was overwritten")
    else:
        staged = copied_summary.with_name(".summary-copy-" + uuid.uuid4().hex + ".json")
        shutil.copyfile(references["parent_summary"], staged)
        if file_hash(staged) != PARENT_SUMMARY_SHA256:
            raise ValueError("Staged training summary copy failed integrity verification")
        staged.replace(copied_summary)
    _verify(config, config_path, prepared, current_source)
    write_json(config_path, config, immutable=True)
    return {"config_path": str(config_path), **evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--parent-runs", required=True)
    prepare.add_argument("--new-runs", required=True)
    prepare.add_argument("--parent-source", required=True)
    prepare.add_argument("--prepared", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--config", required=True)
    verify.add_argument("--prepared", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_inference_recovery(args.parent_runs, args.new_runs, args.prepared, args.parent_source)
    else:
        result = verify_inference_recovery(args.config, args.prepared)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
