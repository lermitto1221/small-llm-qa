"""Check a public source release without loading models or opening private data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


MAX_FILE_BYTES = 10 * 1024 * 1024
METHODS = ("baseline", "random", "similar", "sft")
TASK_COUNTS = {"qa": 330, "rag": 350, "math": 327}
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache", "venv", "env"}
PRIVATE_DIRS = {"data", "artifacts", "exports", "output", "runs", "checkpoints"}
FORBIDDEN_SUFFIXES = {
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".tgz", ".whl",
    ".pem", ".key", ".p12", ".pfx", ".ppk", ".keystore", ".jks",
    ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".gguf", ".onnx", ".h5", ".hdf5", ".npy", ".npz",
    ".csv", ".tsv", ".jsonl", ".parquet", ".arrow", ".sqlite", ".sqlite3", ".db",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".ipynb", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".sh", ".ps1", ".csv", ".tsv"}
SYNTHETIC_EXAMPLES = {"examples/demo_references.json", "examples/demo_predictions.json"}
AGGREGATE_TABLES = {"results/summary.csv", "results/loss_history.csv"}
SENSITIVE_PATTERNS = {
    "Hugging Face credential": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "personal Drive or Colab link": re.compile(r"https?://(?:drive\.google\.com/(?:file/d/|drive/folders/|open\?)|colab\.research\.google\.com/drive/)", re.I),
    "Windows user directory": re.compile(r"\b[A-Za-z]:[\\/]+Users[\\/]+", re.I),
}
IDENTITY_NUMBER = re.compile(r"\b(?:student[ _-]*(?:id|number)|sid|eid)\b[\"']?\s*[:=]\s*[\"']?(\d{5,16}|[A-Za-z][A-Za-z0-9._-]*\d[A-Za-z0-9._-]*)", re.I)
IDENTITY_TEXT = re.compile(r"\b(?:student[ _-]*(?:id|number)|sid|eid)\b[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9][A-Za-z0-9._-]{4,39})", re.I)
INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[\"'][^\"']*[\"'])?\s*\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]\n]+\]:\s*<?([^\s>]+)>?", re.M)
HTML_LINK = re.compile(r"\b(?:href|src)\s*=\s*[\"']([^\"']+)[\"']", re.I)


def ignored_directory(name):
    return name in IGNORED_DIRS or name.startswith(".venv") or name.endswith(".egg-info")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError("non-finite JSON constant")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check(root, write_manifest=False):
    errors, files = [], []
    counts = {"python_files": 0, "notebooks": 0, "local_links": 0}
    def fail(message):
        errors.append(message)
    for directory, dirs, names in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(dirs):
            path = Path(directory) / name
            if ignored_directory(name):
                continue
            if path.is_symlink():
                fail(f"Symbolic-link directory: {path.relative_to(root).as_posix()}")
                continue
            if name in PRIVATE_DIRS or name.startswith("runs_"):
                fail(f"Private-data/runtime directory: {path.relative_to(root).as_posix()}")
            kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                fail(f"Symbolic-link file: {relative}")
                continue
            files.append(path)
            lower = name.lower()
            if (lower == ".env" or lower.startswith(".env.") or lower in {"id_rsa", "id_dsa", "id_ecdsa", "id_ed25519", "credentials.json", "credentials.ini", "secrets.json", "service_account.json", "service-account.json", "client_secret.json"}
                    or (path.suffix.lower() in FORBIDDEN_SUFFIXES and relative not in AGGREGATE_TABLES)):
                fail(f"Forbidden file type or credential filename: {relative}")
            if relative not in SYNTHETIC_EXAMPLES and (lower in {"train.csv", "test.csv", "validation.csv", "validation_tune.csv", "test_answer.csv", "train_sft.jsonl", "validation_sft.jsonl"}
                    or re.search(r"(?:^|_)(?:predictions|answers|test_answers)(?:\.|_)", lower)):
                fail(f"Dataset, answer or prediction export: {relative}")
            if path.stat().st_size > MAX_FILE_BYTES:
                fail(f"File exceeds {MAX_FILE_BYTES} bytes: {relative}")

    texts, parsed_json = {}, {}
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "NOTICE", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError):
            fail(f"Cannot read text as UTF-8: {relative}")
            continue
        texts[path] = text
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                fail(f"Potential {label}: {relative}")
        identity = IDENTITY_NUMBER if path.suffix == ".py" else IDENTITY_TEXT
        if any(len(match.group(1)) >= 5 for match in identity.finditer(text)):
            fail(f"Potential student/account identifier: {relative}")
        if relative in AGGREGATE_TABLES:
            expected = (["method", "qa_em", "qa_token_f1", "rag_em", "rag_token_f1", "math_accuracy", "composite_score"]
                        if path.name == "summary.csv" else ["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"])
            try:
                reader = csv.DictReader(io.StringIO(text))
                if reader.fieldnames != expected:
                    raise ValueError("unexpected columns")
                for row in reader:
                    if set(row) != set(expected) or any(value is None for value in row.values()):
                        raise ValueError("invalid row")
                    for key, value in row.items():
                        if key == "method":
                            if value not in METHODS:
                                raise ValueError("invalid method")
                        elif value and not math.isfinite(float(value)):
                            raise ValueError("non-finite number")
            except (csv.Error, ValueError, TypeError):
                fail(f"Aggregate CSV contains unexpected or nonnumeric content: {relative}")
        if path.suffix == ".py":
            counts["python_files"] += 1
            try:
                compile(text, relative, "exec")
            except (SyntaxError, ValueError):
                fail(f"Python syntax error: {relative}")
        if path.suffix in {".json", ".ipynb"}:
            try:
                parsed_json[path] = strict_json(text)
            except (ValueError, TypeError):
                fail(f"Invalid JSON: {relative}")
        if path.suffix == ".ipynb" and path in parsed_json:
            counts["notebooks"] += 1
            notebook = parsed_json[path]
            if not isinstance(notebook, dict) or notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
                fail(f"Expected a v4 notebook: {relative}")
                continue
            for index, cell in enumerate(notebook["cells"], 1):
                if not isinstance(cell, dict):
                    fail(f"Invalid notebook cell {index}: {relative}")
                    continue
                if cell.get("cell_type") == "code":
                    if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                        fail(f"Notebook cell {index} contains execution state: {relative}")
                    try:
                        source = cell.get("source", "")
                        compile("".join(source) if isinstance(source, list) else source, f"{relative}:cell{index}", "exec")
                    except (SyntaxError, ValueError, TypeError):
                        fail(f"Notebook code syntax error in cell {index}: {relative}")

    def check_export_keys(value, relative):
        if isinstance(value, dict):
            for key, nested in value.items():
                if key.lower() in {"reference_answers", "test_answers", "accepted_answers", "question", "context", "answer"}:
                    fail(f"Per-example content in JSON export: {relative}")
                    return
                check_export_keys(nested, relative)
        elif isinstance(value, list):
            for nested in value:
                check_export_keys(nested, relative)
    for path, value in parsed_json.items():
        relative = path.relative_to(root).as_posix()
        if relative in SYNTHETIC_EXAMPLES:
            if not isinstance(value, list) or not value or any(not isinstance(row, dict) or not str(row.get("Sample_id", "")).startswith("DEMO_") for row in value):
                fail(f"Synthetic examples must contain only explicit DEMO_ records: {relative}")
        elif path.suffix == ".json" and "tests" not in path.relative_to(root).parts:
            check_export_keys(value, relative)

    if root / "README.md" not in texts:
        fail("Missing README.md")
    for path, text in texts.items():
        if path.suffix != ".md" or not (path.name.lower().startswith("readme") or "docs" in path.relative_to(root).parts):
            continue
        without_fences = re.sub(r"(?ms)^\s*```.*?^\s*```\s*$", "", text)
        links = [a or b for a, b in INLINE_LINK.findall(without_fences)]
        links += REFERENCE_LINK.findall(without_fences) + HTML_LINK.findall(without_fences)
        for link in links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            target = (root / unquote(parsed.path).lstrip("/") if parsed.path.startswith("/") else path.parent / unquote(parsed.path)).resolve()
            counts["local_links"] += 1
            if not target.is_relative_to(root) or not target.exists():
                fail(f"Broken or escaping local link in {path.relative_to(root).as_posix()}")

    def result_file(name):
        path = root / "results" / name
        value = parsed_json.get(path)
        if not isinstance(value, dict):
            fail(f"Missing or invalid results/{name}")
            return {}
        return value
    metrics = result_file("metrics.json")
    statistics = result_file("statistics.json")
    training = result_file("training_summary.json")
    selection = result_file("validation_selection.json")
    for label, collection in (("metrics", metrics), ("statistics", statistics)):
        if set(collection) != set(METHODS):
            fail(f"results/{label}.json must contain exactly the four methods")
    for method in METHODS:
        metric, stats = metrics.get(method, {}), statistics.get(method, {})
        if not isinstance(metric, dict) or not isinstance(stats, dict):
            fail(f"Invalid metric/statistics object: {method}")
            continue
        if metric.get("total_count") != 1007 or stats.get("count") != 1007:
            fail(f"Expected 1007 evaluated examples: {method}")
        primary = []
        for task, count in TASK_COUNTS.items():
            values = metric.get(task, {})
            if not isinstance(values, dict) or values.get("count") != count:
                fail(f"Incorrect task count: {method}/{task}")
                continue
            names = ("numeric_accuracy",) if task == "math" else ("exact_match", "token_f1")
            for name in names:
                if not number(values.get(name)) or not 0 <= values[name] <= 1:
                    fail(f"Invalid metric: {method}/{task}/{name}")
            value = values.get("numeric_accuracy" if task == "math" else "token_f1")
            if number(value):
                primary.append(value)
        if len(primary) != 3 or not number(metric.get("macro_score")) or not math.isclose(metric["macro_score"], sum(primary) / 3, rel_tol=0, abs_tol=1e-12):
            fail(f"Macro score does not equal mean(QA F1, RAG F1, math accuracy): {method}")
    scores = selection.get("scores", {})
    means = {}
    if isinstance(scores, dict):
        for key, value in scores.items():
            if not str(key).isdigit() or int(key) <= 0 or not isinstance(value, dict) or any(not number(value.get(method)) or not 0 <= value[method] <= 1 for method in ("random", "similar")):
                fail("Invalid validation n-selection scores")
                continue
            means[int(key)] = (value["random"] + value["similar"]) / 2
    selected = selection.get("selected_n")
    best_n = min((n for n in means if math.isclose(means[n], max(means.values()), rel_tol=0, abs_tol=1e-12)), default=None)
    if not means or not isinstance(selected, int) or isinstance(selected, bool) or selected != best_n:
        fail("selected_n does not maximize the common validation score (smaller n breaks ties)")
    rule = selection.get("rule", selection.get("selection_rule"))
    if not isinstance(rule, str) or not rule.strip():
        fail("Missing validation selection rule")
    if training and (not number(training.get("training_loss")) or not number(training.get("global_step"))):
        fail("Training summary must report finite loss and global_step")

    manifest_path = root / "release_manifest.json"
    inventory = {path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in sorted(files) if path != manifest_path and path.stat().st_size <= MAX_FILE_BYTES}
    manifest_action = "not_written"
    if write_manifest:
        if not errors:
            manifest_path.write_text(json.dumps({"schema_version": 1, "files": inventory}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            manifest_action = "written"
    else:
        manifest = parsed_json.get(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("files") != inventory:
            fail("Release manifest is missing, invalid, or differs from the complete file inventory")
        else:
            manifest_action = "verified"
    return {"status": "failed" if errors else "passed", "files": len(inventory), **counts,
            "manifest": manifest_action, "errors": list(dict.fromkeys(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Create the manifest only after all release checks pass.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Release root (defaults to this script's parent project).")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    try:
        if not root.is_dir():
            raise ValueError("Release root is not a directory")
        report = check(root, arguments.write_manifest)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"status": "failed", "errors": [f"Release check could not complete ({type(error).__name__})"]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
