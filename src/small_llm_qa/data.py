"""Answer-blind, deterministic preparation of the course QA dataset.

Only allowlisted archive members are opened. ``test_answer.csv`` is never
extracted, parsed, or hashed by this module. No third-party packages are needed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TASKS = ("math", "qa", "rag")
INPUT_FIELDS = ("Sample_id", "Task", "Question", "Context")
LABEL_FIELDS = INPUT_FIELDS + ("Answer", "Accepted_answers")
SOURCE_FIELDS = ("Sample_id", "Source", "Source_id", "Source_urls")
SYSTEM_PROMPT = (
    "Answer the question concisely. For a context question, use only the supplied "
    "passages and return UNANSWERABLE if they do not support an answer. For a math "
    "question, return only the final number."
)
ROOT_ALLOWLIST = {"train.csv", "test.csv", "sources.csv", "train_sft.jsonl", "README.txt"}
MAX_MEMBER_BYTES = 100 * 1024 * 1024
MAX_TOTAL_BYTES = 300 * 1024 * 1024


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read UTF-8 CSV strictly, rejecting malformed headers and field counts."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)) or any(not x for x in fields):
            raise ValueError(f"Invalid/duplicate CSV headers: {Path(path).name}")
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {number}: {Path(path).name}")
            rows.append(dict(row))
    return rows


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _extract_allowed(archive: Path, destination: Path) -> tuple[dict[str, str], list[str]]:
    """Validate paths, then stream only selected members; no extractall call."""
    selected: list[tuple[zipfile.ZipInfo, str]] = []
    skipped: list[str] = []
    names: set[str] = set()
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            raw = member.filename
            parts = PurePosixPath(raw).parts
            if (
                not raw or "\\" in raw or raw.startswith("/") or not parts
                or any(p in {".", ".."} or ":" in p for p in parts)
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {raw!r}")
            if member.is_dir():
                continue
            if raw.casefold() in names:
                raise ValueError(f"Duplicate archive member: {raw}")
            names.add(raw.casefold())
            if parts[-1].casefold() == "test_answer.csv":
                skipped.append(raw)
                continue
            allowed = raw in ROOT_ALLOWLIST or (parts[0] == "LICENSES" and len(parts) >= 2)
            if not allowed:
                skipped.append(raw)
                continue
            total += member.file_size
            if member.file_size > MAX_MEMBER_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("Allowlisted archive data exceeds extraction size limit")
            selected.append((member, raw))
        selected_names = {name for _, name in selected}
        missing = ROOT_ALLOWLIST - selected_names
        if missing:
            raise ValueError(f"Required archive files are missing: {sorted(missing)}")
        if not any(name.startswith("LICENSES/") for name in selected_names):
            raise ValueError("Dataset archive must include its LICENSES/ attribution files")
        hashes: dict[str, str] = {}
        for member, name in selected:
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A fresh temporary directory prevents pre-existing symlink traversal.
            with bundle.open(member, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            hashes[name] = sha256_file(target)
    return dict(sorted(hashes.items())), sorted(skipped)


def _validate_rows(rows: list[dict[str, str]], labelled: bool, filename: str) -> dict[str, dict[str, str]]:
    expected = LABEL_FIELDS if labelled else INPUT_FIELDS
    if not rows:
        raise ValueError(f"Empty dataset: {filename}")
    if tuple(rows[0]) != expected:
        raise ValueError(f"Unexpected CSV columns in {filename}; expected {expected}")
    by_id: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["Sample_id"]
        if not sample_id or sample_id.strip() != sample_id or sample_id in by_id:
            raise ValueError(f"Empty, padded, or duplicate Sample_id in {filename}: {sample_id!r}")
        if row["Task"] not in TASKS or not row["Question"].strip():
            raise ValueError(f"Invalid Task or empty Question: {sample_id}")
        if row["Task"] == "rag" and not row["Context"].strip():
            raise ValueError(f"RAG sample has no supplied Context: {sample_id}")
        if row["Task"] != "rag" and row["Context"]:
            raise ValueError(f"Non-RAG sample has unexpected Context: {sample_id}")
        if labelled:
            try:
                accepted = json.loads(row["Accepted_answers"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid Accepted_answers JSON: {sample_id}") from error
            if (
                not row["Answer"].strip() or not isinstance(accepted, list) or not accepted
                or any(not isinstance(answer, str) or not answer.strip() for answer in accepted)
                or row["Answer"] not in accepted
            ):
                raise ValueError(f"Invalid reference answers or primary Answer missing from aliases: {sample_id}")
        by_id[sample_id] = row
    return by_id


def format_user_message(row: dict[str, str]) -> str:
    """The exact user-message convention in the supplied course JSONL."""
    text = f"Task: {row['Task']}\nQuestion: {row['Question']}"
    if row["Context"]:
        text += f"\nContext:\n{row['Context']}"
    return text


def _validate_sft(path: Path, train: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {number}") from error
            if not isinstance(record, dict) or set(record) != {"Sample_id", "messages"}:
                raise ValueError(f"Invalid SFT record structure at line {number}")
            sample_id = record["Sample_id"]
            if not isinstance(sample_id, str) or sample_id not in train or sample_id in found:
                raise ValueError(f"Duplicate/unknown SFT Sample_id at line {number}: {sample_id!r}")
            expected = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(train[sample_id])},
                {"role": "assistant", "content": train[sample_id]["Answer"]},
            ]
            if record["messages"] != expected:
                raise ValueError(f"CSV / JSONL content mismatch: {sample_id}")
            found[sample_id] = record
    missing = set(train) - set(found)
    if missing:
        raise ValueError(f"SFT is missing {len(missing)} training IDs")
    return found


def _validate_sources(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = read_csv(path)
    if not rows or tuple(rows[0]) != SOURCE_FIELDS:
        raise ValueError("Unexpected sources.csv columns")
    by_id = {}
    for row in rows:
        sample_id = row["Sample_id"]
        if sample_id in by_id or sample_id not in expected_ids:
            raise ValueError(f"Duplicate/unknown source Sample_id: {sample_id}")
        try:
            urls = json.loads(row["Source_urls"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Source_urls JSON: {sample_id}") from error
        if (
            not row["Source"] or not row["Source_id"] or not isinstance(urls, list)
            or any(not isinstance(url, str) or not url.strip() for url in urls)
        ):
            raise ValueError(f"Invalid source metadata: {sample_id}")
        by_id[sample_id] = {**row, "urls": urls}
    if set(by_id) != expected_ids:
        raise ValueError("sources.csv does not cover all and only train/test IDs")
    return by_id


def _split(train: dict[str, dict[str, str]], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = random.Random(seed)
    training, validation, tune = [], [], []
    for task in TASKS:
        ids = sorted(sample_id for sample_id, row in train.items() if row["Task"] == task)
        rng.shuffle(ids)
        count = round(len(ids) * 0.1)
        if count < 60:
            raise ValueError(f"Task {task} has only {count} validation rows; need 60 for fixed tuning set")
        validation.extend(ids[:count])
        tune.extend(ids[:60])
        training.extend(ids[count:])
    return sorted(training), sorted(validation), sorted(tune)


def _normal_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _groups(rows: Iterable[dict[str, str]], question_only: bool = False) -> dict[tuple[str, ...], list[str]]:
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows:
        parts = [row["Task"], _normal_text(row["Question"])]
        if not question_only:
            parts.append(_normal_text(row["Context"]))
        groups[tuple(parts)].append(row["Sample_id"])
    return dict(groups)


def _duplicate_summary(rows: list[dict[str, str]], question_only: bool) -> dict[str, Any]:
    duplicates = sorted(sorted(ids) for ids in _groups(rows, question_only).values() if len(ids) > 1)
    return {
        "groups": len(duplicates), "samples": sum(map(len, duplicates)),
        "extra_copies": sum(len(ids) - 1 for ids in duplicates), "example_groups": duplicates[:20],
    }


def _cross_duplicate_summary(left: list[dict[str, str]], right: list[dict[str, str]], question_only: bool) -> dict[str, Any]:
    left_groups, right_groups = _groups(left, question_only), _groups(right, question_only)
    shared = sorted(set(left_groups) & set(right_groups))
    return {
        "shared_keys": len(shared),
        "left_samples": sum(len(left_groups[key]) for key in shared),
        "right_samples": sum(len(right_groups[key]) for key in shared),
        "example_id_pairs": [
            {"left": sorted(left_groups[key]), "right": sorted(right_groups[key])} for key in shared[:20]
        ],
    }


def _source_overlap(left: list[str], right: list[str], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def source_values(ids: list[str], kind: str) -> dict[Any, set[str]]:
        values: dict[Any, set[str]] = defaultdict(set)
        for sample_id in ids:
            source = sources[sample_id]
            keys = ([source["Source"]] if kind == "source" else
                    [(source["Source"], source["Source_id"])] if kind == "source_id" else source["urls"])
            for key in keys:
                values[key].add(sample_id)
        return values
    result = {}
    for kind in ("source", "source_id", "source_url"):
        lv, rv = source_values(left, kind), source_values(right, kind)
        shared = sorted(set(lv) & set(rv))
        result[kind] = {
            "shared_keys": len(shared),
            "left_samples": len(set().union(*(lv[key] for key in shared))) if shared else 0,
            "right_samples": len(set().union(*(rv[key] for key in shared))) if shared else 0,
            "examples": [list(key) if isinstance(key, tuple) else key for key in shared[:20]],
        }
    return result


def _task_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["Task"] for row in rows)
    return {task: counts[task] for task in TASKS}


def _audit_text(audit: dict[str, Any]) -> str:
    lines = [
        "数据准备与隔离检查（未读取测试答案）", "",
        f"随机种子：{audit['seed']}；按任务分层划分约 10% 验证集。",
        "CSV 与 SFT JSONL 的全部 ID、system、user、assistant 内容逐条一致。",
        "验证 ID 与训练/示例池完全隔离；固定调参集每类 60 条，共 180 条。",
        "所有来源与许可证原样保留。test_answer.csv 未打开、提取或计算内容哈希。", "",
    ]
    for name, counts in audit["counts"].items():
        lines.append(f"{name}: {sum(counts.values())}；" + ", ".join(f"{key}={value}" for key, value in counts.items()))
    lines.extend(["", "内容检查使用任务 + 问题（+ 段落），统一大小写与空白；不使用测试答案。"])
    for name, summary in audit["duplicates"].items():
        lines.append(f"{name}: " + ", ".join(f"{kind}重复组={value['groups']}" for kind, value in summary.items()))
    for name, summary in audit["cross_split"].items():
        lines.append(f"{name}: 问题重叠键={summary['question']['shared_keys']}；问题+段落重叠键={summary['content']['shared_keys']}")
        overlap = summary["sources"]
        lines.append("  来源重叠：" + ", ".join(f"{kind}={value['shared_keys']}" for kind, value in overlap.items()))
    lines.extend(["", *audit["interpretation"], "", "详细 ID 示例及哈希见 audit.json 和 manifest.json。GPU token 预算尚待 Colab 实测。", ""])
    return "\n".join(lines)


def prepare_dataset(archive: str | Path, output: str | Path, seed: int = 42) -> dict[str, Any]:
    """Prepare immutable splits. Identical reruns are no-ops; changes fail closed.

    A changed source, seed, split algorithm, or previously prepared artifact
    requires a new output directory, preserving earlier experiment provenance.
    """
    archive, output = Path(archive), Path(output)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output.parent) as staging:
        extracted, prepared = Path(staging) / "source", Path(staging) / "prepared"
        extracted.mkdir()
        prepared.mkdir()
        source_hashes, skipped = _extract_allowed(archive, extracted)
        original_train, test_rows = read_csv(extracted / "train.csv"), read_csv(extracted / "test.csv")
        train = _validate_rows(original_train, True, "train.csv")
        test = _validate_rows(test_rows, False, "test.csv")
        if set(train) & set(test):
            raise ValueError("Training and test Sample_id sets overlap")
        sft = _validate_sft(extracted / "train_sft.jsonl", train)
        sources = _validate_sources(extracted / "sources.csv", set(train) | set(test))
        train_ids, validation_ids, tune_ids = _split(train, seed)
        test_ids = sorted(test)
        subsets = {
            "train": [train[key] for key in train_ids],
            "validation": [train[key] for key in validation_ids],
            "validation_tune": [train[key] for key in tune_ids],
            "test": [test[key] for key in test_ids],
        }
        for name, rows in subsets.items():
            _write_csv(prepared / f"{name}.csv", rows, INPUT_FIELDS if name == "test" else LABEL_FIELDS)
        _write_jsonl(prepared / "train_sft.jsonl", [sft[key] for key in train_ids])
        _write_jsonl(prepared / "validation_sft.jsonl", [sft[key] for key in validation_ids])
        shutil.copyfile(extracted / "sources.csv", prepared / "sources.csv")
        shutil.copyfile(extracted / "README.txt", prepared / "DATASET_README.txt")
        shutil.copytree(extracted / "LICENSES", prepared / "LICENSES")
        audit: dict[str, Any] = {
            "schema_version": 1, "seed": seed, "test_answers_opened": False,
            "skipped_archive_members": skipped,
            "csv_sft_alignment": {"checked_rows": len(train), "all_ids_and_contents_match": True},
            "counts": {"original_train": _task_counts(original_train), **{name: _task_counts(rows) for name, rows in subsets.items()}},
            "duplicates": {
                name: {kind: _duplicate_summary(rows, kind == "question") for kind in ("content", "question")}
                for name, rows in {"original_train": original_train, **subsets}.items()
            },
            "cross_split": {},
            "text_quality": {
                "replacement_character_rows": sum("\ufffd" in row["Question"] + row["Context"] for row in original_train + test_rows),
                "literal_backslash_n_rows": sum("\\n" in row["Question"] + row["Context"] for row in original_train + test_rows),
                "max_question_characters": max(len(row["Question"]) for row in original_train + test_rows),
                "max_context_characters": max(len(row["Context"]) for row in original_train + test_rows),
            },
            "interpretation": [
                "分层随机划分保留课程任务比例，不实施按来源分组划分；不因检查结果改变固定划分。",
                "相同来源数据集并不等于泄漏；相同来源 ID 或 URL 表明共享原题/文档的可能，需结合内容判断。",
                "问题或问题+段落重复可能使验证/测试结果偏乐观；本检查不能排除语义改写或预训练数据污染。",
                "仅检查测试问题、段落和公开来源元数据。测试答案仅可在方案冻结且四组完整预测保存后由最终评测程序读取。",
            ],
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            left_rows, right_rows = subsets[left], subsets[right]
            audit["cross_split"][f"{left}__{right}"] = {
                "content": _cross_duplicate_summary(left_rows, right_rows, False),
                "question": _cross_duplicate_summary(left_rows, right_rows, True),
                "sources": _source_overlap([row["Sample_id"] for row in left_rows], [row["Sample_id"] for row in right_rows], sources),
            }
        _dump_json(prepared / "audit.json", audit)
        (prepared / "audit.txt").write_text(_audit_text(audit), encoding="utf-8")
        artifact_hashes = {path.relative_to(prepared).as_posix(): sha256_file(path) for path in sorted(prepared.rglob("*")) if path.is_file()}
        manifest = {
            "schema_version": 1, "seed": seed,
            "split_algorithm": "sorted IDs; Python Random(seed); tasks math,qa,rag shuffled in order; round(0.1*N) validation; first 60 validation/task tuning; output sorted",
            "validation_fraction": 0.1, "tune_per_task": 60,
            "source_sha256": source_hashes, "artifact_sha256": artifact_hashes,
            "task_counts": audit["counts"], "train_ids": train_ids,
            "validation_ids": validation_ids, "tune_ids": tune_ids, "test_ids": test_ids,
            "test_answers_read": False,
        }
        _dump_json(prepared / "manifest.json", manifest)
        if output.exists():
            if output.is_symlink() or not output.is_dir():
                raise FileExistsError(f"Output is not a regular directory: {output}")
            if any(output.iterdir()):
                existing_path = output / "manifest.json"
                if not existing_path.is_file() or load_json(existing_path) != manifest:
                    raise FileExistsError("Prepared data differs from existing manifest; use a NEW output directory")
                for name, digest in artifact_hashes.items():
                    existing_file = output / name
                    if not existing_file.is_file() or existing_file.is_symlink() or sha256_file(existing_file) != digest:
                        raise FileExistsError(f"Existing prepared artifact was modified or is missing: {name}")
                return audit
            output.rmdir()
        prepared.rename(output)
        return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/prepared"), type=Path)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    audit = prepare_dataset(args.archive, args.output, args.seed)
    print(json.dumps({"output": str(args.output.resolve()), "counts": audit["counts"], "test_answers_opened": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
