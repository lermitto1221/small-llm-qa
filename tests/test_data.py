"""Data-integrity tests use synthetic answers; no real test labels are opened."""

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from small_llm_qa.data import (
    INPUT_FIELDS, LABEL_FIELDS, SOURCE_FIELDS, SYSTEM_PROMPT,
    format_user_message, load_json, prepare_dataset, read_csv, sha256_file,
)


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fixture_members():
    train, test, sft, sources = [], [], [], []
    for task in ("math", "qa", "rag"):
        for index in range(600):
            row = {
                "Sample_id": f"TRAIN_{task}_{index:04d}", "Task": task,
                "Question": f"{task} question {index}?", "Context": "[1] passage" if task == "rag" else "",
                "Answer": "2", "Accepted_answers": '["2", "two"]',
            }
            train.append(row)
            sft.append({"Sample_id": row["Sample_id"], "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": format_user_message(row)},
                {"role": "assistant", "content": row["Answer"]},
            ]})
            sources.append({"Sample_id": row["Sample_id"], "Source": "synthetic", "Source_id": str(index), "Source_urls": '["https://example.org/article"]'})
        row = {"Sample_id": f"TEST_{task}", "Task": task, "Question": f"{task} heldout?", "Context": "[1] test passage" if task == "rag" else ""}
        test.append(row)
        sources.append({"Sample_id": row["Sample_id"], "Source": "synthetic", "Source_id": "0", "Source_urls": '["https://example.org/article"]'})
    return {
        "train.csv": csv_bytes(train, LABEL_FIELDS), "test.csv": csv_bytes(test, INPUT_FIELDS),
        "sources.csv": csv_bytes(sources, SOURCE_FIELDS),
        "train_sft.jsonl": ("\n".join(json.dumps(row) for row in sft) + "\n").encode(),
        "README.txt": b"Synthetic fixture. Not experiment data.",
        "LICENSES/ATTRIBUTION.txt": b"Synthetic test fixture only.",
        "test_answer.csv": b"INTENTIONALLY INVALID CSV; MUST NEVER OPEN",
    }


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.members = fixture_members()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, overrides=None, name="dataset.zip"):
        members = {**self.members, **(overrides or {})}
        target = self.root / name
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for key, value in members.items():
                if value is not None:
                    bundle.writestr(key, value)
        return target

    def test_split_integrity_and_test_answers_never_opened(self):
        archive = self.archive()
        original_open = zipfile.ZipFile.open

        def guarded_open(bundle, member, *args, **kwargs):
            name = member.filename if isinstance(member, zipfile.ZipInfo) else member
            if name.rsplit("/", 1)[-1].lower() == "test_answer.csv":
                raise AssertionError("Attempted to open protected test answers")
            return original_open(bundle, member, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "open", guarded_open):
            audit = prepare_dataset(archive, self.root / "prepared")
        prepared = self.root / "prepared"
        manifest = load_json(prepared / "manifest.json")
        train_ids, val_ids, tune_ids = map(set, (manifest["train_ids"], manifest["validation_ids"], manifest["tune_ids"]))
        self.assertEqual((len(train_ids), len(val_ids), len(tune_ids)), (1620, 180, 180))
        self.assertFalse(train_ids & val_ids)
        self.assertLessEqual(tune_ids, val_ids)
        self.assertEqual(manifest["task_counts"]["validation_tune"], {"math": 60, "qa": 60, "rag": 60})
        sft_ids = {json.loads(line)["Sample_id"] for line in (prepared / "train_sft.jsonl").read_text().splitlines()}
        self.assertEqual(sft_ids, train_ids)
        self.assertFalse((prepared / "test_answer.csv").exists())
        self.assertNotIn("test_answer.csv", manifest["source_sha256"])
        self.assertFalse(audit["test_answers_opened"])
        self.assertEqual((prepared / "LICENSES/ATTRIBUTION.txt").read_bytes(), self.members["LICENSES/ATTRIBUTION.txt"])
        self.assertGreater(audit["cross_split"]["train__validation"]["sources"]["source_url"]["shared_keys"], 0)
        for path, expected in manifest["artifact_sha256"].items():
            self.assertEqual(sha256_file(prepared / path), expected)

    def test_repeat_is_deterministic_and_changed_seed_is_rejected(self):
        archive, target = self.archive(), self.root / "prepared"
        first = prepare_dataset(archive, target)
        before = (target / "manifest.json").read_bytes()
        second = prepare_dataset(archive, target)
        self.assertEqual(first, second)
        self.assertEqual(before, (target / "manifest.json").read_bytes())
        with self.assertRaisesRegex(FileExistsError, "NEW output"):
            prepare_dataset(archive, target, seed=43)

    def test_changed_prepared_file_is_rejected(self):
        archive, target = self.archive(), self.root / "prepared"
        prepare_dataset(archive, target)
        (target / "train.csv").write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "modified"):
            prepare_dataset(archive, target)

    def test_changed_source_is_rejected(self):
        archive, target = self.archive(), self.root / "prepared"
        prepare_dataset(archive, target)
        changed = self.archive({"README.txt": b"changed source"}, name="changed.zip")
        with self.assertRaisesRegex(FileExistsError, "NEW output"):
            prepare_dataset(changed, target)

    def test_sft_content_mismatch_fails_before_output(self):
        records = [json.loads(line) for line in self.members["train_sft.jsonl"].splitlines()]
        records[0]["messages"][2]["content"] = "wrong"
        changed = ("\n".join(map(json.dumps, records)) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "content mismatch"):
            prepare_dataset(self.archive({"train_sft.jsonl": changed}), self.root / "prepared")
        self.assertFalse((self.root / "prepared").exists())

    def test_sft_missing_id_is_rejected(self):
        lines = self.members["train_sft.jsonl"].splitlines()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            prepare_dataset(self.archive({"train_sft.jsonl": b"\n".join(lines[1:])}), self.root / "prepared")

    def test_duplicate_train_id_is_rejected(self):
        rows = list(csv.DictReader(io.StringIO(self.members["train.csv"].decode())))
        rows[-1]["Sample_id"] = rows[0]["Sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate Sample_id"):
            prepare_dataset(self.archive({"train.csv": csv_bytes(rows, LABEL_FIELDS)}), self.root / "prepared")

    def test_invalid_accepted_answers_are_rejected(self):
        rows = list(csv.DictReader(io.StringIO(self.members["train.csv"].decode())))
        rows[0]["Accepted_answers"] = "[]"
        with self.assertRaisesRegex(ValueError, "reference answers"):
            prepare_dataset(self.archive({"train.csv": csv_bytes(rows, LABEL_FIELDS)}), self.root / "prepared")

    def test_test_answer_columns_in_unlabelled_test_are_rejected(self):
        rows = list(csv.DictReader(io.StringIO(self.members["test.csv"].decode())))
        for row in rows:
            row["Answer"] = "accidental labels"
        with self.assertRaisesRegex(ValueError, "Unexpected CSV columns"):
            prepare_dataset(self.archive({"test.csv": csv_bytes(rows, INPUT_FIELDS + ("Answer",))}), self.root / "prepared")

    def test_path_traversal_and_missing_license_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            prepare_dataset(self.archive({"../escape.txt": b"bad"}), self.root / "prepared")
        with self.assertRaisesRegex(ValueError, "LICENSES"):
            prepare_dataset(self.archive({"LICENSES/ATTRIBUTION.txt": None}, name="nolicense.zip"), self.root / "prepared")

    def test_malformed_csv_is_rejected(self):
        path = self.root / "broken.csv"
        path.write_text("a,b\n1,2,3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Malformed CSV"):
            read_csv(path)
        path.write_text("a,a\n1,2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate CSV headers"):
            read_csv(path)


if __name__ == "__main__":
    unittest.main()
