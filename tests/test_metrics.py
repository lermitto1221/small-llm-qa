"""Synthetic tests: never access the course archive or its held-out answers."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from small_llm_qa.evaluate import METHODS, evaluate_final, evaluate_validation, sha256_file
from small_llm_qa.metrics import exact_match, normalize_answer, parse_numeric_answer, score_records, token_f1


def reference(sample_id: str, task: str, answer: str, aliases: list[str] | None = None) -> dict[str, str]:
    return {"Sample_id": sample_id, "Task": task, "Answer": answer, "Accepted_answers": json.dumps(aliases or [answer])}


class NormalizationTests(unittest.TestCase):
    def test_documented_squad_normalization(self):
        self.assertEqual(normalize_answer("  The, QUICK\tFox! A an fox. "), "quick fox fox")
        self.assertEqual(exact_match("The United States!", "united states"), 1.0)

    def test_f1_counts_token_multiplicity(self):
        self.assertAlmostEqual(token_f1("cat cat dog", "cat dog"), 0.8)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("", "dog"), 0.0)
        self.assertEqual(token_f1("red", "blue"), 0.0)

    def test_accepted_answers_maximum(self):
        result = score_records([reference("v1", "qa", "United States", ["USA", "US"])], [{"Sample_id": "v1", "Answer": "USA"}])
        self.assertEqual(result["metrics"]["qa"]["exact_match"], 1.0)
        self.assertIsNone(result["metrics"]["macro_score"])

    def test_malformed_aliases_rejected(self):
        for aliases in ('"USA"', "[2]", "invalid json", '{"x":"USA"}'):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                score_records([{"Sample_id": "v1", "Task": "qa", "Answer": "USA", "Accepted_answers": aliases}], [{"Sample_id": "v1", "Answer": "USA"}])


class NumericTests(unittest.TestCase):
    def test_equivalent_numeric_formats(self):
        for value in ("1/2", "0.5", ".5", "5e-1", "50/100", "+1.0 / +2.0", "5E-1", "500/1,000"):
            with self.subTest(value=value):
                parsed = parse_numeric_answer(value)
                self.assertIsNone(parsed.error)
                self.assertEqual(parsed.value, 0.5)
                self.assertEqual(parsed.canonical, "1/2")
        self.assertEqual(parse_numeric_answer(" -1,234.50 ").value, -1234.5)
        self.assertEqual(parse_numeric_answer("1/-2").value, -0.5)

    def test_invalid_and_adversarial_values_fail_without_execution(self):
        values = ["", "nan", "NaN", "inf", "1/0", "1/2/3", "1,23", "12,34.5", "42 dollars", "Answer: 42", "1+1", "2 = 2", "```42```", "\\boxed{42}", "50%", "__import__('os').system('echo unsafe')", "1e999999999999999999999", "9" * 513]
        for value in values:
            with self.subTest(value=value):
                parsed = parse_numeric_answer(value)
                self.assertIsNotNone(parsed.error)
                self.assertIsNone(parsed.value)

    def test_absolute_relative_tolerances_and_parse_failures(self):
        refs = [reference("m1", "math", "0"), reference("m2", "math", "1000"), reference("m3", "math", "0"), reference("m4", "math", "42")]
        predictions = [{"Sample_id": "m1", "Answer": "0.0000009"}, {"Sample_id": "m2", "Answer": "1000.009"}, {"Sample_id": "m3", "Answer": "0.0000011"}, {"Sample_id": "m4", "Answer": "Answer: 42"}]
        result = score_records(refs, predictions)
        self.assertEqual(result["metrics"]["math"]["numeric_accuracy"], 0.5)
        self.assertEqual(result["metrics"]["math"]["parse_failure_count"], 1)
        self.assertEqual(result["samples"][3]["prediction"], "Answer: 42")
        self.assertEqual(result["samples"][3]["parsed_prediction"]["error"], "invalid_numeric_format")

    def test_bad_numeric_reference_is_data_error(self):
        with self.assertRaisesRegex(ValueError, "invalid numeric answer"):
            score_records([reference("m1", "math", "forty")], [{"Sample_id": "m1", "Answer": "40"}])


class AlignmentAndAggregationTests(unittest.TestCase):
    def setUp(self):
        self.references = [reference("q1", "qa", "red"), reference("r1", "rag", "blue"), reference("m1", "math", "1/2")]
        self.predictions = [{"Sample_id": "m1", "Answer": "0.5"}, {"Sample_id": "q1", "Answer": "red"}, {"Sample_id": "r1", "Answer": "blue"}]

    def test_alignment_uses_ids_not_rows(self):
        result = score_records(self.references, self.predictions)
        self.assertEqual(result["metrics"]["macro_score"], 1.0)

    def test_duplicate_reference_and_prediction_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_records(self.references, self.predictions + [self.predictions[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_records(self.references + [self.references[0]], self.predictions)

    def test_missing_and_extra_predictions_rejected(self):
        for predictions in (self.predictions[:-1], self.predictions + [{"Sample_id": "unexpected", "Answer": "x"}]):
            with self.assertRaisesRegex(ValueError, "Sample_id mismatch"):
                score_records(self.references, predictions)

    def test_macro_uses_equal_task_weights(self):
        refs = self.references + [reference("q2", "qa", "red"), reference("q3", "qa", "red")]
        preds = [{**row, "Answer": "wrong" if row["Sample_id"] == "r1" else row["Answer"]} for row in self.predictions]
        preds += [{"Sample_id": "q2", "Answer": "red"}, {"Sample_id": "q3", "Answer": "red"}]
        self.assertAlmostEqual(score_records(refs, preds)["metrics"]["macro_score"], 2 / 3)

    def test_rejection_confusion_counts(self):
        refs = [reference("r1", "rag", "UNANSWERABLE"), reference("r2", "rag", "UNANSWERABLE"), reference("r3", "rag", "blue"), reference("r4", "rag", "red")]
        preds = [{"Sample_id": "r1", "Answer": "unanswerable."}, {"Sample_id": "r2", "Answer": "invented"}, {"Sample_id": "r3", "Answer": "UNANSWERABLE"}, {"Sample_id": "r4", "Answer": "red"}]
        metrics = score_records(refs, preds)["metrics"]["rag"]
        self.assertEqual(metrics["unanswerable_precision"], 0.5)
        self.assertEqual(metrics["unanswerable_recall"], 0.5)
        self.assertEqual(metrics["false_rejection_count"], 1)
        self.assertEqual(metrics["false_rejection_rate"], 0.5)
        self.assertEqual(metrics["false_acceptance_count"], 1)


class EvaluationGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.csv_file("test.csv", [{"Sample_id": "t1", "Task": "qa", "Question": "Synthetic?", "Context": ""}, {"Sample_id": "t2", "Task": "rag", "Question": "Synthetic?", "Context": "None"}, {"Sample_id": "t3", "Task": "math", "Question": "Synthetic?", "Context": ""}])
        self.data_manifest = {"test_ids": ["t1", "t2", "t3"], "artifact_sha256": {"test.csv": sha256_file(self.root / "test.csv")}, "source_sha256": {"test.csv": sha256_file(self.root / "test.csv")}}
        self.json_file("manifest.json", self.data_manifest)
        self.config = {"experiment_configuration_frozen": True, "model": {"name": "synthetic", "revision": "fixed"}, "input_budget": 2048, "max_new_tokens": 128, "selected_n": 3, "source_code_sha256": "a" * 64, "data_manifest_sha256": sha256_file(self.root / "manifest.json")}
        self.json_file("frozen_config.json", self.config)
        self.bundle = {"schema_version": 1, "config_path": "frozen_config.json", "config_sha256": sha256_file(self.root / "frozen_config.json"), "test_path": "test.csv", "test_sha256": sha256_file(self.root / "test.csv"), "data_manifest_path": "manifest.json", "data_manifest_sha256": self.config["data_manifest_sha256"], "runs": {}}
        self.inference = {"model": self.config["model"], "input_budget": 2048, "max_new_tokens": 128, "quantization": "nf4-double", "enable_thinking": False, "do_sample": False, "num_beams": 1, "prompt_version": "course-task-context-v1"}
        for method in METHODS:
            self.csv_file(f"{method}.csv", [{"Sample_id": "t3", "Answer": "0.5"}, {"Sample_id": "t1", "Answer": "alias"}, {"Sample_id": "t2", "Answer": "UNANSWERABLE"}])
            entry = {"prediction_path": f"{method}.csv", "prediction_sha256": sha256_file(self.root / f"{method}.csv"), "run_manifest_path": f"{method}.json"}
            manifest = {"method": method, "config_sha256": self.bundle["config_sha256"], "test_sha256": self.bundle["test_sha256"], "prediction_sha256": entry["prediction_sha256"], "completed": True, "inference_config": self.inference, "code_sha256": self.config["source_code_sha256"], "split": "test", "limit_per_task": None, "input_sha256": self.bundle["test_sha256"], "n": 3 if method in ("random", "similar") else 0, "count": 3, "expected_ids": ["t1", "t2", "t3"]}
            self.json_file(f"{method}.json", manifest)
            entry["run_manifest_sha256"] = sha256_file(self.root / f"{method}.json")
            self.bundle["runs"][method] = entry
        self.json_file("final_bundle.json", self.bundle)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=["Sample_id", "Answer", "Accepted_answers"])
        writer.writeheader()
        writer.writerows([{"Sample_id": "t2", "Answer": "UNANSWERABLE", "Accepted_answers": '[]'}, {"Sample_id": "t3", "Answer": "1/2", "Accepted_answers": '[]'}, {"Sample_id": "t1", "Answer": "primary", "Accepted_answers": '["alias"]'}])
        with zipfile.ZipFile(self.root / "synthetic.zip", "w") as archive:
            archive.writestr("test.csv", (self.root / "test.csv").read_bytes())
            archive.writestr("test_answer.csv", buffer.getvalue())

    def tearDown(self):
        self.temporary.cleanup()

    def json_file(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def csv_file(self, name, rows):
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def evaluate(self):
        return evaluate_final(self.root / "final_bundle.json", self.root / "synthetic.zip", self.root / "results.json")

    def assert_gate_blocks_archive(self):
        with patch("small_llm_qa.evaluate.zipfile.ZipFile") as open_archive:
            with self.assertRaises(ValueError):
                self.evaluate()
            open_archive.assert_not_called()

    def update_manifest(self, method, **updates):
        path = self.root / f"{method}.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(updates)
        self.json_file(path.name, manifest)
        self.bundle["runs"][method]["run_manifest_sha256"] = sha256_file(path)
        self.json_file("final_bundle.json", self.bundle)

    def test_final_only_after_four_complete_predictions(self):
        result = self.evaluate()
        self.assertEqual(set(result["runs"]), set(METHODS))
        self.assertTrue(all(run["metrics"]["macro_score"] == 1.0 for run in result["runs"].values()))
        self.assertTrue((self.root / "results.json").exists())

    def test_unfrozen_configuration_blocks_archive(self):
        self.config["experiment_configuration_frozen"] = False
        self.json_file("frozen_config.json", self.config)
        self.bundle["config_sha256"] = sha256_file(self.root / "frozen_config.json")
        self.json_file("final_bundle.json", self.bundle)
        self.assert_gate_blocks_archive()

    def test_missing_fourth_run_blocks_archive(self):
        del self.bundle["runs"]["sft"]
        self.json_file("final_bundle.json", self.bundle)
        self.assert_gate_blocks_archive()

    def test_changed_predictions_block_archive(self):
        with (self.root / "sft.csv").open("a", encoding="utf-8") as handle:
            handle.write("bad,unexpected\n")
        self.assert_gate_blocks_archive()

    def test_incomplete_prediction_ids_block_archive_even_with_updated_hash(self):
        self.csv_file("sft.csv", [{"Sample_id": "t1", "Answer": "alias"}])
        digest = sha256_file(self.root / "sft.csv")
        self.bundle["runs"]["sft"]["prediction_sha256"] = digest
        self.update_manifest("sft", prediction_sha256=digest)
        self.assert_gate_blocks_archive()

    def test_incomplete_run_manifest_blocks_archive(self):
        self.update_manifest("sft", completed=False)
        self.assert_gate_blocks_archive()

    def test_inference_change_blocks_archive(self):
        self.update_manifest("sft", inference_config={**self.inference, "do_sample": True})
        self.assert_gate_blocks_archive()

    def test_subset_public_test_csv_blocks_archive_even_with_updated_bundle_hash(self):
        self.csv_file("test.csv", [{"Sample_id": "t1", "Task": "qa", "Question": "Synthetic?", "Context": ""}])
        self.bundle["test_sha256"] = sha256_file(self.root / "test.csv")
        self.json_file("final_bundle.json", self.bundle)
        self.assert_gate_blocks_archive()

    def test_changed_data_manifest_blocks_archive_even_with_updated_bundle_hash(self):
        self.data_manifest["test_ids"] = ["t1"]
        self.json_file("manifest.json", self.data_manifest)
        self.bundle["data_manifest_sha256"] = sha256_file(self.root / "manifest.json")
        self.json_file("final_bundle.json", self.bundle)
        self.assert_gate_blocks_archive()

    def test_changed_run_code_blocks_archive(self):
        self.update_manifest("sft", code_sha256="b" * 64)
        self.assert_gate_blocks_archive()

    def test_wrong_n_or_split_blocks_archive(self):
        self.update_manifest("random", n=1)
        self.assert_gate_blocks_archive()
        self.update_manifest("random", n=3, split="validation_tune")
        self.assert_gate_blocks_archive()

    def test_duplicate_planned_ids_block_archive(self):
        self.update_manifest("sft", expected_ids=["t1", "t2", "t3", "t3"])
        self.assert_gate_blocks_archive()

    def test_wrong_archive_source_rejected_before_opening_answers(self):
        wrong_archive = self.root / "wrong_source.zip"
        with zipfile.ZipFile(wrong_archive, "w") as archive:
            # Keep the same IDs but change question contents: ID checks alone fail to detect this.
            archive.writestr("test.csv", (self.root / "test.csv").read_bytes().replace(b"Synthetic?", b"Different?"))
            archive.writestr("test_answer.csv", "Sample_id,Answer\nt1,never read\n")
        original_open = zipfile.ZipFile.open
        opened = []

        def record_open(archive, name, *args, **kwargs):
            opened.append(name)
            return original_open(archive, name, *args, **kwargs)

        with patch("small_llm_qa.evaluate.zipfile.ZipFile.open", new=record_open):
            with self.assertRaisesRegex(ValueError, "original prepared dataset source"):
                evaluate_final(self.root / "final_bundle.json", wrong_archive, self.root / "unused_results.json")
        self.assertEqual(opened, ["test.csv"])

    def test_validation_only_accepts_manifest_ids(self):
        rows = [reference("v1", "qa", "red"), reference("v2", "rag", "blue"), reference("v3", "math", "1/2")]
        self.csv_file("validation.csv", rows)
        self.csv_file("validation_predictions.csv", [{"Sample_id": row["Sample_id"], "Answer": row["Answer"]} for row in rows])
        self.json_file("split.json", {"validation_ids": ["v1", "v2", "v3"], "train_ids": ["train1"], "test_ids": ["t1", "t2", "t3"], "tune_ids": ["v1"]})
        result = evaluate_validation(self.root / "validation.csv", self.root / "validation_predictions.csv", self.root / "split.json", self.root / "validation_results.json")
        self.assertEqual(result["metrics"]["macro_score"], 1.0)
        self.csv_file("validation.csv", [reference("t1", "qa", "leaked")])
        with self.assertRaisesRegex(ValueError, "Sample_id mismatch"):
            evaluate_validation(self.root / "validation.csv", self.root / "validation_predictions.csv", self.root / "split.json", self.root / "validation_results.json")

    def test_validation_rejects_arbitrary_reference_filename(self):
        with self.assertRaisesRegex(ValueError, "prepared validation"):
            evaluate_validation(self.root / "test_answer.csv", self.root / "baseline.csv", self.root / "unused.json", self.root / "unused_output.json")


if __name__ == "__main__":
    unittest.main()
