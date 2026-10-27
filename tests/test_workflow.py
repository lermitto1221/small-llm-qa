"""CPU orchestration tests with an explicitly fake generator, never model scores."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from small_llm_qa import workflow
from small_llm_qa.io_utils import file_hash, read_journal, tree_hash, write_json
from small_llm_qa.modeling import ModelConfig
from small_llm_qa.selection import choose_common_n, random_examples
from small_llm_qa.training import TrainingConfig


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert kwargs["enable_thinking"] is False
        text = "\n".join(f"{row['role']}:{row['content']}" for row in messages)
        if kwargs.get("add_generation_prompt"):
            text += "\nassistant:"
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


class FakeIndex:
    preparation_seconds = 0.01
    original_index_seconds = 0.01
    cache_hit = False

    def __init__(self, pool, config, cache_dir):
        self.pool = pool

    def select(self, query, n):
        assert set(query) == {"Sample_id", "Task", "Question", "Context"}
        return [row for row in self.pool if row["Task"] == query["Task"]][:n], 0.001


def fake_generate(loaded, prompt, max_new_tokens):
    # The held-out answer is deliberately detectable if it ever enters a prompt.
    assert "PRIVATE_REFERENCE" not in prompt.text
    return {"Answer": "2", "generated_text": "2", "raw_output": "2<|im_end|>",
            "generated_token_ids": [50], "input_tokens": prompt.input_tokens,
            "generated_tokens": 1, "elapsed_seconds": 0.01,
            "peak_memory_allocated_bytes": 100, "peak_memory_reserved_bytes": 200,
            "hit_token_limit": False}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.config = {"schema_version": 1, "seed": 42,
                       "model": asdict(ModelConfig(compute_dtype="float16")),
                       "embedding": {"name": "synthetic-embedding", "revision": "a" * 40},
                       "input_budget": 8192, "max_new_tokens": 128, "n_candidates": [1, 3, 5],
                       "training": asdict(TrainingConfig()), "runtime_versions": {},
                       "experiment_configuration_frozen": False}
        self.config_path = self.root / "run_config.json"
        self.loaded = SimpleNamespace(tokenizer=FakeTokenizer(), metadata={
            "config": self.config["model"], "compute_dtype": "float16", "gpu": "FAKE_CPU_TEST",
            "dependencies": {}, "adapter_path": None,
        })
        self.pool, self.validation, self.test = [], [], []
        for task in ("qa", "rag", "math"):
            for index in range(6):
                self.pool.append(self.row(f"TRAIN_{task}_{index}", task, "2"))
            for index in range(60):
                answer = "2" if task == "math" else "PRIVATE_REFERENCE"
                self.validation.append(self.row(f"VALID_{task}_{index:02d}", task, answer))
            self.test.append({key: value for key, value in self.row(f"TEST_{task}", task, "unused").items()
                              if key not in {"Answer", "Accepted_answers"}})
        for name, rows in (("train", self.pool), ("validation", self.validation),
                           ("validation_tune", self.validation), ("test", self.test)):
            self.write_csv(self.prepared / f"{name}.csv", rows)
        self.manifest = {"train_ids": [r["Sample_id"] for r in self.pool],
                         "validation_ids": [r["Sample_id"] for r in self.validation],
                         "tune_ids": [r["Sample_id"] for r in self.validation],
                         "test_ids": [r["Sample_id"] for r in self.test],
                         "artifact_sha256": {p.name: file_hash(p) for p in self.prepared.glob("*.csv")}}
        write_json(self.prepared / "manifest.json", self.manifest)
        self.config.update({"data_manifest_sha256": file_hash(self.prepared / "manifest.json"),
                            "source_code_sha256": "c" * 64})
        write_json(self.config_path, self.config)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.fingerprint = patch("small_llm_qa.workflow.code_fingerprint", return_value="c" * 64)
        self.fingerprint.start()
        self.dependencies = patch("small_llm_qa.modeling.dependency_versions", return_value={})
        self.dependencies.start()

    def tearDown(self):
        self.fingerprint.stop()
        self.dependencies.stop()
        self.redirect.__exit__(None, None, None)
        self.temp.cleanup()

    @staticmethod
    def row(sample_id, task, answer):
        return {"Sample_id": sample_id, "Task": task, "Question": f"Question {sample_id}",
                "Context": "[1] Entire supplied context." if task == "rag" else "",
                "Answer": answer, "Accepted_answers": json.dumps([answer])}

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def run_predict(self, folder, **kwargs):
        with patch("small_llm_qa.modeling.generate_answer", side_effect=fake_generate):
            return workflow.predict(self.prepared, self.config_path, folder, loaded=self.loaded, **kwargs)

    def selection(self):
        folder = self.root / "tuning"
        with patch("small_llm_qa.modeling.load_model", return_value=self.loaded), \
             patch("small_llm_qa.modeling.generate_answer", side_effect=fake_generate), \
             patch("small_llm_qa.workflow.SimilarityIndex", FakeIndex):
            workflow.tune(self.prepared, self.config_path, folder)
        return folder / "selection.json"

    def training_evidence(self, **overrides):
        adapter = self.root / "training" / "adapter"
        adapter.mkdir(parents=True)
        # Fixture bytes only: these are never loaded as a model.
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"synthetic test fixture")
        evidence = {"status": "complete", "completed": True, "smoke": False,
                    "config_sha256": file_hash(self.config_path),
                    "train_csv_sha256": file_hash(self.prepared / "train.csv"),
                    "validation_csv_sha256": file_hash(self.prepared / "validation.csv"),
                    "code_sha256": "c" * 64,
                    "data_manifest_sha256": file_hash(self.prepared / "manifest.json"),
                    "dependencies": {}, "adapter_sha256": tree_hash(adapter),
                    "environment": self.loaded.metadata, "train_rows": len(self.pool),
                    "validation_rows": len(self.validation), "global_step": 3,
                    **overrides}
        write_json(adapter.parent / "training_summary.json", evidence)
        write_json(adapter.parent / "training_manifest.json", {
            "model": self.config["model"], "training": self.config["training"], "smoke": False,
            "train_sha256": file_hash(self.prepared / "train.csv"),
            "validation_sha256": file_hash(self.prepared / "validation.csv"),
            "prompt_version": "course-task-context-v1"})
        return adapter

    def test_predict_same_task_demos_exclude_validation_labels_and_resume(self):
        folder = self.root / "random"
        first = self.run_predict(folder, method="random", n=3)
        records = read_journal(folder / "raw_predictions.jsonl", set(self.manifest["tune_ids"]))
        self.assertEqual(first["count"], 180)
        self.assertEqual(first["mean_actual_shots"], 3)
        pool = {r["Sample_id"]: r for r in self.pool}
        for record in records.values():
            self.assertEqual(len(record["demo_ids"]), 3)
            self.assertTrue(all(pool[sid]["Task"] == record["Task"] for sid in record["demo_ids"]))
        original = (folder / "predictions.csv").read_bytes()
        with patch("small_llm_qa.modeling.generate_answer", side_effect=AssertionError("Completed rows must not rerun")):
            resumed = workflow.predict(self.prepared, self.config_path, folder, method="random", n=3, loaded=self.loaded)
        self.assertEqual(resumed["count"], 180)
        self.assertEqual(original, (folder / "predictions.csv").read_bytes())

    def test_interrupted_run_resumes_exactly_missing_ids(self):
        folder = self.root / "interrupted"
        calls = []

        def interrupt(loaded, prompt, limit):
            if len(calls) == 2:
                raise RuntimeError("Simulated interruption")
            calls.append(prompt.text)
            return fake_generate(loaded, prompt, limit)

        with patch("small_llm_qa.modeling.generate_answer", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "Simulated"):
                workflow.predict(self.prepared, self.config_path, folder, loaded=self.loaded)
        with patch("small_llm_qa.modeling.generate_answer", side_effect=fake_generate) as generate:
            workflow.predict(self.prepared, self.config_path, folder, loaded=self.loaded)
        self.assertEqual(generate.call_count, 178)
        self.assertEqual(len(workflow.read_rows(folder / "predictions.csv")), 180)

    def test_changed_run_config_refuses_resume(self):
        folder = self.root / "changed"
        self.run_predict(folder, limit=1)
        self.config["max_new_tokens"] = 64
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "Immutable"):
            self.run_predict(folder, limit=1)

    def test_runtime_dependency_change_is_rejected(self):
        with patch("small_llm_qa.modeling.dependency_versions", return_value={"torch": "different"}):
            with self.assertRaisesRegex(ValueError, "Dependency versions"):
                self.run_predict(self.root / "wrong-runtime", limit=1)

    def test_target_ids_must_equal_manifest_ids(self):
        self.manifest["tune_ids"] = self.manifest["tune_ids"][1:]
        write_json(self.prepared / "manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "Target IDs"):
            self.run_predict(self.root / "wrong-targets")

    def test_loaded_adapter_must_match_requested_method(self):
        self.loaded.metadata["adapter_path"] = str(self.root / "unexpected-adapter")
        with self.assertRaisesRegex(ValueError, "[Aa]dapter"):
            self.run_predict(self.root / "wrong-baseline", method="baseline", limit=1)
        self.loaded.metadata["adapter_path"] = None
        adapter = self.training_evidence()
        self.config.update({"adapter_path": str(adapter), "adapter_sha256": tree_hash(adapter)})
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "[Aa]dapter"):
            self.run_predict(self.root / "wrong-sft", method="sft", limit=1)

    def test_duplicate_and_partial_journal_handling(self):
        journal = self.root / "journal.jsonl"
        journal.write_bytes(b'{"Sample_id":"A"}\n{"Sample_id":')
        records = read_journal(journal, {"A", "B"})
        self.assertEqual(set(records), {"A"})
        self.assertTrue(journal.with_suffix(".incomplete-tail").exists())
        with journal.open("ab") as handle:
            handle.write(b'{"Sample_id":"A"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_journal(journal, {"A", "B"})

    def test_test_predictions_require_freeze_and_full_coverage(self):
        folder = self.root / "test"
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.run_predict(folder, split="test")
        self.config.update({"experiment_configuration_frozen": True, "selected_n": 3})
        write_json(self.config_path, self.config)
        with self.assertRaisesRegex(ValueError, "full test coverage"):
            self.run_predict(folder, split="test", limit=1)
        stats = self.run_predict(folder, split="test", method="random", n=1)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["mean_actual_shots"], 3)
        self.assertFalse((folder / "evaluation.json").exists())
        self.assertEqual(workflow.read_json(folder / "identity.json")["n"], 3)

    def test_tune_all_six_runs_uses_fixed_validation_and_tie_break(self):
        folder = self.root / "tuning"
        with patch("small_llm_qa.modeling.load_model", return_value=self.loaded), \
             patch("small_llm_qa.modeling.generate_answer", side_effect=fake_generate), \
             patch("small_llm_qa.workflow.SimilarityIndex", FakeIndex):
            workflow.tune(self.prepared, self.config_path, folder)
        selection = workflow.read_json(folder / "selection.json")
        self.assertEqual(selection["selected_n"], 1)
        self.assertEqual(set(selection["validation_ids"]), set(self.manifest["tune_ids"]))
        for n in (1, 3, 5):
            for method in ("random", "similar"):
                run = workflow.read_json(folder / f"{method}_{n}" / "run_manifest.json")
                self.assertEqual(run["count"], 180)
                self.assertEqual(set(run["expected_ids"]), set(self.manifest["tune_ids"]))

    def test_freeze_accepts_completed_training_evidence(self):
        selection, adapter = self.selection(), self.training_evidence()
        frozen = self.root / "frozen.json"
        workflow.freeze(self.prepared, self.config_path, selection, adapter, frozen)
        config = workflow.read_json(frozen)
        self.assertTrue(config["experiment_configuration_frozen"])
        self.assertEqual(config["selected_n"], 1)
        self.assertEqual(config["training_summary_sha256"], file_hash(adapter.parent / "training_summary.json"))

    def test_freeze_rejects_smoke_adapter(self):
        selection, adapter = self.selection(), self.training_evidence(smoke=True)
        with self.assertRaisesRegex(ValueError, "Smoke|smoke"):
            workflow.freeze(self.prepared, self.config_path, selection, adapter, self.root / "frozen.json")

    def test_freeze_rejects_training_from_another_config_or_split(self):
        selection, adapter = self.selection(), self.training_evidence()
        summary = adapter.parent / "training_summary.json"
        correct = workflow.read_json(summary)
        for field in ("config_sha256", "train_csv_sha256", "validation_csv_sha256",
                      "code_sha256", "data_manifest_sha256", "adapter_sha256"):
            write_json(summary, {**correct, field: "0" * 64})
            with self.subTest(field=field), self.assertRaises(ValueError):
                workflow.freeze(self.prepared, self.config_path, selection, adapter, self.root / f"frozen_{field}.json")
        write_json(summary, {**correct, "dependencies": {"torch": "different"}})
        with self.assertRaises(ValueError):
            workflow.freeze(self.prepared, self.config_path, selection, adapter, self.root / "frozen_dependencies.json")
        write_json(summary, correct)

    def test_freeze_rejects_selection_inconsistent_with_predictions(self):
        selection, adapter = self.selection(), self.training_evidence()
        changed = workflow.read_json(selection)
        changed["scores"]["1"]["random"] = 0.9
        write_json(selection, changed)
        with self.assertRaisesRegex(ValueError, "score does not match"):
            workflow.freeze(self.prepared, self.config_path, selection, adapter, self.root / "frozen.json")

    def test_random_examples_are_order_independent_nested_and_answer_blind(self):
        query = self.validation[0]
        first = random_examples(self.pool, query, 1)
        five = random_examples(list(reversed(self.pool)), {**query, "Answer": "changed reference"}, 5)
        self.assertEqual(first, five[:1])
        self.assertTrue(all(row["Task"] == query["Task"] for row in five))

    def test_common_n_tie_break_ignores_binary_float_roundoff(self):
        results = {1: {"random": 0.3, "similar": 0.6}, 3: {"random": 0.4, "similar": 0.5}}
        self.assertEqual(choose_common_n(results), 1)


if __name__ == "__main__":
    unittest.main()
