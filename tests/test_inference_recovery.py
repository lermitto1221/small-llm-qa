"""Synthetic provenance tests only; no model, training, or held-out answers."""
import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from small_llm_qa import recovery, workflow
from small_llm_qa.io_utils import file_hash, tree_hash, write_json
from small_llm_qa.metrics import score_records


class InferenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / "old_runs"
        self.new = self.root / "new_runs"
        self.old_source = self.root / "old_source"
        self.source = self.root / "new_source"
        self.prepared = self.root / "prepared"
        for directory in (self.parent / "sft" / "adapter", self.old_source, self.source, self.prepared):
            directory.mkdir(parents=True)
        self.old_modules = {"__init__.py": "", "training.py": "def train(): return 'unchanged'\n",
                            "prompts.py": "PROMPT = 'unchanged'\n", "io_utils.py": "VALUE = 1\n",
                            "workflow.py": "def freeze(): return 'original'\n",
                            "modeling.py": "import math\ndef load_model(): return 1\ndef generate_answer(): return 'sample'\n"}
        for name, content in self.old_modules.items():
            (self.old_source / name).write_text(content, encoding="utf-8")
            (self.source / name).write_text(content, encoding="utf-8")
        (self.source / "modeling.py").write_text(self.old_modules["modeling.py"].replace("'sample'", "'greedy'"), encoding="utf-8")
        (self.source / "workflow.py").write_text("def freeze(): return 'verified recovery'\n", encoding="utf-8")
        (self.source / "recovery.py").write_text("RECOVERY = True\n", encoding="utf-8")
        self.rows = [{"Sample_id": f"V{i:03d}", "Task": ("qa", "rag", "math")[i % 3],
                      "Question": f"Synthetic {i}", "Context": "", "Answer": "2", "Accepted_answers": '["2"]'}
                     for i in range(180)]
        self.train = [{**self.rows[0], "Sample_id": "T000"}]
        self.test = [{key: value for key, value in {**self.rows[0], "Sample_id": "TEST000"}.items()
                      if key not in {"Answer", "Accepted_answers"}}]
        for name, rows in (("train", self.train), ("validation", self.rows), ("validation_tune", self.rows), ("test", self.test)):
            self.csv(self.prepared / f"{name}.csv", rows)
        manifest = {"train_ids": ["T000"], "validation_ids": [row["Sample_id"] for row in self.rows],
                    "tune_ids": [row["Sample_id"] for row in self.rows], "test_ids": ["TEST000"],
                    "artifact_sha256": {p.name: file_hash(p) for p in self.prepared.glob("*.csv")}}
        write_json(self.prepared / "manifest.json", manifest)
        self.dependencies = {"transformers": "4.51.3", "torch": "synthetic"}
        self.parent_code = recovery._source_hash(self.old_source)
        self.current_code = recovery._source_hash(self.source)
        self.config = {"schema_version": 1, "seed": 42, "model": {"name": "synthetic", "compute_dtype": "float16"},
                       "training": {"epochs": 1}, "input_budget": 2048, "max_new_tokens": 128,
                       "n_candidates": [1, 3, 5], "runtime_versions": self.dependencies,
                       "source_code_sha256": self.parent_code,
                       "data_manifest_sha256": file_hash(self.prepared / "manifest.json"),
                       "experiment_configuration_frozen": False, "dependency_status": "preflight_passed"}
        write_json(self.parent / "run_config.json", self.config)
        adapter = self.parent / "sft" / "adapter"
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"Synthetic adapter; never loaded")
        self.adapter_hash = tree_hash(adapter)
        shared = {"config_sha256": file_hash(self.parent / "run_config.json"), "code_sha256": self.parent_code,
                  "data_manifest_sha256": self.config["data_manifest_sha256"], "dependencies": self.dependencies}
        self.summary = {**shared, "status": "complete", "completed": True, "smoke": False,
                        "train_csv_sha256": file_hash(self.prepared / "train.csv"),
                        "validation_csv_sha256": file_hash(self.prepared / "validation.csv"),
                        "train_rows": 1, "validation_rows": 180, "adapter_sha256": self.adapter_hash}
        write_json(self.parent / "sft" / "training_summary.json", self.summary)
        self.summary_hash = file_hash(self.parent / "sft" / "training_summary.json")
        self.training_manifest = {**shared, "model": self.config["model"], "training": self.config["training"],
                                  "smoke": False, "train_sha256": self.summary["train_csv_sha256"],
                                  "validation_sha256": self.summary["validation_csv_sha256"]}
        write_json(self.parent / "sft" / "training_manifest.json", self.training_manifest)
        write_json(self.parent / "frozen_config.json", {**self.config, "experiment_configuration_frozen": True,
                   "selected_n": 1, "selection_sha256": "a" * 64, "adapter_path": "sft/adapter",
                   "adapter_sha256": self.adapter_hash, "training_summary_sha256": self.summary_hash,
                   "dependency_status": "validated_by_actual_validation_inference_and_completed_sft"})
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # Test-only anchors replace real user artifact hashes; production has no override.
        for name, value in (("PARENT_CODE_SHA256", self.parent_code), ("PARENT_SUMMARY_SHA256", self.summary_hash),
                            ("PARENT_ADAPTER_SHA256", self.adapter_hash)):
            self.stack.enter_context(patch.object(recovery, name, value))
        self.stack.enter_context(patch.object(recovery, "dependency_versions", return_value=self.dependencies))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def csv(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def prepare(self):
        return recovery.prepare_inference_recovery(self.parent, self.new, self.prepared, self.old_source, self.source)

    def verify(self, config_path=None):
        return recovery.verify_inference_recovery(config_path or self.new / "run_config.json", self.prepared, self.source)

    def test_preparation_preserves_original_bytes_and_is_idempotent(self):
        original = tree_hash(self.parent)
        with patch("small_llm_qa.training.run_training", side_effect=AssertionError("Must never train")):
            first, second = self.prepare(), self.prepare()
        self.assertEqual(first, second)
        self.assertEqual(tree_hash(self.parent), original)
        self.assertEqual(file_hash(first["training_summary_path"]), self.summary_hash)
        self.assertEqual(tree_hash(first["adapter_path"]), self.adapter_hash)
        config = workflow.read_json(first["config_path"])
        self.assertEqual({k: v for k, v in config.items() if k not in {"source_code_sha256", "inference_recovery"}},
                         {k: v for k, v in self.config.items() if k != "source_code_sha256"})
        self.assertTrue(self.verify()["training_reused"])
        self.assertFalse(self.verify()["training_rerun"])
        self.assertEqual(workflow.read_json(first["training_summary_path"])["code_sha256"], self.parent_code)

    def test_changed_shared_modules_or_model_loader_rejected_before_writes(self):
        changes = {"training.py": "def train(): return 'changed'\n", "prompts.py": "PROMPT = 'changed'\n",
                   "io_utils.py": "VALUE = 2\n", "modeling.py": "import math\ndef load_model(): return 2\ndef generate_answer(): return 'greedy'\n"}
        for name, changed in changes.items():
            path, original = self.source / name, (self.source / name).read_bytes()
            with self.subTest(name=name):
                path.write_text(changed, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.prepare()
                self.assertFalse(self.new.exists())
            path.write_bytes(original)

    def test_added_or_removed_module_is_rejected(self):
        extra = self.source / "hidden_training_change.py"
        extra.write_text("x=1", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only recovery.py"):
            self.prepare()
        extra.unlink()
        (self.source / "prompts.py").unlink()
        with self.assertRaisesRegex(ValueError, "only recovery.py"):
            self.prepare()

    def test_parent_source_does_not_match_anchor(self):
        (self.old_source / "training.py").write_text("x=1", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recorded original run"):
            self.prepare()

    def test_changed_dependency_versions_rejected(self):
        with patch.object(recovery, "dependency_versions", return_value={"transformers": "different"}):
            with self.assertRaisesRegex(ValueError, "dependency versions"):
                self.prepare()

    def test_parent_frozen_settings_must_match_unfrozen_config(self):
        path = self.parent / "frozen_config.json"
        value = workflow.read_json(path)
        value["max_new_tokens"] = 7
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "changed experiment settings"):
            self.prepare()

    def test_unfinished_or_smoke_training_rejected_even_with_authenticated_summary(self):
        path = self.parent / "sft" / "training_summary.json"
        frozen_path = self.parent / "frozen_config.json"
        for changes in ({"smoke": True}, {"completed": False}, {"status": "smoke_passed"}, {"validation_rows": 3}):
            write_json(path, {**self.summary, **changes})
            digest = file_hash(path)
            frozen = workflow.read_json(frozen_path)
            frozen["training_summary_sha256"] = digest
            write_json(frozen_path, frozen)
            with self.subTest(changes=changes), patch.object(recovery, "PARENT_SUMMARY_SHA256", digest):
                with self.assertRaisesRegex(ValueError, "completed full SFT|complete split"):
                    self.prepare()

    def test_training_manifest_must_authenticate_data_model_and_settings(self):
        path = self.parent / "sft" / "training_manifest.json"
        for changes in ({"model": {"name": "changed"}}, {"training": {"epochs": 2}}, {"code_sha256": "0" * 64},
                        {"train_sha256": "0" * 64}, {"validation_sha256": "0" * 64}, {"dependencies": {}}):
            write_json(path, {**self.training_manifest, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare()

    def test_recovery_settings_cannot_be_changed(self):
        self.prepare()
        path = self.new / "run_config.json"
        original = workflow.read_json(path)
        for changes in ({"max_new_tokens": 2}, {"model": {"name": "other"}}, {"training": {"epochs": 2}},
                        {"source_code_sha256": "0" * 64}, {"runtime_versions": {}}):
            write_json(path, {**original, **changes})
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.verify()

    def test_copied_or_original_evidence_mutation_refused(self):
        self.prepare()
        paths = [self.new / "sft" / "training_summary.json", self.new / "sft" / "adapter" / "adapter_model.safetensors",
                 self.parent / "sft" / "training_summary.json", self.parent / "sft" / "adapter" / "adapter_model.safetensors"]
        for path in paths:
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.verify()
            path.write_bytes(original)

    def test_prepared_artifact_mutation_refused(self):
        self.prepare()
        path = self.prepared / "train.csv"
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Prepared artifact changed"):
            self.verify()

    def test_parent_run_directory_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, "separate run directory"):
            recovery.prepare_inference_recovery(self.parent, self.parent, self.prepared, self.old_source, self.source)

    def test_interrupted_adapter_copy_retries_without_overwriting_parent(self):
        real_copy = shutil.copytree
        def interrupted(source, dest):
            Path(dest).mkdir()
            (Path(dest) / "partial").write_bytes(b"partial")
            raise OSError("Simulated Drive disconnect")
        with patch.object(recovery.shutil, "copytree", side_effect=interrupted):
            with self.assertRaisesRegex(OSError, "Simulated"):
                self.prepare()
        self.assertFalse((self.new / "sft" / "adapter").exists())
        self.assertEqual(len(list((self.new / "sft").glob(".adapter-copy-*"))), 1)
        with patch.object(recovery.shutil, "copytree", side_effect=real_copy):
            self.prepare()
        self.assertEqual(tree_hash(self.new / "sft" / "adapter"), self.adapter_hash)

    def test_invalid_recovery_blocks_each_entry_before_model_loading(self):
        self.prepare()
        with patch.object(recovery, "verify_inference_recovery", side_effect=ValueError("Recovery rejected")), \
             patch("small_llm_qa.modeling.load_model", side_effect=AssertionError("GPU must not load")), \
             patch("small_llm_qa.modeling.load_tokenizer", side_effect=AssertionError("Tokenizer must not load")):
            for call in (lambda: workflow.preflight(self.prepared, self.new / "run_config.json", self.new / "preflight.json"),
                         lambda: workflow.predict(self.prepared, self.new / "run_config.json", self.new / "predict"),
                         lambda: workflow.tune(self.prepared, self.new / "run_config.json", self.new / "tuning")):
                with self.assertRaisesRegex(ValueError, "Recovery rejected"):
                    call()

    def selection(self):
        config_path = self.new / "run_config.json"
        scores, evidence = {}, {}
        predictions = [{"Sample_id": row["Sample_id"], "Answer": "2"} for row in self.rows]
        report = score_records(self.rows, predictions)
        for n in (1, 3, 5):
            scores[str(n)] = {}
            for method in ("random", "similar"):
                key = f"{method}_{n}"
                folder = self.new / "tuning" / key
                self.csv(folder / "predictions.csv", predictions)
                write_json(folder / "evaluation.json", report)
                write_json(folder / "run_manifest.json", {"completed": True, "config_sha256": file_hash(config_path),
                           "n": n, "method": method})
                evidence[key] = {f"{name}_sha256": file_hash(folder / artifact) for name, artifact in
                                 (("evaluation", "evaluation.json"), ("run_manifest", "run_manifest.json"),
                                  ("prediction", "predictions.csv"))}
                scores[str(n)][method] = report["metrics"]["macro_score"]
        path = self.new / "tuning" / "selection.json"
        write_json(path, {"config_sha256": file_hash(config_path), "code_sha256": self.current_code,
                   "data_manifest_sha256": self.config["data_manifest_sha256"],
                   "validation_ids": [row["Sample_id"] for row in self.rows], "scores": scores,
                   "evidence": evidence, "selected_n": 1})
        return path

    def test_freeze_preserves_parent_training_provenance_and_uses_new_inference_code(self):
        self.prepare()
        selection = self.selection()
        frozen = self.new / "frozen_config.json"
        real_verify = recovery.verify_inference_recovery
        with patch.object(recovery, "verify_inference_recovery", side_effect=lambda c, p: real_verify(c, p, self.source)), \
             patch.object(workflow, "code_fingerprint", return_value=self.current_code):
            workflow.freeze(self.prepared, self.new / "run_config.json", selection, self.new / "sft" / "adapter", frozen)
        value = workflow.read_json(frozen)
        self.assertEqual(value["source_code_sha256"], self.current_code)
        self.assertEqual(value["training_summary_sha256"], self.summary_hash)
        self.assertEqual(self.verify(frozen)["parent_code_sha256"], self.parent_code)
        self.assertEqual(workflow.read_json(self.new / "sft" / "training_summary.json"), self.summary)
        self.assertEqual(value["adapter_path"], "sft/adapter")

    def test_freeze_rejects_relocated_output_and_unverified_original_adapter(self):
        self.prepare()
        selection = self.selection()
        real_verify = recovery.verify_inference_recovery
        with patch.object(recovery, "verify_inference_recovery", side_effect=lambda c, p: real_verify(c, p, self.source)), \
             patch.object(workflow, "code_fingerprint", return_value=self.current_code):
            with self.assertRaisesRegex(ValueError, "beside"):
                workflow.freeze(self.prepared, self.new / "run_config.json", selection, self.new / "sft" / "adapter", self.root / "frozen.json")
            with self.assertRaisesRegex(ValueError, "verified unchanged SFT copy"):
                workflow.freeze(self.prepared, self.new / "run_config.json", selection, self.parent / "sft" / "adapter", self.new / "frozen_config.json")


if __name__ == "__main__":
    unittest.main()
