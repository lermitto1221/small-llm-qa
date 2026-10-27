"""Mask/padding tests check answer-only supervision independently of GPU training."""

import json
from pathlib import Path
import tempfile
import unittest

from small_llm_qa.training import (IGNORE_INDEX, TrainingLengthExceeded, _checkpoint_for_resume,
                                    _start_timing_session, _timing_summary, _update_timing_session,
                                    encode_assistant_example, pad_features)
from test_prompts import FakeTokenizer, row


class MaskTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_each_task_masks_full_prompt_and_supervises_answer_eos_only(self):
        for task, answer in (("qa", "Paris"), ("rag", "UNANSWERABLE"), ("math", "-3/4")):
            with self.subTest(task=task):
                encoded = encode_assistant_example(row(task=task, answer=answer), self.tokenizer)
                prefix = encoded["mask_metadata"]["prompt_tokens"]
                self.assertTrue(all(label == IGNORE_INDEX for label in encoded["labels"][:prefix]))
                self.assertEqual(encoded["labels"][prefix:], encoded["input_ids"][prefix:])
                supervised = [label for label in encoded["labels"] if label != IGNORE_INDEX]
                self.assertEqual(self.tokenizer.decode(supervised), answer + "<|im_end|>")
                self.assertEqual(supervised[-1], self.tokenizer.eos_token_id)
                self.assertTrue(encoded["mask_metadata"]["masked_prefix_tail"].endswith("<think>\n\n</think>\n\n"))

    def test_no_silent_sft_truncation(self):
        encoded = encode_assistant_example(row(), self.tokenizer)
        with self.assertRaises(TrainingLengthExceeded):
            encode_assistant_example(row(), self.tokenizer, len(encoded["input_ids"]) - 1)

    def test_template_answer_transformation_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            encode_assistant_example(row(answer="\nParis"), self.tokenizer)

    def test_internal_eos_cannot_end_answer_early(self):
        with self.assertRaisesRegex(ValueError, "internal EOS"):
            encode_assistant_example(row(answer="Paris<|im_end|>London"), self.tokenizer)

    def test_padding_is_masked_even_when_pad_equals_eos(self):
        short = encode_assistant_example(row(answer="4"), self.tokenizer)
        long = encode_assistant_example(row(answer="123456"), self.tokenizer)
        batch = pad_features([short, long], self.tokenizer.eos_token_id)
        end = len(short["input_ids"])
        self.assertEqual(batch["labels"][0][end - 1], self.tokenizer.eos_token_id)
        self.assertTrue(all(value == IGNORE_INDEX for value in batch["labels"][0][end:]))
        self.assertTrue(all(value == 0 for value in batch["attention_mask"][0][end:]))
        self.assertTrue(all(value == self.tokenizer.eos_token_id for value in batch["input_ids"][0][end:]))
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))


class CheckpointTests(unittest.TestCase):
    def make_checkpoint(self, root, step, complete=True, fingerprint="expected"):
        directory = root / f"checkpoint-{step}"
        directory.mkdir()
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json", "adapter_model.safetensors"):
            (directory / name).write_bytes(b"fixture")
        (directory / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        if complete:
            (directory / "complete.json").write_text(json.dumps({"global_step": step, "manifest_sha256": fingerprint}), encoding="utf-8")
        return directory

    def test_resume_chooses_latest_complete_checkpoint_after_interrupted_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = self.make_checkpoint(root, 100)
            self.make_checkpoint(root, 200, complete=False)
            self.assertEqual(Path(_checkpoint_for_resume(root, "auto", "expected")), older)

    def test_incomplete_checkpoint_cannot_silently_restart_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self.make_checkpoint(root, 100)
            (checkpoint / "optimizer.pt").unlink()
            with self.assertRaisesRegex(ValueError, "No complete checkpoint"):
                _checkpoint_for_resume(root, "auto", "expected")

    def test_foreign_checkpoint_and_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            foreign = self.make_checkpoint(root, 100)
            with self.assertRaisesRegex(ValueError, "directly inside"):
                _checkpoint_for_resume(run, str(foreign), "expected")
            self.make_checkpoint(run, 100, fingerprint="different")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                _checkpoint_for_resume(run, "auto", "expected")


class TimingTests(unittest.TestCase):
    def test_interrupted_session_preserves_only_recorded_lower_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = _start_timing_session(output)
            _update_timing_session(output, first, 12.5)
            second = _start_timing_session(output)
            self.assertNotEqual(first, second)
            _update_timing_session(output, second, 7.0, "complete")
            summary = _timing_summary(output)
            self.assertEqual(summary["total_recorded_training_seconds"], 19.5)
            self.assertTrue(summary["timing_is_lower_bound"])
            sessions = json.loads((output / "training_sessions.json").read_text(encoding="utf-8"))["sessions"]
            self.assertEqual(sessions[0]["status"], "interrupted")

    def test_completed_and_handled_failure_sessions_have_measured_intervals(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = _start_timing_session(output)
            _update_timing_session(output, first, 2.0, "failed")
            second = _start_timing_session(output)
            _update_timing_session(output, second, 3.0, "complete")
            summary = _timing_summary(output)
            self.assertEqual(summary["total_recorded_training_seconds"], 5.0)
            self.assertFalse(summary["timing_is_lower_bound"])


if __name__ == "__main__":
    unittest.main()
