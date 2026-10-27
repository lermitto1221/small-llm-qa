"""AMP safety/state tests; no GPU training or overflow recovery is claimed here."""

import json
import math
from pathlib import Path
import tempfile
import unittest

from small_llm_qa.training import AmpStepTracker, _checkpoint_for_resume, _sanitize_amp_logs


def attempt(tracker, *, skipped=False, scale_before=65536.0, scale_after=65536.0,
            gradients_finite=True, weights_finite=True):
    return tracker.record(step=tracker.attempted_steps + 1,
                          optimizer_step_skipped=skipped, scale_before=scale_before,
                          scale_after=scale_after, gradients_finite=gradients_finite,
                          weights_finite=weights_finite)


class AmpRecoveryTests(unittest.TestCase):
    def test_overflow_attempts_do_not_count_as_real_updates(self):
        tracker = AmpStepTracker()
        attempt(tracker, skipped=True, scale_before=65536, scale_after=32768, gradients_finite=False)
        attempt(tracker, skipped=True, scale_before=32768, scale_after=16384, gradients_finite=False)
        self.assertEqual(tracker.successful_updates, 0)
        with self.assertRaisesRegex(RuntimeError, "No actual optimizer update"):
            tracker.require_success(smoke=True)
        attempt(tracker, scale_before=16384, scale_after=16384)
        tracker.require_success(smoke=True)
        self.assertEqual(tracker.attempted_steps, 3)
        self.assertEqual(tracker.successful_updates, 1)
        self.assertEqual(tracker.skipped_steps, 2)
        self.assertEqual(tracker.consecutive_skips, 0)

    def test_sixteen_consecutive_skips_exhaust_bound_without_success(self):
        tracker = AmpStepTracker()
        scale = 65536.0
        for count in range(16):
            attempt(tracker, skipped=True, scale_before=scale, scale_after=scale / 2,
                    gradients_finite=False)
            scale /= 2
            if count < 15:
                tracker.check_skip_bound()
        with self.assertRaisesRegex(FloatingPointError, "16 consecutive"):
            tracker.check_skip_bound()
        with self.assertRaises(FloatingPointError):
            tracker.require_success(smoke=True)
        self.assertEqual(tracker.attempted_steps, 16)
        self.assertEqual(tracker.final_scale, 1.0)

    def test_skip_requires_a_finite_positive_strict_scaler_backoff(self):
        for before, after in ((None, None), (1024, 1024), (1024, 2048), (1024, 0),
                              (math.inf, 1024), (1024, math.nan), (None, 1024)):
            with self.subTest(before=before, after=after), self.assertRaises(FloatingPointError):
                attempt(AmpStepTracker(), skipped=True, scale_before=before, scale_after=after,
                        gradients_finite=False)

    def test_unskipped_bad_gradients_or_weights_remain_fatal(self):
        for gradients, weights in ((False, True), (True, False), (False, False)):
            with self.subTest(gradients=gradients, weights=weights), self.assertRaises(FloatingPointError):
                attempt(AmpStepTracker(), gradients_finite=gradients, weights_finite=weights)

    def test_success_without_scaler_is_supported_but_bad_gradients_are_not(self):
        tracker = AmpStepTracker()
        attempt(tracker, scale_before=None, scale_after=None)
        tracker.require_success(smoke=True)
        with self.assertRaises(FloatingPointError):
            attempt(tracker, scale_before=None, scale_after=None, gradients_finite=False)

    def test_smoke_cannot_report_more_than_one_real_update(self):
        tracker = AmpStepTracker()
        attempt(tracker)
        attempt(tracker)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            tracker.require_success(smoke=True)

    def test_scale_cannot_decrease_without_a_reported_skip(self):
        with self.assertRaisesRegex(FloatingPointError, "without an optimizer skip"):
            attempt(AmpStepTracker(), scale_before=65536, scale_after=32768)

    def test_proved_overflow_logs_remain_strict_json_with_raw_marker(self):
        tracker = AmpStepTracker()
        attempt(tracker, skipped=True, scale_before=65536, scale_after=32768, gradients_finite=False)
        original = {"loss": 2.25, "grad_norm": math.inf}
        clean = _sanitize_amp_logs(original, tracker, 1)
        # Model Trainer.log's copy into history, then the JSON checkpoint/history write.
        history = [{**clean, "step": 1}]
        restored = json.loads(json.dumps(history, allow_nan=False))
        self.assertIsNone(restored[0]["grad_norm"])
        self.assertEqual(restored[0]["grad_norm_nonfinite"], "inf")
        self.assertTrue(restored[0]["optimizer_step_skipped"])
        self.assertEqual(restored[0]["successful_optimizer_updates"], 0)
        self.assertTrue(math.isinf(original["grad_norm"]))

    def test_nonfinite_loss_is_never_excused_by_overflow_skip(self):
        tracker = AmpStepTracker()
        attempt(tracker, skipped=True, scale_before=65536, scale_after=32768, gradients_finite=False)
        for field in ("loss", "eval_loss", "train_loss"):
            with self.subTest(field=field), self.assertRaises(FloatingPointError):
                _sanitize_amp_logs({field: math.nan, "grad_norm": math.inf}, tracker, 1)

    def test_unskipped_or_stale_nonfinite_norm_is_not_sanitized(self):
        updated = AmpStepTracker()
        attempt(updated)
        with self.assertRaises(FloatingPointError):
            _sanitize_amp_logs({"grad_norm": math.inf}, updated, 1)
        skipped = AmpStepTracker()
        attempt(skipped, skipped=True, scale_before=65536, scale_after=32768, gradients_finite=False)
        with self.assertRaises(FloatingPointError):
            _sanitize_amp_logs({"grad_norm": math.inf}, skipped, 2)

    def test_checkpoint_roundtrip_preserves_counts_and_continues_next_attempt(self):
        original = AmpStepTracker()
        attempt(original, skipped=True, scale_before=65536, scale_after=32768, gradients_finite=False)
        state = json.loads(json.dumps(original.state_dict(), allow_nan=False))
        restored = AmpStepTracker.from_state_dict(state, expected_step=1)
        attempt(restored, scale_before=32768, scale_after=32768)
        restored.require_success()
        self.assertEqual(restored.summary()["attempted_optimizer_steps"], 2)
        self.assertEqual(restored.summary()["successful_optimizer_updates"], 1)
        self.assertEqual(restored.summary()["amp_scale_initial"], 65536)

    def test_checkpoint_counters_must_match_saved_trainer_step(self):
        tracker = AmpStepTracker()
        attempt(tracker)
        with self.assertRaises(ValueError):
            AmpStepTracker.from_state_dict(tracker.state_dict(), expected_step=2)
        corrupted = tracker.state_dict()
        corrupted["successful_updates"] = 0
        with self.assertRaises(ValueError):
            AmpStepTracker.from_state_dict(corrupted, expected_step=1)

    def test_missing_amp_state_falls_back_to_older_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for step in (100, 200):
                folder = output / f"checkpoint-{step}"
                folder.mkdir()
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json",
                             "adapter_model.safetensors", "scaler.pt"):
                    (folder / name).write_bytes(b"test fixture")
                (folder / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
                (folder / "complete.json").write_text(json.dumps({"global_step": step, "manifest_sha256": "same"}), encoding="utf-8")
            (output / "checkpoint-100" / "amp_state.json").write_text("{}", encoding="utf-8")
            selected = _checkpoint_for_resume(output, "auto", "same", require_scaler=True, require_amp_state=True)
            self.assertEqual(Path(selected).name, "checkpoint-100")


if __name__ == "__main__":
    unittest.main()
