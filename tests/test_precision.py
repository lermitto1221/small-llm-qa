"""Regression tests for native BF16 detection, using no real GPU dependencies.

PyTorch 2.6 defaults including_emulation=True. The mocked T4 deliberately accepts
BF16 tensor allocation but reports no native BF16 support, reproducing the
capability-detection issue separately from model-answer quality.
"""

from contextlib import contextmanager, redirect_stdout
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from small_llm_qa import modeling, workflow
from small_llm_qa.io_utils import file_hash, write_json
from test_prompts import FakeTokenizer


@contextmanager
def fake_gpu_libraries(native_bf16):
    """Emulation succeeds on both devices, while native support differs."""
    torch = ModuleType("torch")
    torch.float16 = "mock.float16"
    torch.bfloat16 = "mock.bfloat16"
    torch.version = SimpleNamespace(cuda="12.4")
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        cudnn=SimpleNamespace(allow_tf32=True, benchmark=True, deterministic=False),
    )
    torch.cuda = SimpleNamespace(
        is_available=Mock(return_value=True),
        is_bf16_supported=Mock(side_effect=lambda including_emulation=True:
                               True if including_emulation else native_bf16),
        current_device=Mock(return_value=0),
        get_device_name=Mock(return_value="Mock A100" if native_bf16 else "Mock T4"),
        get_device_properties=Mock(return_value=SimpleNamespace(
            major=8 if native_bf16 else 7, minor=0 if native_bf16 else 5,
            total_memory=16 * 1024 ** 3)),
    )
    model = SimpleNamespace(is_loaded_in_4bit=True, config=SimpleNamespace(_commit_hash=modeling.QWEN_REVISION),
                            requires_grad_=Mock(), eval=Mock())
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=Mock(return_value=model))
    transformers.BitsAndBytesConfig = Mock(return_value=SimpleNamespace())
    transformers.set_seed = Mock()
    modules = {"torch": torch, "transformers": transformers,
               **{name: ModuleType(name) for name in ("bitsandbytes", "peft", "sentence_transformers")}}
    with patch.dict(sys.modules, modules), \
         patch.object(modeling, "load_tokenizer", return_value=FakeTokenizer()), \
         patch.object(modeling, "dependency_versions", return_value={}):
        yield torch, transformers


class NativePrecisionTests(unittest.TestCase):
    def test_model_auto_uses_fp16_when_only_bf16_emulation_is_available(self):
        with fake_gpu_libraries(native_bf16=False) as (torch, transformers):
            loaded = modeling.load_model(modeling.ModelConfig(compute_dtype="auto"))
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            self.assertEqual(loaded.metadata["compute_dtype"], "float16")
            self.assertEqual(transformers.BitsAndBytesConfig.call_args.kwargs["bnb_4bit_compute_dtype"], torch.float16)
            self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], torch.float16)

    def test_model_auto_keeps_bf16_on_native_hardware(self):
        with fake_gpu_libraries(native_bf16=True) as (torch, transformers):
            loaded = modeling.load_model(modeling.ModelConfig(compute_dtype="auto"))
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            self.assertEqual(loaded.metadata["compute_dtype"], "bfloat16")
            self.assertEqual(transformers.BitsAndBytesConfig.call_args.kwargs["bnb_4bit_compute_dtype"], torch.bfloat16)
            self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], torch.bfloat16)

    def test_frozen_bf16_is_rejected_on_emulation_only_hardware(self):
        with fake_gpu_libraries(native_bf16=False) as (torch, transformers):
            with self.assertRaisesRegex(RuntimeError, "frozen bfloat16"):
                modeling.load_model(modeling.ModelConfig(compute_dtype="bfloat16"))
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_environment_snapshot_reports_native_support(self):
        with tempfile.TemporaryDirectory() as temporary, \
             fake_gpu_libraries(native_bf16=False) as (torch, _), \
             patch.object(workflow.subprocess, "check_output", return_value="mock-only\n"), \
             redirect_stdout(io.StringIO()):
            output = Path(temporary) / "environment.json"
            result = workflow.environment_snapshot(output)
            torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
            self.assertIs(result["bf16_supported"], False)
            self.assertIs(json.loads(output.read_text(encoding="utf-8"))["bf16_supported"], False)

    def test_gpu_preflight_freezes_native_precision_for_each_hardware_class(self):
        for native_bf16, expected_dtype in ((False, "float16"), (True, "bfloat16")):
            with self.subTest(native_bf16=native_bf16), tempfile.TemporaryDirectory() as temporary, \
                 fake_gpu_libraries(native_bf16) as (torch, _), redirect_stdout(io.StringIO()):
                root = Path(temporary)
                prepared = root / "prepared"
                prepared.mkdir()
                hashes = {}
                for split, task in (("train", "qa"), ("validation", "rag"), ("test", "math")):
                    row = {"Sample_id": split, "Task": task, "Question": "What is one plus one?",
                           "Context": "[1] One plus one is two." if task == "rag" else "", "Answer": "2"}
                    path = prepared / f"{split}.csv"
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    hashes[path.name] = file_hash(path)
                write_json(prepared / "manifest.json", {"artifact_sha256": hashes})
                config_path = root / "config.json"
                write_json(config_path, {"model": asdict(modeling.ModelConfig(compute_dtype="auto")),
                                         "input_budget": 2048, "training": {"max_length": 2048}})
                result = workflow.preflight(prepared, config_path, root / "preflight.json", gpu=True)
                torch.cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
                self.assertEqual(result["status"], "passed")
                frozen = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
                self.assertEqual(frozen["model"]["compute_dtype"], expected_dtype)


if __name__ == "__main__":
    unittest.main()
