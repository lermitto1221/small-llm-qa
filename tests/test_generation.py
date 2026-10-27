"""Pinned-library CPU regression for model-default sampling overrides.

Run with the tokenizer/runtime test environment containing transformers==4.51.3.
The real upstream config resolver is extracted without importing GPU model code.
No model inference or CUDA execution is claimed by these tests.
"""

import ast
import contextlib
import copy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Dict, Optional, Tuple
import unittest
from unittest.mock import patch
import warnings

from small_llm_qa.modeling import LoadedModel, generate_answer

try:
    import transformers
    from transformers import GenerationConfig
    from packaging import version
except ImportError:
    transformers = None


def actual_upstream_resolver():
    """Compile the installed pinned method unchanged, not a reimplementation."""
    source = Path(transformers.__file__).parent / "generation" / "utils.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    mixin = next(node for node in parsed.body if isinstance(node, ast.ClassDef)
                 and node.name == "GenerationMixin")
    method = next(node for node in mixin.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_prepare_generation_config")
    namespace = {"copy": copy, "version": version, "warnings": warnings,
                 "Optional": Optional, "Dict": Dict, "Tuple": Tuple,
                 "GenerationConfig": GenerationConfig,
                 "logger": SimpleNamespace(warning_once=lambda message: None)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_prepare_generation_config"]


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return FakeTensor(self.values[key])

    def __len__(self):
        return len(self.values)

    def tolist(self):
        return self.values


def fake_torch():
    module = ModuleType("torch")
    module.long = "fake-long"
    module.tensor = lambda values, **kwargs: FakeTensor(values)
    module.ones_like = lambda value: value
    module.inference_mode = contextlib.nullcontext
    module.cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
                                  max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    return module


def qwen_generation_defaults():
    # Exact pinned model defaults, read from the official immutable revision:
    # https://huggingface.co/Qwen/Qwen3-1.7B/raw/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/generation_config.json
    return GenerationConfig(bos_token_id=151643, do_sample=True, eos_token_id=[151645, 151643],
                            pad_token_id=151643, temperature=0.6, top_k=20, top_p=0.95,
                            transformers_version="4.51.0")


class ConfigResolvingModel:
    """No weights: both precheck and pretend generate use the actual resolver."""
    def __init__(self, resolver):
        self.resolver = resolver
        self.generation_config = qwen_generation_defaults()
        self.generate_calls = []
        self.resolved_calls = []

    def eval(self):
        return self

    def get_input_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(device="fake-cpu"))

    def _prepare_generation_config(self, generation_config, **kwargs):
        return self.resolver(self, generation_config, **kwargs)

    def generate(self, *, input_ids, attention_mask, generation_config, **kwargs):
        self.generate_calls.append(kwargs)
        resolved, unused = self.resolver(self, generation_config, **kwargs)
        if unused:
            raise AssertionError("Unexpected unused generation arguments")
        self.resolved_calls.append(resolved)
        return FakeTensor([input_ids.values[0] + [501, 151645]])


@unittest.skipUnless(transformers is not None and transformers.__version__ == "4.51.3",
                     "Requires pinned transformers==4.51.3 config utilities (no torch needed)")
class GenerationPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = staticmethod(actual_upstream_resolver())

    def loaded(self):
        model = ConfigResolvingModel(self.resolver)
        tokenizer = SimpleNamespace(pad_token_id=151643, eos_token_id=151645,
            decode=lambda ids, skip_special_tokens, **kwargs: "answer" if skip_special_tokens else "answer<|im_end|>")
        return LoadedModel(model, tokenizer, {})

    def test_original_generation_config_false_is_actually_overridden_to_sampling(self):
        model = ConfigResolvingModel(self.resolver)
        requested = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=128,
                                     pad_token_id=151643, eos_token_id=151645, use_cache=True)
        actual, _ = self.resolver(model, requested)
        self.assertFalse(requested.do_sample)
        self.assertTrue(actual.do_sample)
        self.assertEqual(actual.get_generation_mode().value, "sample")
        self.assertEqual((actual.temperature, actual.top_k, actual.top_p), (0.6, 20, 0.95))

    def test_generate_answer_resolves_greedy_and_records_effective_config(self):
        loaded = self.loaded()
        model_before = loaded.model.generation_config.to_dict()
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            record = generate_answer(loaded, SimpleNamespace(input_ids=[101, 102]), 128)
        self.assertEqual(len(loaded.model.generate_calls), 1)
        self.assertEqual(loaded.model.generate_calls[0],
                         {"use_model_defaults": False, "do_sample": False, "num_beams": 1})
        actual = loaded.model.resolved_calls[0]
        self.assertEqual(actual.get_generation_mode().value, "greedy_search")
        self.assertFalse(actual.do_sample)
        self.assertEqual(actual.max_new_tokens, 128)
        self.assertEqual(actual.eos_token_id, 151645)
        self.assertEqual((actual.temperature, actual.top_k, actual.top_p), (1.0, 50, 1.0))
        self.assertEqual(record["generation_mode"], "greedy_search")
        self.assertFalse(record["use_model_defaults"])
        for key, value in record["effective_generation_config"].items():
            self.assertEqual(value, getattr(actual, key))
        self.assertEqual(record["generated_token_ids"], [501, 151645])
        self.assertEqual(loaded.model.generation_config.to_dict(), model_before)
        json.dumps(record, allow_nan=False)

    def test_each_call_is_greedy_without_mutating_model_defaults(self):
        loaded = self.loaded()
        with patch.dict(sys.modules, {"torch": fake_torch()}):
            first = generate_answer(loaded, SimpleNamespace(input_ids=[101]), 32)
            second = generate_answer(loaded, SimpleNamespace(input_ids=[101]), 64)
        self.assertTrue(loaded.model.generation_config.do_sample)
        self.assertEqual([config.max_new_tokens for config in loaded.model.resolved_calls], [32, 64])
        self.assertTrue(all(config.get_generation_mode().value == "greedy_search"
                            for config in loaded.model.resolved_calls))
        self.assertEqual(first["generation_mode"], second["generation_mode"])

    def test_unexpected_resolved_policy_stops_before_generate(self):
        loaded = self.loaded()
        original = loaded.model._prepare_generation_config

        def tampered(*args, **kwargs):
            resolved, unused = original(*args, **kwargs)
            resolved.do_sample = True
            return resolved, unused

        loaded.model._prepare_generation_config = tampered
        with patch.dict(sys.modules, {"torch": fake_torch()}), self.assertRaisesRegex(RuntimeError, "greedy/EOS"):
            generate_answer(loaded, SimpleNamespace(input_ids=[101]), 128)
        self.assertEqual(loaded.model.generate_calls, [])


if __name__ == "__main__":
    unittest.main()
