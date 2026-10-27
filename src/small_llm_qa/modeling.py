"""Lazy GPU model loading, immutable revisions, and shared greedy generation."""

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
import os
import re
import time
from typing import Any

from .prompts import PromptResult


QWEN_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "Qwen/Qwen3-1.7B"
    revision: str = QWEN_REVISION
    compute_dtype: str = "auto"
    seed: int = 42

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Model revision must be an immutable 40-character commit SHA")
        if self.compute_dtype not in {"auto", "bfloat16", "float16"}:
            raise ValueError("compute_dtype must be auto, bfloat16, or float16")


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    metadata: dict[str, Any]


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers",
                 "huggingface-hub", "sentence-transformers", "numpy", "safetensors", "jinja2", "matplotlib"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def load_tokenizer(config: ModelConfig):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision,
                                              trust_remote_code=False, use_fast=True)
    if tokenizer.eos_token != "<|im_end|>":
        raise ValueError("Expected Qwen chat EOS <|im_end|>; audit this tokenizer before use")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: ModelConfig | dict[str, Any], training: bool = False,
               adapter_path: str | None = None) -> LoadedModel:
    if isinstance(config, dict):
        config = ModelConfig(**config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this 4-bit experiment; select a Colab GPU runtime")
    if training and adapter_path is not None:
        raise ValueError("Resume training with Trainer checkpoints, not inference adapter_path")
    set_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
    dtype_name = ("bfloat16" if bf16_supported else "float16") if config.compute_dtype == "auto" else config.compute_dtype
    if dtype_name == "bfloat16" and not bf16_supported:
        raise RuntimeError("This GPU cannot run the frozen bfloat16 configuration; use the same GPU class or restart all controls with an explicitly revised configuration")
    dtype = getattr(torch, dtype_name)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = load_tokenizer(config)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, revision=config.revision, quantization_config=quantization,
        torch_dtype=dtype, device_map={"": torch.cuda.current_device()},
        trust_remote_code=False, attn_implementation="sdpa",
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Expected bitsandbytes 4-bit weights but model does not report them")
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    if not training:
        model.requires_grad_(False)
        model.eval()
    metadata = {
        "config": asdict(config), "resolved_revision": getattr(model.config, "_commit_hash", None) or config.revision,
        "compute_dtype": dtype_name, "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_version": torch.version.cuda, "dependencies": dependency_versions(),
        "quantization": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True},
        "load_seconds": time.perf_counter() - started, "adapter_path": adapter_path,
        "enable_thinking": False, "do_sample": False, "num_beams": 1,
    }
    return LoadedModel(model, tokenizer, metadata)


def generate_answer(loaded: LoadedModel, prompt: PromptResult,
                    max_new_tokens: int = 128) -> dict[str, Any]:
    import torch
    from transformers import GenerationConfig
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model, tokenizer = loaded.model, loaded.tokenizer
    model.eval()
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([prompt.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(inputs)
    # Transformers 4.51.3 can replace even explicit global-default values with
    # model defaults. Disable that fallback and repeat critical flags as kwargs.
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    generation_kwargs = {"use_model_defaults": False, "do_sample": False, "num_beams": 1}
    resolved_config, unused_kwargs = model._prepare_generation_config(generation_config, **generation_kwargs)
    generation_mode = resolved_config.get_generation_mode().value
    required = {"do_sample": False, "num_beams": 1, "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if unused_kwargs or generation_mode != "greedy_search" or any(
            getattr(resolved_config, key) != value for key, value in required.items()):
        raise RuntimeError("Resolved generation settings violate the fixed greedy/EOS policy")
    effective_config = {key: getattr(resolved_config, key) for key in (
        "do_sample", "num_beams", "max_new_tokens", "temperature", "top_k", "top_p",
        "bos_token_id", "pad_token_id", "eos_token_id", "use_cache")}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        sequence = model.generate(input_ids=inputs, attention_mask=attention_mask,
                                  generation_config=resolved_config, **generation_kwargs)[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated = sequence[len(prompt.input_ids):].tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {
        "Answer": text.strip(), "generated_text": text,
        "raw_output": tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        "generated_token_ids": generated, "input_tokens": len(prompt.input_ids),
        "generation_mode": generation_mode, "use_model_defaults": False,
        "effective_generation_config": effective_config,
        "generated_tokens": len(generated), "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "hit_token_limit": len(generated) == max_new_tokens and (not generated or generated[-1] != tokenizer.eos_token_id),
    }
