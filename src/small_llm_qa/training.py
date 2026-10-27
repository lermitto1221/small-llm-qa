"""QLoRA with an explicitly verified assistant-only loss mask.

The standard Transformers Trainer consumes our labels directly. Qwen3's chat
template does not provide TRL generation-mask tags; relying on automatic role
masking would hide mistakes around the empty non-thinking prefix. No TRL import
or template patch is needed here. GPU imports stay inside run_training().
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid
from typing import Any, Mapping, Sequence

from .io_utils import append_journal, canonical_hash, file_hash, tree_hash
from .modeling import ModelConfig, dependency_versions, load_model, load_tokenizer
from .prompts import PROMPT_VERSION, build_messages, chat_render, field, sample_id, task_name


IGNORE_INDEX = -100
AMP_MAX_CONSECUTIVE_SKIPS = 16
SMOKE_MAX_ATTEMPTS = 16


class AmpStepTracker:
    """Count actual updates separately from Trainer's attempted-step counter."""

    def __init__(self, max_consecutive_skips: int = AMP_MAX_CONSECUTIVE_SKIPS):
        if max_consecutive_skips < 1:
            raise ValueError("AMP skip bound must be positive")
        self.max_consecutive_skips = max_consecutive_skips
        self.attempted_steps = 0
        self.successful_updates = 0
        self.skipped_steps = 0
        self.consecutive_skips = 0
        self.initial_scale = None
        self.final_scale = None
        self.latest = None

    def record(self, *, step: int, gradients_finite: bool, weights_finite: bool,
               optimizer_step_skipped: bool, scale_before: float | None,
               scale_after: float | None) -> dict[str, Any]:
        if step != self.attempted_steps + 1:
            raise ValueError("AMP attempt counter differs from Trainer global_step")
        if (scale_before is None) != (scale_after is None):
            raise FloatingPointError("AMP scaler appeared/disappeared during an optimizer attempt")
        for scale in (scale_before, scale_after):
            if scale is not None and (not math.isfinite(scale) or scale <= 0):
                raise FloatingPointError("AMP scale must be finite and positive")
        if not weights_finite:
            raise FloatingPointError("Non-finite trainable weights after optimizer attempt")
        if optimizer_step_skipped:
            if scale_before is None or not scale_after < scale_before:
                raise FloatingPointError("Optimizer skip was not accompanied by a verified GradScaler backoff")
            status = "amp_overflow_skipped"
        else:
            if not gradients_finite:
                raise FloatingPointError("Non-finite gradients on an optimizer attempt that was not skipped")
            if scale_before is not None and scale_after < scale_before:
                raise FloatingPointError("AMP scale decreased without an optimizer skip")
            status = "updated"
        if self.attempted_steps == 0:
            self.initial_scale = scale_before
        self.attempted_steps = step
        self.final_scale = scale_after
        if optimizer_step_skipped:
            self.skipped_steps += 1
            self.consecutive_skips += 1
        else:
            self.successful_updates += 1
            self.consecutive_skips = 0
        self.latest = {"attempted_step": step, "status": status,
                       "optimizer_step_skipped": optimizer_step_skipped,
                       "gradients_finite_after_clip": gradients_finite,
                       "trainable_weights_finite": weights_finite,
                       "scale_before": scale_before, "scale_after": scale_after,
                       "successful_optimizer_updates": self.successful_updates,
                       "amp_skipped_steps": self.skipped_steps,
                       "consecutive_amp_skips": self.consecutive_skips}
        return dict(self.latest)

    def check_skip_bound(self):
        if self.consecutive_skips >= self.max_consecutive_skips:
            raise FloatingPointError(f"AMP skipped {self.consecutive_skips} consecutive optimizer attempts; "
                                     "bounded recovery exhausted without a finite update")

    def require_success(self, smoke: bool = False):
        self.check_skip_bound()
        if self.successful_updates < 1:
            raise RuntimeError("No actual optimizer update occurred; attempted steps are not successful updates")
        if smoke and self.successful_updates != 1:
            raise RuntimeError("Smoke must stop after exactly one actual optimizer update")

    def state_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "max_consecutive_skips": self.max_consecutive_skips,
                "attempted_steps": self.attempted_steps, "successful_updates": self.successful_updates,
                "skipped_steps": self.skipped_steps, "consecutive_skips": self.consecutive_skips,
                "initial_scale": self.initial_scale, "final_scale": self.final_scale, "latest": self.latest}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], expected_step: int):
        tracker = cls()
        if state.get("schema_version") != 1 or state.get("max_consecutive_skips") != tracker.max_consecutive_skips:
            raise ValueError("Checkpoint AMP policy differs from this run")
        for name in ("attempted_steps", "successful_updates", "skipped_steps", "consecutive_skips"):
            value = state.get(name)
            if type(value) is not int or value < 0:
                raise ValueError("Invalid checkpoint AMP counters")
            setattr(tracker, name, value)
        if tracker.attempted_steps != expected_step or tracker.successful_updates + tracker.skipped_steps != expected_step:
            raise ValueError("Checkpoint AMP counters differ from Trainer state")
        if tracker.consecutive_skips > tracker.skipped_steps:
            raise ValueError("Invalid checkpoint consecutive AMP skip count")
        tracker.initial_scale, tracker.final_scale, tracker.latest = state["initial_scale"], state["final_scale"], state["latest"]
        for scale in (tracker.initial_scale, tracker.final_scale):
            if scale is not None and (not math.isfinite(scale) or scale <= 0):
                raise ValueError("Invalid checkpoint AMP scale")
        if expected_step and (not tracker.latest or tracker.latest["attempted_step"] != expected_step):
            raise ValueError("Checkpoint AMP last event differs from Trainer state")
        tracker.check_skip_bound()
        return tracker

    def summary(self) -> dict[str, Any]:
        return {"attempted_optimizer_steps": self.attempted_steps,
                "successful_optimizer_updates": self.successful_updates,
                "amp_skipped_steps": self.skipped_steps,
                "consecutive_amp_skips": self.consecutive_skips,
                "amp_scale_initial": self.initial_scale, "amp_scale_final": self.final_scale,
                "amp_max_consecutive_skips": self.max_consecutive_skips,
                "amp_counts_scope": "Current checkpoint trajectory. The per-session journal also preserves attempts rolled back after interruption."}


def _sanitize_amp_logs(logs: Mapping[str, Any], tracker: AmpStepTracker, step: int) -> dict[str, Any]:
    """Sanitize only proved overflow events before Trainer stores its log history."""
    clean = dict(logs)
    for name in ("loss", "eval_loss", "train_loss"):
        if name in clean and not math.isfinite(float(clean[name])):
            raise FloatingPointError(f"Non-finite {name} at attempted optimizer step {step}")
    if "grad_norm" in clean and clean["grad_norm"] is not None and not math.isfinite(float(clean["grad_norm"])):
        event = tracker.latest
        if not event or event["attempted_step"] != step or event["status"] != "amp_overflow_skipped":
            raise FloatingPointError(f"Non-finite grad_norm without a proved AMP skip at attempted step {step}")
        clean["grad_norm_nonfinite"] = str(float(clean["grad_norm"]))
        clean["grad_norm"] = None
    if "grad_norm" in clean and tracker.latest and tracker.latest["attempted_step"] == step:
        clean.update(optimizer_step_skipped=tracker.latest["optimizer_step_skipped"],
                     amp_scale_before=tracker.latest["scale_before"], amp_scale_after=tracker.latest["scale_after"],
                     successful_optimizer_updates=tracker.successful_updates)
    return clean


class TrainingLengthExceeded(ValueError):
    def __init__(self, row_id: str, tokens: int, limit: int):
        self.sample_id, self.tokens, self.limit = row_id, tokens, limit
        super().__init__(f"SFT sample {row_id!r} needs {tokens} tokens including the answer and EOS; "
                         f"max_length={limit}. No truncation or row dropping is allowed.")


def encode_assistant_example(row: Mapping[str, Any], tokenizer: Any,
                             max_length: int = 2048) -> dict[str, Any]:
    """Verify the inference prefix, answer, EOS and mask against the real template."""
    answer = field(row, "Answer")
    if not answer:
        raise ValueError(f"Empty SFT answer for {sample_id(row)!r}")
    messages = build_messages(row)
    prompt_text, prompt_ids = chat_render(tokenizer, messages, generation=True)
    full_text, _ = chat_render(tokenizer, messages + [{"role": "assistant", "content": answer}], generation=False)
    eos = tokenizer.eos_token
    if not eos or eos != "<|im_end|>" or tokenizer.eos_token_id is None:
        raise ValueError("SFT requires the audited Qwen <|im_end|> EOS")
    expected = prompt_text + answer + eos + "\n"
    if full_text != expected:
        raise ValueError(f"Chat template changed the non-thinking answer boundary for {sample_id(row)!r}; "
                         "expected inference prefix + unchanged answer + <|im_end|> + newline")
    # Exclude the template's newline after EOS. It is neither answer nor EOS.
    input_ids = tokenizer.encode(prompt_text + answer + eos, add_special_tokens=False)
    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("Tokenizer merged across the assistant boundary; mask requires an exact token prefix")
    target_ids = input_ids[len(prompt_ids):]
    if not target_ids or target_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("Assistant target must finish with the chat EOS")
    if tokenizer.eos_token_id in target_ids[:-1]:
        raise ValueError("Assistant answer unexpectedly contains an internal EOS")
    decoded = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != answer + eos:
        raise ValueError("Supervised tokens do not decode to exactly the answer plus EOS")
    if len(input_ids) > max_length:
        raise TrainingLengthExceeded(sample_id(row), len(input_ids), max_length)
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    if any(label != IGNORE_INDEX for label in labels[:len(prompt_ids)]):
        raise AssertionError("Prompt tokens leaked into the loss")
    if labels[-1] != tokenizer.eos_token_id or not any(label != IGNORE_INDEX for label in labels):
        raise AssertionError("Missing supervised answer/EOS")
    return {
        "input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels,
        "mask_metadata": {"Sample_id": sample_id(row), "Task": task_name(row),
                          "prompt_tokens": len(prompt_ids), "supervised_tokens": len(target_ids),
                          "total_tokens": len(input_ids), "supervised_text": decoded,
                          "masked_prefix_tail": prompt_text[-80:]},
    }


def pad_features(features: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad inputs and independently mask all padding, including EOS-as-pad."""
    if not features:
        raise ValueError("Cannot collate an empty batch")
    maximum = max(len(item["input_ids"]) for item in features)
    batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in features:
        length = len(item["input_ids"])
        if len(item["labels"]) != length or len(item["attention_mask"]) != length:
            raise ValueError("Input, attention, and label lengths differ")
        padding = maximum - length
        batch["input_ids"].append(list(item["input_ids"]) + [pad_token_id] * padding)
        batch["attention_mask"].append(list(item["attention_mask"]) + [0] * padding)
        batch["labels"].append(list(item["labels"]) + [IGNORE_INDEX] * padding)
    return batch


class AssistantOnlyCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in pad_features(features, self.pad_token_id).items()}


class EncodedDataset:
    def __init__(self, examples: Sequence[Mapping[str, Any]]):
        self.examples = [{key: item[key] for key in ("input_ids", "attention_mask", "labels")}
                         for item in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: float = 1.0
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    save_steps: int = 100
    logging_steps: int = 10

    def __post_init__(self):
        for name in ("epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "max_length",
                     "lora_rank", "lora_alpha", "save_steps", "logging_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0,1)")


def _read_rows(path: Path) -> list[dict[str, str]]:
    # Deliberately accept only prepared CSV, not arbitrary JSONL chat transcripts.
    # data.prepare audits CSV/JSONL correspondence and filters both by split IDs.
    if path.suffix.lower() != ".csv":
        raise ValueError("Training takes prepared train.csv/validation.csv; use the audited data preparation first")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [sample_id(row) for row in rows]
    if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Empty or duplicate training/validation IDs in {path}")
    if any(not field(row, "Answer") for row in rows):
        raise ValueError(f"Missing training/validation answers in {path}")
    return rows


def _write_json(path: Path, payload: Any):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _start_timing_session(output: Path) -> str:
    path = output / "training_sessions.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sessions": []}
    for session in report["sessions"]:
        if session["status"] == "running":
            # A process terminated without an exception handler cannot report the
            # interval after its last durable snapshot. Preserve that lower bound.
            session["status"] = "interrupted"
            session["timing_is_lower_bound"] = True
    identifier = str(uuid.uuid4())
    report["sessions"].append({"session_id": identifier, "started_utc": _utc_now(),
                               "seconds_recorded": 0.0, "status": "running",
                               "timing_is_lower_bound": True})
    _write_json(path, report)
    return identifier


def _update_timing_session(output: Path, identifier: str, elapsed: float, status: str = "running"):
    if status not in {"running", "complete", "failed"} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid training timing snapshot")
    path = output / "training_sessions.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    matches = [session for session in report["sessions"] if session["session_id"] == identifier]
    if len(matches) != 1:
        raise ValueError("Unknown or duplicate training timing session")
    session = matches[0]
    if elapsed < session["seconds_recorded"]:
        raise ValueError("Training session timing cannot go backwards")
    session.update(seconds_recorded=elapsed, status=status, updated_utc=_utc_now(),
                   timing_is_lower_bound=status == "running")
    _write_json(path, report)


def _timing_summary(output: Path) -> dict[str, Any]:
    report = json.loads((output / "training_sessions.json").read_text(encoding="utf-8"))
    return {"total_recorded_training_seconds": sum(session["seconds_recorded"] for session in report["sessions"]),
            "timing_is_lower_bound": any(session["timing_is_lower_bound"] for session in report["sessions"]),
            "training_timing_sessions": len(report["sessions"]),
            "timing_scope": "Wall time inside Trainer.train, including in-training validation/checkpoint saves; excludes loading and final validation. Interrupted sessions retain only their last recorded interval."}


def _verify_prepared_data(train_path: Path, validation_path: Path, train_rows, validation_rows) -> str:
    if train_path.resolve().parent != validation_path.resolve().parent:
        raise ValueError("Training and validation CSV must belong to the same prepared-data directory")
    if train_path.name != "train.csv" or validation_path.name != "validation.csv":
        raise ValueError("Use the prepared train.csv and validation.csv splits")
    manifest_path = train_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for path, split, rows in ((train_path, "train", train_rows), (validation_path, "validation", validation_rows)):
        if file_hash(path) != manifest["artifact_sha256"].get(path.name):
            raise ValueError(f"Prepared {split} data has changed after its audited split")
        if {sample_id(row) for row in rows} != set(manifest[f"{split}_ids"]):
            raise ValueError(f"Prepared {split} IDs do not match the split manifest")
    return file_hash(manifest_path)


def _checkpoint_for_resume(output: Path, resume: str, manifest_sha256: str,
                           require_scaler: bool = False, require_amp_state: bool = False) -> str | None:
    if resume == "none":
        return None
    if resume == "auto":
        candidates = sorted((item for item in output.glob("checkpoint-*")
                             if item.is_dir() and re.fullmatch(r"checkpoint-\d+", item.name)),
                            key=lambda item: int(item.name.split("-")[1]), reverse=True)
    else:
        candidate = Path(resume).resolve()
        if candidate.parent != output.resolve() or not re.fullmatch(r"checkpoint-\d+", candidate.name):
            raise ValueError("Manual checkpoint must be checkpoint-N directly inside this training output directory")
        candidates = [candidate]
    for checkpoint in candidates:
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                    "adapter_config.json", "adapter_model.safetensors", "complete.json")
        if require_scaler:
            required += ("scaler.pt",)
        if require_amp_state:
            required += ("amp_state.json",)
        if not all((checkpoint / name).is_file() for name in required):
            if resume != "auto":
                raise ValueError(f"Incomplete checkpoint: {checkpoint}")
            continue
        state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        completion = json.loads((checkpoint / "complete.json").read_text(encoding="utf-8"))
        step = int(checkpoint.name.split("-")[1])
        if state["global_step"] != step or completion != {"global_step": step, "manifest_sha256": manifest_sha256}:
            raise ValueError(f"Checkpoint state/provenance mismatch: {checkpoint}")
        return str(checkpoint)
    if candidates:
        raise ValueError("No complete checkpoint is available; preserve this failed run and use a fresh output directory")
    return None


def _balanced_examples(rows: Sequence[Mapping[str, Any]], per_task: int = 1) -> list[Mapping[str, Any]]:
    return [row for task in ("qa", "rag", "math")
            for row in [item for item in rows if task_name(item) == task][:per_task]]


def _encode_all(rows, tokenizer, limit):
    encoded, oversize = [], []
    for row in rows:
        try:
            encoded.append(encode_assistant_example(row, tokenizer, limit))
        except TrainingLengthExceeded as error:
            oversize.append({"Sample_id": error.sample_id, "tokens": error.tokens, "limit": error.limit})
    return encoded, oversize


def _write_loss_history(output_dir: Path, history: list[dict[str, Any]]):
    _write_json(output_dir / "loss_history.json", history)
    with (output_dir / "loss_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "epoch", "loss", "eval_loss", "learning_rate", "grad_norm"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(figsize=(7, 4))
    for key, title in (("loss", "Training loss"), ("eval_loss", "Validation loss")):
        points = [entry for entry in history if key in entry]
        if points:
            axes.plot([entry["step"] for entry in points], [entry[key] for entry in points], marker=".", label=title)
    axes.set(xlabel="Optimizer step", ylabel="Assistant-token cross entropy")
    axes.grid(alpha=0.2)
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(figure)


def run_training(train_csv: str | Path, validation_csv: str | Path, output_dir: str | Path,
                 model_config: ModelConfig | None = None, training_config: TrainingConfig | None = None,
                 resume: str = "auto", smoke: bool = False,
                 config_sha256: str | None = None) -> dict[str, Any]:
    """Train one final-epoch adapter; validation loss is logged, never test answers."""
    model_config = model_config or ModelConfig()
    settings = training_config or TrainingConfig()
    if model_config.compute_dtype == "auto":
        raise ValueError("Run GPU preflight first and use its explicit float16/bfloat16 configuration for training")
    train_path, validation_path, output = Path(train_csv), Path(validation_csv), Path(output_dir)
    train_rows, validation_rows = _read_rows(train_path), _read_rows(validation_path)
    overlap = {sample_id(row) for row in train_rows} & {sample_id(row) for row in validation_rows}
    if overlap:
        raise ValueError(f"Train/validation ID leakage: {sorted(overlap)[:10]}")
    data_manifest_sha256 = _verify_prepared_data(train_path, validation_path, train_rows, validation_rows)
    code_sha256 = canonical_hash({path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))})
    manifest = {"model": asdict(model_config), "training": asdict(settings), "smoke": smoke,
                "train_sha256": _digest(train_path), "validation_sha256": _digest(validation_path),
                "config_sha256": config_sha256,
                "data_manifest_sha256": data_manifest_sha256, "code_sha256": code_sha256,
                "dependencies": dependency_versions(),
                "amp_policy": {"max_consecutive_skips": AMP_MAX_CONSECUTIVE_SKIPS,
                    "initial_scale": "unchanged PyTorch/Accelerate default",
                    "smoke_max_attempts": SMOKE_MAX_ATTEMPTS, "smoke_target_updates": 1,
                    "smoke_samples_per_task": 8, "smoke_checkpoints": False},
                "prompt_version": PROMPT_VERSION, "selection_rule": "final_epoch; no best-checkpoint selection"}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Output directory belongs to different data/configuration; choose a fresh directory")
        if resume == "none":
            raise ValueError("Output already contains a run; use resume=auto or a fresh directory")
        summary_path = output / "training_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_adapter = output / ("smoke_adapter_do_not_evaluate" if smoke else "adapter")
            if not saved_adapter.is_dir() or tree_hash(saved_adapter) != summary.get("adapter_sha256"):
                raise ValueError("Completed adapter is missing or changed; do not reuse this completion marker")
            return summary
        if smoke:
            raise ValueError("Incomplete smoke runs do not resume; preserve this diagnostic and use a fresh output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Training output directory is nonempty but has no matching manifest")
        _write_json(manifest_path, manifest)
    manifest_sha256 = canonical_hash(manifest)
    checkpoint = _checkpoint_for_resume(output, resume, manifest_sha256,
                                        require_scaler=model_config.compute_dtype == "float16",
                                        require_amp_state=True)
    if smoke and checkpoint:
        raise ValueError("Smoke checkpoints are not supported; use a fresh diagnostic directory")
    amp = AmpStepTracker()
    if checkpoint:
        checkpoint_state = json.loads((Path(checkpoint) / "trainer_state.json").read_text(encoding="utf-8"))
        amp = AmpStepTracker.from_state_dict(
            json.loads((Path(checkpoint) / "amp_state.json").read_text(encoding="utf-8")),
            expected_step=checkpoint_state["global_step"])
    tokenizer = load_tokenizer(model_config)
    # Always audit the entire split first, even when doing a one-step smoke test.
    train_encoded, train_oversize = _encode_all(train_rows, tokenizer, settings.max_length)
    validation_encoded, validation_oversize = _encode_all(validation_rows, tokenizer, settings.max_length)
    _write_json(output / "length_audit.json", {"train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "max_length": settings.max_length, "train_oversize": train_oversize, "validation_oversize": validation_oversize,
        "train_max_tokens": max((len(item["input_ids"]) for item in train_encoded), default=0),
        "validation_max_tokens": max((len(item["input_ids"]) for item in validation_encoded), default=0)})
    if train_oversize or validation_oversize:
        raise ValueError(f"SFT length audit found {len(train_oversize)} train and {len(validation_oversize)} validation oversize rows; "
                         f"see {output / 'length_audit.json'}. No rows were dropped; explicitly revise the configuration.")
    by_id = {item["mask_metadata"]["Sample_id"]: item for item in train_encoded}
    mask_examples = [by_id[sample_id(row)]["mask_metadata"] for row in _balanced_examples(train_rows)]
    _write_json(output / "mask_audit.json", {"checked_train_rows": len(train_rows), "checked_validation_rows": len(validation_rows),
        "rule": "Only unchanged assistant answer plus <|im_end|>; prompt/empty thinking prefix/padding ignored", "examples": mask_examples})
    if smoke:
        train_encoded = [by_id[sample_id(row)] for row in _balanced_examples(train_rows, per_task=8)]
        if len(train_encoded) != 24 or len(train_encoded) % (settings.batch_size * settings.gradient_accumulation_steps):
            raise ValueError("Smoke requires 8 samples per task and complete gradient-accumulation windows")
        validation_map = {item["mask_metadata"]["Sample_id"]: item for item in validation_encoded}
        validation_encoded = [validation_map[sample_id(row)] for row in _balanced_examples(validation_rows)]

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainerCallback, TrainingArguments

    loaded = load_model(model_config, training=True)
    model = prepare_model_for_kbit_training(loaded.model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=settings.lora_rank, lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM", revision=model_config.revision))
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    before = {}
    dtype_name = loaded.metadata["compute_dtype"]
    timing_state = {"session_id": None, "started": None, "active": False}
    pending_amp = {}

    def all_finite(tensors):
        tensors = list(tensors)
        return bool(tensors) and bool(torch.stack([torch.isfinite(tensor.detach()).all() for tensor in tensors]).all().item())

    def current_scale():
        scaler = trainer.accelerator.scaler
        return float(scaler.get_scale()) if scaler is not None else None

    def record_timing(status="running"):
        if timing_state["active"]:
            _update_timing_session(output, timing_state["session_id"],
                                   time.perf_counter() - timing_state["started"], status)

    class LossLogger(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if amp.attempted_steps != state.global_step:
                raise ValueError("Restored AMP counters differ from Trainer state")
            if amp.attempted_steps and current_scale() != amp.final_scale:
                raise ValueError("Restored GradScaler scale differs from checkpoint AMP evidence")
            if not all_finite(param for _, param in trainable):
                raise FloatingPointError("Non-finite initial trainable weights")
            if smoke:
                before.update({name: param.detach().float().cpu().clone() for name, param in trainable})

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            pending_amp.clear()
            gradients = [param.grad for _, param in trainable if param.grad is not None]
            if not gradients:
                raise RuntimeError("No trainable gradients were produced")
            pending_amp.update(gradients_finite=all_finite(gradients),
                               scale_before=current_scale())
            if not pending_amp["gradients_finite"] and pending_amp["scale_before"] is None:
                raise FloatingPointError("Non-finite gradients without an AMP scaler; optimizer step prevented")

        def on_optimizer_step(self, args, state, control, **kwargs):
            # Accelerate has already executed scaler.step/update, but Trainer has
            # not incremented global_step yet. This counts attempts, including skips.
            event = amp.record(step=state.global_step + 1,
                gradients_finite=pending_amp["gradients_finite"],
                weights_finite=all_finite(param for _, param in trainable),
                optimizer_step_skipped=bool(trainer.accelerator.optimizer_step_was_skipped),
                scale_before=pending_amp["scale_before"], scale_after=current_scale())
            append_journal(output / "amp_steps.jsonl", {"utc": _utc_now(),
                "session_id": timing_state["session_id"], **event})
            _write_json(output / "amp_state.json", amp.state_dict())
            record_timing()
            amp.check_skip_bound()
            if smoke and amp.successful_updates == 1:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            record_timing()
            if logs:
                payload = {"utc": _utc_now(), "step": state.global_step, **logs}
                for name in ("loss", "eval_loss", "train_loss"):
                    if name in payload and not math.isfinite(float(payload[name])):
                        raise FloatingPointError(f"Non-finite {name} at optimizer step {state.global_step}")
                with (output / "training_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")

        def on_save(self, args, state, control, **kwargs):
            # This callback runs after Trainer has saved model, optimizer, RNG and state.
            checkpoint_dir = output / f"checkpoint-{state.global_step}"
            if amp.attempted_steps != state.global_step:
                raise ValueError("Cannot checkpoint inconsistent AMP/Trainer counters")
            _write_json(checkpoint_dir / "amp_state.json", amp.state_dict())
            _write_json(checkpoint_dir / "complete.json", {"global_step": state.global_step,
                                                        "manifest_sha256": manifest_sha256})
            record_timing()

    class FiniteLossTrainer(Trainer):
        def log(self, logs, start_time=None):
            # Trainer.log copies entries into state.log_history before callbacks.
            # Sanitize proved skips here so checkpoint and history JSON stay strict.
            return super().log(_sanitize_amp_logs(logs, amp, self.state.global_step), start_time)

        def _maybe_log_save_evaluate(self, tr_loss, grad_norm, *args, **kwargs):
            # This pinned Trainer hook runs every attempted step, including steps
            # between logging intervals. A finite-tensor gradient can still have
            # an overflowed aggregate norm; do not silently accept that update.
            if grad_norm is not None:
                _sanitize_amp_logs({"grad_norm": float(grad_norm)}, amp, self.state.global_step)
            return super()._maybe_log_save_evaluate(tr_loss, grad_norm, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                          num_items_in_batch=num_items_in_batch)
            loss = result[0] if return_outputs else result
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(f"Non-finite loss at optimizer step {self.state.global_step}")
            return result

    arguments = TrainingArguments(
        output_dir=str(output), num_train_epochs=settings.epochs, max_steps=SMOKE_MAX_ATTEMPTS if smoke else -1,
        learning_rate=settings.learning_rate, per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=1, gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=dtype_name == "float16", bf16=dtype_name == "bfloat16", tf32=False,
        optim="adamw_torch", lr_scheduler_type="linear", warmup_ratio=0.0, weight_decay=0.0,
        max_grad_norm=1.0, logging_steps=1 if smoke else settings.logging_steps,
        logging_first_step=True, logging_nan_inf_filter=False,
        save_strategy="no" if smoke else "steps", save_steps=settings.save_steps,
        save_total_limit=2, save_only_model=False, save_safetensors=True,
        eval_strategy="no" if smoke else "steps", eval_steps=settings.save_steps,
        load_best_model_at_end=False, prediction_loss_only=True,
        report_to=[], seed=model_config.seed, data_seed=model_config.seed,
        dataloader_num_workers=0, remove_unused_columns=False, label_names=["labels"],
    )
    trainer = FiniteLossTrainer(model=model, args=arguments, train_dataset=EncodedDataset(train_encoded),
        eval_dataset=EncodedDataset(validation_encoded), data_collator=AssistantOnlyCollator(tokenizer.pad_token_id),
        processing_class=tokenizer, callbacks=[LossLogger()])
    _write_json(output / "training_environment.json", loaded.metadata)
    _write_json(output / "training_arguments.json", arguments.to_dict())
    _write_json(output / "train_status.json", {"status": "running", "started_utc": _utc_now(), "resume_checkpoint": checkpoint, "smoke": smoke})
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    timing_state["session_id"] = _start_timing_session(output)
    started = time.perf_counter()
    timing_state.update(started=started, active=True)
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        _update_timing_session(output, timing_state["session_id"], elapsed, "complete")
        timing_state["active"] = False
        if not math.isfinite(float(result.training_loss)):
            raise FloatingPointError("Training finished with non-finite loss")
        amp.require_success(smoke=smoke)
        if result.global_step != amp.attempted_steps:
            raise ValueError("Final Trainer/AMP attempted-step counters differ")
        evaluation = trainer.evaluate()
        if not math.isfinite(float(evaluation["eval_loss"])):
            raise FloatingPointError("Validation loss is non-finite")
        if not all_finite(param for _, param in trainable):
            raise FloatingPointError("Non-finite adapter weights at completion")
        delta = max((float((param.detach().float().cpu() - before[name]).abs().max())
                     for name, param in trainable), default=0.0) if smoke else None
        if smoke and (delta is None or not math.isfinite(delta) or delta <= 0):
            raise RuntimeError("Smoke optimizer step did not produce a finite nonzero LoRA weight change")
        adapter_dir = output / ("smoke_adapter_do_not_evaluate" if smoke else "adapter")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        trainer.save_state()
        _write_loss_history(output, trainer.state.log_history)
        summary = {"status": "smoke_passed" if smoke else "complete", "completed_utc": _utc_now(), "smoke": smoke,
            "completed": True, "config_sha256": config_sha256,
            "train_csv_sha256": _digest(train_path), "validation_csv_sha256": _digest(validation_path),
            "code_sha256": code_sha256, "data_manifest_sha256": data_manifest_sha256,
            "dependencies": manifest["dependencies"],
            "adapter_path": str(adapter_dir), "adapter_sha256": tree_hash(adapter_dir),
            "training_seconds_this_session": elapsed,
            **_timing_summary(output),
            "resume_checkpoint": checkpoint, "global_step": result.global_step,
            "global_step_semantics": "attempted optimizer steps, including AMP-skipped attempts",
            **amp.summary(),
            "training_loss": float(result.training_loss), "validation": evaluation,
            "train_rows": len(train_encoded), "validation_rows": len(validation_encoded),
            "trainable_parameters": sum(param.numel() for _, param in trainable),
            "trainable_names": [name for name, _ in trainable], "adapter_max_weight_delta": delta,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(), "environment": loaded.metadata,
            "selection_rule": "final_epoch; validation loss recorded only", "trainer_metrics": result.metrics}
        _write_json(output / "training_summary.json", summary)
        _write_json(output / "train_status.json", {"status": summary["status"], "completed_utc": summary["completed_utc"], "smoke": smoke})
        return summary
    except Exception as error:
        record_timing("failed")
        timing_state["active"] = False
        _write_json(output / "train_status.json", {"status": "failed", "utc": _utc_now(),
            "error_type": type(error).__name__, "error": str(error), "smoke": smoke,
            **amp.summary(),
            "recovery": ("Preserve this smoke diagnostic and rerun in a fresh output directory; smoke has no resumable checkpoints"
                         if smoke else "Run the same command with --resume auto to recover the last complete checkpoint")})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-config", help="Full experiment JSON, or a ModelConfig JSON object")
    parser.add_argument("--resume", default="auto", help="auto, none, or a checkpoint directory")
    parser.add_argument("--smoke", action="store_true", help="At most 16 attempts for one real optimizer update; incomplete smoke runs require a fresh directory")
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.model_config).read_text(encoding="utf-8")) if arguments.model_config else {}
    model_config = ModelConfig(**configuration.get("model", configuration))
    training_config = TrainingConfig(**configuration.get("training", {}))
    summary = run_training(arguments.train_csv, arguments.validation_csv, arguments.output_dir,
                           model_config, training_config, arguments.resume, arguments.smoke,
                           _digest(Path(arguments.model_config)) if arguments.model_config else None)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
