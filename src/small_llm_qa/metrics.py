"""Deterministic, dependency-free metrics for the three course tasks.

Scores are fractions in [0, 1]. English text uses the SQuAD normalization:
lowercase, remove ASCII punctuation, remove a/an/the, collapse whitespace.
Numeric answers must consist only of a number or a fraction of numbers; we
never extract a convenient number from prose and never execute model output.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterable, Mapping

ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5
TASKS = ("qa", "rag", "math")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_REMOVE_PUNCTUATION = str.maketrans("", "", string.punctuation)
_UNSIGNED = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
_NUMBER = rf"[+-]?{_UNSIGNED}(?:[eE][+-]?\d+)?"
_NUMERIC = re.compile(rf"(?P<numerator>{_NUMBER})(?:\s*/\s*(?P<denominator>{_NUMBER}))?", re.ASCII)


def normalize_answer(text: str) -> str:
    """Normalize English answers exactly as documented above."""
    if not isinstance(text, str):
        raise ValueError("Answers must be strings.")
    without_punctuation = text.lower().translate(_REMOVE_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", without_punctuation).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class NumericAnswer:
    value: float | None
    canonical: str | None
    error: str | None


def parse_numeric_answer(text: str) -> NumericAnswer:
    """Parse a full final answer, allowing decimals, scientific notation, fractions.

    Correctly grouped thousands separators are allowed. Units, prose, code,
    percentages, equations, markdown and non-finite numbers are rejected. A
    bounded input and exponent prevent pathological arbitrary-precision work.
    """
    if not isinstance(text, str):
        return NumericAnswer(None, None, "not_a_string")
    value_text = text.strip()
    if not value_text:
        return NumericAnswer(None, None, "empty")
    if len(value_text) > 512:
        return NumericAnswer(None, None, "too_long")
    match = _NUMERIC.fullmatch(value_text)
    if match is None:
        return NumericAnswer(None, None, "invalid_numeric_format")
    parts = [match.group("numerator"), match.group("denominator") or "1"]
    try:
        decimals = []
        for part in parts:
            decimal = Decimal(part.replace(",", ""))
            if not decimal.is_finite() or abs(decimal.as_tuple().exponent) > 308:
                return NumericAnswer(None, None, "numeric_range_exceeded")
            decimals.append(decimal)
        if decimals[1] == 0:
            return NumericAnswer(None, None, "zero_denominator")
        rational = Fraction(decimals[0]) / Fraction(decimals[1])
        value = float(rational)
        if not math.isfinite(value):
            return NumericAnswer(None, None, "non_finite")
        return NumericAnswer(value, str(rational), None)
    except (InvalidOperation, ValueError, OverflowError):
        return NumericAnswer(None, None, "numeric_range_exceeded")


def accepted_answers(record: Mapping[str, Any]) -> list[str]:
    """Include the primary answer and all explicitly listed answer aliases."""
    answer = record.get("Answer")
    if not isinstance(answer, str):
        raise ValueError(f"Reference {record.get('Sample_id')!r} has no string Answer.")
    aliases = record.get("Accepted_answers", "")
    if aliases in (None, ""):
        aliases = []
    elif isinstance(aliases, str):
        try:
            aliases = json.loads(aliases)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Accepted_answers JSON for {record.get('Sample_id')!r}.") from error
    if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
        raise ValueError(f"Accepted_answers must be a JSON list of strings for {record.get('Sample_id')!r}.")
    return list(dict.fromkeys([answer, *aliases]))


def index_records(records: Iterable[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = record.get("Sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id != sample_id.strip():
            raise ValueError(f"{label}: each Sample_id must be a nonempty string without surrounding whitespace.")
        if sample_id in indexed:
            raise ValueError(f"{label}: duplicate Sample_id {sample_id!r}.")
        indexed[sample_id] = record
    if not indexed:
        raise ValueError(f"{label}: no records supplied.")
    return indexed


def require_same_ids(expected: Iterable[str], actual: Iterable[str], label: str) -> None:
    expected_set, actual_set = set(expected), set(actual)
    missing, extra = sorted(expected_set - actual_set), sorted(actual_set - expected_set)
    if missing or extra:
        raise ValueError(
            f"{label}: Sample_id mismatch: missing={len(missing)} {missing[:5]!r}; "
            f"extra={len(extra)} {extra[:5]!r}."
        )


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_records(
    references: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score by exact ID, retaining raw predictions and every numeric parse result.

    An absent task has null aggregate metrics; macro_score is null unless all
    three tasks are present. UNANSWERABLE precision/recall use zero when their
    denominators are zero, and also expose confusion counts for interpretation.
    """
    gold = index_records(references, "references")
    predicted = index_records(predictions, "predictions")
    require_same_ids(gold, predicted, "predictions")
    per_sample = []
    task_results: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for sample_id, reference in gold.items():
        task = reference.get("Task")
        if task not in TASKS:
            raise ValueError(f"Unsupported Task {task!r} for {sample_id!r}.")
        prediction = predicted[sample_id].get("Answer")
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction {sample_id!r} has no string Answer.")
        answers = accepted_answers(reference)
        sample: dict[str, Any] = {
            "Sample_id": sample_id, "Task": task, "prediction": prediction,
            "reference_answers": answers,
        }
        if task in ("qa", "rag"):
            sample.update(
                exact_match=max(exact_match(prediction, answer) for answer in answers),
                token_f1=max(token_f1(prediction, answer) for answer in answers),
                normalized_prediction=normalize_answer(prediction),
            )
            if task == "rag":
                is_unanswerable = [normalize_answer(answer) == "unanswerable" for answer in answers]
                if any(is_unanswerable) and not all(is_unanswerable):
                    raise ValueError(f"Reference {sample_id!r} mixes UNANSWERABLE and answerable aliases.")
                sample.update(
                    reference_unanswerable=any(is_unanswerable),
                    predicted_unanswerable=normalize_answer(prediction) == "unanswerable",
                )
        else:
            parsed = parse_numeric_answer(prediction)
            parsed_answers = [parse_numeric_answer(answer) for answer in answers]
            if any(answer.error is not None for answer in parsed_answers):
                raise ValueError(f"Math reference {sample_id!r} contains an invalid numeric answer.")
            correct = parsed.error is None and any(
                math.isclose(parsed.value, answer.value, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
                for answer in parsed_answers
            )
            sample.update(
                numeric_accuracy=float(correct), parsed_prediction=asdict(parsed),
                parsed_references=[asdict(answer) for answer in parsed_answers],
            )
        task_results[task].append(sample)
        per_sample.append(sample)

    metrics: dict[str, Any] = {}
    for task in ("qa", "rag"):
        samples = task_results[task]
        metrics[task] = {
            "count": len(samples),
            "exact_match": _mean([sample["exact_match"] for sample in samples]),
            "token_f1": _mean([sample["token_f1"] for sample in samples]),
        }
    rag = task_results["rag"]
    true_positive = sum(sample["reference_unanswerable"] and sample["predicted_unanswerable"] for sample in rag)
    false_positive = sum(not sample["reference_unanswerable"] and sample["predicted_unanswerable"] for sample in rag)
    false_negative = sum(sample["reference_unanswerable"] and not sample["predicted_unanswerable"] for sample in rag)
    true_negative = len(rag) - true_positive - false_positive - false_negative
    metrics["rag"].update(
        unanswerable_precision=true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0,
        unanswerable_recall=true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        unanswerable_true_positive=true_positive,
        unanswerable_false_positive=false_positive,
        unanswerable_false_negative=false_negative,
        unanswerable_true_negative=true_negative,
        answerable_count=true_negative + false_positive,
        unanswerable_count=true_positive + false_negative,
        false_rejection_count=false_positive,
        false_rejection_rate=false_positive / (true_negative + false_positive) if true_negative + false_positive else 0.0,
        false_acceptance_count=false_negative,
    )
    math_samples = task_results["math"]
    metrics["math"] = {
        "count": len(math_samples),
        "numeric_accuracy": _mean([sample["numeric_accuracy"] for sample in math_samples]),
        "parse_failure_count": sum(sample["parsed_prediction"]["error"] is not None for sample in math_samples),
        "absolute_tolerance": ABS_TOLERANCE, "relative_tolerance": REL_TOLERANCE,
    }
    primary = [metrics["qa"]["token_f1"], metrics["rag"]["token_f1"], metrics["math"]["numeric_accuracy"]]
    metrics["macro_score"] = sum(primary) / 3 if all(value is not None for value in primary) else None
    metrics["total_count"] = len(gold)
    return {"metrics": metrics, "samples": per_sample}
