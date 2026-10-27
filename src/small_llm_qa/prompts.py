"""Shared task prompts and exact, non-truncating chat-template budgets."""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SYSTEM_PROMPT = (
    "Answer the question concisely. For a context question, use only the supplied "
    "passages and return UNANSWERABLE if they do not support an answer. "
    "For a math question, return only the final number."
)
PROMPT_VERSION = "course-task-context-v1"


def field(row: Mapping[str, Any], name: str, default: str = "") -> str:
    value = row.get(name, row.get(name.lower(), default))
    return default if value is None else str(value)


def sample_id(row: Mapping[str, Any]) -> str:
    return field(row, "Sample_id")


def task_name(row: Mapping[str, Any]) -> str:
    task = field(row, "Task").lower().strip()
    if task not in {"qa", "rag", "math"}:
        raise ValueError(f"Unknown task {task!r} for {sample_id(row)!r}")
    return task


def user_content(row: Mapping[str, Any]) -> str:
    """Keep question/context bytes as text; never strip or truncate these fields."""
    question = field(row, "Question")
    if not question:
        raise ValueError(f"Empty question for {sample_id(row)!r}")
    content = f"Task: {task_name(row)}\nQuestion: {question}"
    context = field(row, "Context")
    if context:
        content += f"\nContext:\n{context}"
    return content


class InputBudgetExceeded(ValueError):
    def __init__(self, row_id: str, tokens: int, budget: int):
        self.sample_id = row_id
        self.tokens = tokens
        self.budget = budget
        super().__init__(
            f"Sample {row_id!r} needs {tokens} input tokens without demonstrations; "
            f"budget is {budget}. Nothing was truncated or discarded. Audit all "
            "oversize IDs and explicitly revise the shared budget before evaluation."
        )


@dataclass
class PromptResult:
    messages: list[dict[str, str]]
    text: str
    input_ids: list[int]
    input_tokens: int
    requested_shots: int
    actual_shots: int
    demo_ids: list[str]


def chat_render(tokenizer: Any, messages: Sequence[Mapping[str, str]], *, generation: bool) -> tuple[str, list[int]]:
    arguments = {"add_generation_prompt": generation, "enable_thinking": False}
    text = tokenizer.apply_chat_template(list(messages), tokenize=False, **arguments)
    ids = tokenizer.apply_chat_template(list(messages), tokenize=True, **arguments)
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        raise TypeError("Chat template must produce a flat list of token IDs")
    # Detect accidental BOS/EOS addition or a tokenizer/template mismatch early.
    separately_encoded = tokenizer.encode(text, add_special_tokens=False)
    if ids != separately_encoded:
        raise ValueError("Rendered text and chat-template tokenization disagree")
    return text, ids


def build_messages(row: Mapping[str, Any], demonstrations: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, str]]:
    task = task_name(row)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    seen = set()
    for demonstration in demonstrations:
        row_id = sample_id(demonstration)
        if not row_id or row_id == sample_id(row) or row_id in seen:
            raise ValueError("Demonstrations require unique IDs distinct from the query")
        if task_name(demonstration) != task:
            raise ValueError("Every demonstration must have the same task as its query")
        answer = field(demonstration, "Answer")
        if not answer:
            raise ValueError(f"Demonstration {row_id!r} has no answer")
        # The selection layer is responsible for membership in the training subset.
        seen.add(row_id)
        messages.extend([
            {"role": "user", "content": user_content(demonstration)},
            {"role": "assistant", "content": answer},
        ])
    messages.append({"role": "user", "content": user_content(row)})
    return messages


def build_prompt(row: Mapping[str, Any], tokenizer: Any,
                 demonstrations: Sequence[Mapping[str, Any]] = (),
                 input_budget: int = 2048) -> PromptResult:
    """Drop the lowest-priority demos until the exact prompt fits the input budget.

    ``demonstrations`` is already ordered by the selection strategy. The query's
    Answer/Accepted_answers fields are deliberately never inspected.
    """
    if input_budget <= 0:
        raise ValueError("input_budget must be positive")
    demonstrations = list(demonstrations)
    # Validate even demos that later fail to fit, so invalid pools cannot hide.
    build_messages(row, demonstrations)
    current = build_messages(row)
    current_text, current_ids = chat_render(tokenizer, current, generation=True)
    if len(current_ids) > input_budget:
        raise InputBudgetExceeded(sample_id(row), len(current_ids), input_budget)
    for count in range(len(demonstrations), -1, -1):
        messages = build_messages(row, demonstrations[:count]) if count else current
        text, ids = chat_render(tokenizer, messages, generation=True) if count else (current_text, current_ids)
        if len(ids) <= input_budget:
            return PromptResult(messages, text, ids, len(ids), len(demonstrations), count,
                                [sample_id(item) for item in demonstrations[:count]])
    raise AssertionError("The zero-demonstration prompt was already checked")


def audit_input_lengths(rows: Sequence[Mapping[str, Any]], tokenizer: Any,
                        input_budget: int = 2048) -> dict[str, Any]:
    """Read every sample without answers; report all oversize samples together."""
    lengths = []
    oversize = []
    for row in rows:
        _, ids = chat_render(tokenizer, build_messages(row), generation=True)
        lengths.append(len(ids))
        if len(ids) > input_budget:
            oversize.append({"Sample_id": sample_id(row), "Task": task_name(row), "input_tokens": len(ids)})
    return {"rows": len(rows), "input_budget": input_budget,
            "max_input_tokens": max(lengths, default=0),
            "mean_input_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
            "oversize_count": len(oversize), "oversize": oversize,
            "status": "blocked_oversize" if oversize else "passed"}
