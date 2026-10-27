"""Meaningful prompt-budget tests without downloading a model or importing torch."""

import unittest

from small_llm_qa.prompts import InputBudgetExceeded, audit_input_lengths, build_prompt, user_content


class FakeTokenizer:
    """Tiny reversible tokenizer that models Qwen's non-thinking chat boundary."""
    eos_token = "<|im_end|>"
    eos_token_id = 151645
    pad_token_id = 151643
    special = {"<|im_start|>": 151644, "<|im_end|>": 151645}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        ids = []
        while text:
            for token, token_id in self.special.items():
                if text.startswith(token):
                    ids.append(token_id)
                    text = text[len(token):]
                    break
            else:
                ids.append(ord(text[0]) + 1)
                text = text[1:]
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        reverse = {value: key for key, value in self.special.items()}
        return "".join(("" if skip_special_tokens else reverse[token]) if token in reverse else chr(token - 1)
                       for token in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        result = ""
        for position, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role == "assistant" and position == len(messages) - 1:
                content = "<think>\n\n</think>\n\n" + content.lstrip("\n")
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        if add_generation_prompt:
            result += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return self.encode(result, add_special_tokens=False) if tokenize else result


def row(identifier="query", task="rag", question="Which city?", context="[1] City: Paris.", answer="Paris"):
    return {"Sample_id": identifier, "Task": task, "Question": question, "Context": context, "Answer": answer}


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_query_answer_is_never_read(self):
        first = build_prompt(row(answer="secret one"), self.tokenizer)
        second = build_prompt(row(answer="secret two"), self.tokenizer)
        self.assertEqual(first.input_ids, second.input_ids)
        self.assertNotIn("secret", first.text)

    def test_current_question_and_context_preserved_verbatim(self):
        example = row(question="  Which\ncity? ", context=" [1] First.\n\n[2] Second. ")
        prompt = build_prompt(example, self.tokenizer)
        self.assertIn(example["Question"], prompt.messages[-1]["content"])
        self.assertTrue(prompt.messages[-1]["content"].endswith(example["Context"]))
        self.assertEqual(prompt.input_tokens, len(self.tokenizer.encode(prompt.text)))

    def test_budget_reduces_only_lowest_priority_demonstrations(self):
        query = row()
        demos = [row("d1", question="A?"), row("d2", question="B?")]
        one = build_prompt(query, self.tokenizer, demos[:1], 10000)
        result = build_prompt(query, self.tokenizer, demos, one.input_tokens)
        self.assertEqual(result.actual_shots, 1)
        self.assertEqual(result.requested_shots, 2)
        self.assertEqual(result.demo_ids, ["d1"])
        self.assertEqual(result.messages[-1]["content"], user_content(query))
        self.assertEqual(result.messages[1]["content"], user_content(demos[0]))
        self.assertEqual(result.messages[2]["content"], demos[0]["Answer"])

    def test_oversize_query_raises_and_audit_reports_every_id(self):
        current = build_prompt(row(), self.tokenizer)
        with self.assertRaises(InputBudgetExceeded) as captured:
            build_prompt(row(), self.tokenizer, input_budget=current.input_tokens - 1)
        self.assertEqual(captured.exception.tokens, current.input_tokens)
        audit = audit_input_lengths([row("a"), row("b")], self.tokenizer, current.input_tokens - 1)
        self.assertEqual(audit["oversize_count"], 2)
        self.assertEqual([item["Sample_id"] for item in audit["oversize"]], ["a", "b"])

    def test_demo_task_id_and_answer_validation(self):
        for demos in ([row("d", task="qa")], [row()], [row("d"), row("d")], [row("d", answer="")]):
            with self.subTest(demos=demos), self.assertRaises(ValueError):
                build_prompt(row(), self.tokenizer, demos)


if __name__ == "__main__":
    unittest.main()
