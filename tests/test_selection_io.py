import json
import tempfile
import unittest
from pathlib import Path

from small_llm_qa.io_utils import append_journal, read_journal, write_json
from small_llm_qa.selection import choose_common_n, random_examples


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"Sample_id": f"TRAIN_{i}", "Task": "qa", "Answer": str(i)} for i in range(20)]
        self.pool += [{"Sample_id": "TRAIN_M", "Task": "math", "Answer": "0"}]
        self.query = {"Sample_id": "VALIDATION_1", "Task": "qa", "Answer": "SECRET"}

    def test_random_same_task_nested_and_answer_independent(self):
        three = random_examples(self.pool, self.query, 3)
        five = random_examples(self.pool, self.query, 5)
        self.assertEqual(three, five[:3])
        self.assertTrue(all(r["Task"] == "qa" for r in five))
        self.assertEqual(five, random_examples(list(reversed(self.pool)), {**self.query, "Answer": "CHANGED"}, 5))

    def test_common_n_mean_and_tie(self):
        self.assertEqual(choose_common_n({1: {"random": .5, "similar": .5},
                                         3: {"random": .4, "similar": .6}}), 1)
        with self.assertRaises(ValueError):
            choose_common_n({1: {"random": float("nan"), "similar": .5}})


class JournalTests(unittest.TestCase):
    def test_recover_partial_then_resume_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "records.jsonl"
            append_journal(p, {"Sample_id": "1", "Answer": "a"})
            with p.open("ab") as f:
                f.write(b'{"Sample_id": "2", "Ans')
            self.assertEqual(set(read_journal(p, {"1", "2"})), {"1"})
            self.assertTrue(p.with_suffix(".incomplete-tail").exists())
            append_journal(p, {"Sample_id": "2", "Answer": "b"})
            self.assertEqual(set(read_journal(p, {"1", "2"})), {"1", "2"})
            append_journal(p, {"Sample_id": "2", "Answer": "b"})
            with self.assertRaises(ValueError):
                read_journal(p, {"1", "2"})

    def test_corrupt_complete_line_must_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "records.jsonl"
            p.write_bytes(b'not-json\n')
            with self.assertRaises(ValueError):
                read_journal(p, {"1"})

    def test_immutable_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "config.json"
            write_json(p, {"n": 1}, immutable=True)
            write_json(p, {"n": 1}, immutable=True)
            with self.assertRaises(ValueError):
                write_json(p, {"n": 3}, immutable=True)


if __name__ == "__main__":
    unittest.main()
