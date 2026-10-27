# Data requirements

The recorded experiment used a mixed QA dataset assembled for coursework:
9,067 labelled training examples and 1,007 unlabelled test inputs, spanning
general QA, QA over supplied passages, and short mathematics problems.
This repository does not redistribute that dataset or its answer key.
Obtain an authorized copy from its provider to reproduce the complete run.
No public download link for the exact course compilation is asserted here.

`scripts/demo.py` instead uses six hand-written synthetic examples from
`examples/`. Its fixed predictions illustrate scoring only, not model quality.
The core test suite creates its own temporary fixtures without course data.

## Original archive structure

Provide `dataset.zip` containing the following paths at its root:

```text
train.csv
train_sft.jsonl
test.csv
test_answer.csv       # final scoring only
sources.csv
README.txt
LICENSES/
```

CSV inputs use UTF-8 and these exact column names:

- Input columns: `Sample_id`, `Task`, `Question`, `Context`.
- Labelled files also include `Answer` and `Accepted_answers` (a JSON list).
- `Task` is `qa`, `rag`, or `math`; context is supplied for `rag`.
- `sources.csv` uses `Sample_id`, `Source`, `Source_id`, `Source_urls`.
- Conversation JSONL uses exactly `Sample_id` and `messages`; system/user/
  assistant contents must match the CSV and the checked chat template.

Refer to `src/small_llm_qa/data.py` for the full schema and validation rules.
The preparer allowlists training records, public test inputs, attribution,
and license files. It never extracts, reads, or hashes `test_answer.csv`.
Only the final evaluator opens the answer key, after checking a frozen bundle
with four complete prediction sets. Keep the answer archive private.

The recorded split uses seed 42, a task-stratified approximately 10% validation
fraction, and a fixed 60-per-task tuning subset. The preparation workflow
expects enough examples for that policy; a few demonstration rows cannot
replace the full dataset. Other datasets require an explicitly revised and
separately documented protocol, rather than pretending to reproduce these
historical results.

## Attribution

Source datasets include Natural Questions, TriviaQA, SQuAD 2.0, HotpotQA,
GSM8K, and SVAMP, together with course-authored material. The original
attribution text and source license notices are preserved under
[data_licenses/](data_licenses/). These notices describe external source data;
they do not license the original project code or imply that course data is
included. Public-source datasets alone do not reconstruct the exact course
selection, passages, authored material, or split.

Reproduction should retain the original `sources.csv` and `LICENSES/` files.
Prepared records, generated per-example journals, answers, and checkpoints
belong in ignored local/Drive directories. Review notebook outputs before
sharing a newly executed run.
