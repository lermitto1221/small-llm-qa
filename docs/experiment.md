# Experiment record

This document describes the completed Qwen3-1.7B experiment summarized in `results/`. It distinguishes recorded findings from what can be reproduced using the public package alone. The original coursework data and adapter weights are not distributed here, and packaging has not been followed by a fresh end-to-end GPU run.

## Research question and task definitions

The experiment tests whether selecting demonstrations or training a lightweight adapter improves a small language model under a Colab GPU budget.

| Task field | Input and expected behavior | Primary metric |
| --- | --- | --- |
| `qa` | Answer a general knowledge question concisely | Token F1 |
| `rag` | Answer using only the supplied passages; return `UNANSWERABLE` when unsupported | Token F1 |
| `math` | Solve the problem and return the final number | Numerical accuracy |

`rag` is the dataset label. Passages are supplied; the system does not retrieve external documents. The retrieval component selects labelled training demonstrations for in-context learning. Mathematical training uses the supplied final answers, rather than a separately constructed chain-of-thought dataset.

## Data and isolation

The supplied mixed dataset contains 9,067 labelled examples. With seed 42, a task-stratified split produces:

| Split | QA | Passage QA | Math | Total |
| --- | ---: | ---: | ---: | ---: |
| Training | 2,700 | 2,880 | 2,580 | 8,160 |
| Validation | 300 | 320 | 287 | 907 |
| Validation tuning subset | 60 | 60 | 60 | 180 |
| Test | 330 | 350 | 327 | 1,007 |

The tuning subset is part of validation, not an additional split. Only training examples enter the demonstration pool or SFT. Tabular and conversation-format training records were checked for identical IDs and message content.

An audit normalizes whitespace and case to detect repeated questions and repeated question-plus-context combinations. It found no exact overlap between the prepared splits under these checks. This does not rule out shared documents, paraphrases, or pretraining contamination. The split is stratified by task, not grouped by source.

Test answers were accessed only for final scoring after configuration freezing and completion of all four test prediction sets. Error analysis interprets that frozen result and is not used to change its prompts or hyperparameters.

## Shared model and prompts

- Model: `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Hardware: Tesla T4; 4-bit NF4 with double quantization and FP16 computation.
- Input budget: 2,048 tokens; maximum new tokens: 128.
- Thinking disabled; explicit `do_sample=False` and `num_beams=1`.
- One example per generation call.
- Same task prompts and target inputs across methods.

The system instruction requests concise answers, context-grounded answers or `UNANSWERABLE` for passage questions, and only the final number for math. Demonstrations are full user/assistant pairs before the target question. Token budgeting measures the tokenizer's rendered chat template and drops lower-priority demonstrations without truncating the target. An oversized target causes an explicit error.

The recorded runtime uses PyTorch 2.6.0+cu124, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, and bitsandbytes 0.45.5. Pinned model revisions and dependencies support provenance; they do not guarantee bitwise identical execution on other hardware.

## Demonstration selection

Random selection uses same-task training candidates and a deterministic per-query seed. Similarity selection embeds question text only with `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Embeddings are normalized and ranked by dot product, with sample IDs breaking ties. Encoding runs on CPU in batches of 64. This project explicitly sets the embedding maximum sequence length to 512 tokens.

Candidate requested counts are 1, 3, and 5. A predefined rule chooses the count maximizing the mean of the two methods' composite scores on the fixed 180-example tuning subset; ties within `1e-12` favor the smaller count.

| Requested count | Random score (%) | Similar score (%) | Mean (%) |
| --- | ---: | ---: | ---: |
| 1 | 15.01 | 18.21 | 16.61 |
| 3 | 18.57 | 22.15 | 20.36 |
| 5 | 17.37 | 25.48 | 21.43 |

The common requested count is therefore 5. Random selection alone peaks at 3 on this subset; the experiment does not choose a separate optimum per method.

All test QA and math inputs retain five demonstrations. Prompt budgeting reduces the count for many passage inputs. Random retains 3/4/5 demonstrations for 38/222/90 passage questions; similar retains 2/3/4/5 for 1/51/225/73 questions. Overall mean actual counts are 4.704 and 4.672. Selection changes both demonstration content and prompt length, so these effects are not isolated.

## QLoRA training

The base weights are frozen. Adapters target attention `q_proj`, `k_proj`, `v_proj`, and `o_proj`, with rank 16, alpha 32, and dropout 0.05. There are 6,422,528 trainable parameters.

Training uses one epoch, learning rate `2e-4`, AdamW, linear decay, no warm-up or weight decay, batch size 1, gradient accumulation 8, and gradient checkpointing. Evaluation and checkpointing occur every 100 attempted optimizer steps. The final-epoch adapter is evaluated; there is no automatic best-checkpoint selection.

Loss supervises only the unchanged assistant answer and its end token. System/user tokens, the empty thinking prefix, and padding are masked out. Training sequences use a 2,048-token limit without silent truncation. All recorded training and validation sequences fit that limit.

The retained trajectory has 1,020 attempted steps, 1,017 successful optimizer updates, and three AMP overflow skips. Attempted steps include skipped attempts. The final aggregate training loss is 1.228810; final validation loss is 1.120395 over 907 examples. The loss plot uses attempted steps on its horizontal axis.

An earlier interrupted session produced two successful updates before any checkpoint and was discarded. The completed session restarted from the base model. The training time ledger separates those sessions rather than treating the discarded updates as part of the retained adapter.

## Evaluation

Predictions are joined by sample ID; duplicates, missing IDs, and unexpected IDs are rejected. QA and passage QA report normalized exact match and token F1. Normalization lowercases, removes ASCII punctuation and English articles, and collapses whitespace. The best score across the primary answer and accepted alternatives is used.

Math accuracy accepts finite numbers and fractions within absolute tolerance `1e-6` and relative tolerance `1e-5`. The entire response must parse. Prose or equations count as parse failures rather than extracting a number from a potentially ambiguous response.

The primary composite score is:

```text
(mean QA token F1 + mean passage-QA token F1 + math accuracy) / 3
```

The three tasks receive equal weight despite different example counts. EM is reported separately and does not enter the composite score. ROUGE is not used. All reported percentage-point differences are computed before rounding displayed values.

## Findings

The exact aggregate scores and runtime summaries are provided in `results/metrics.json` and `results/statistics.json`; the README gives the main result table. Similarity selection improves the composite over random selection by 4.04 points; SFT improves over baseline by 27.07 points.

The comparison includes regressions. Using binary normalized EM for text tasks and numerical correctness for math, SFT fixes 284 baseline failures but loses 27 baseline successes. Similar selection fixes 76 random-selection failures but loses 41 successes. These binary counts are distinct from the F1-based composite.

Passage QA contains 290 answerable and 60 unanswerable questions:

| Method | Incorrect refusals on answerable inputs | Correct refusals on unanswerable inputs | Failures to abstain |
| --- | ---: | ---: | ---: |
| Baseline | 181 | 51 | 9 |
| Random | 95 | 32 | 28 |
| Similar | 81 | 29 | 31 |
| SFT | 21 | 30 | 30 |

SFT substantially reduces false rejection of answerable questions, but also fails to abstain more often than baseline on unsupported questions. This is a trade-off, not a uniform reliability improvement.

Math correct counts are 12, 31, 38, and 80 out of 327; parse failures are 200, 13, 5, and 0. SFT's remaining 247 answers are parseable but numerically wrong. Its improved format compliance should not be equated with reliable mathematical reasoning. General QA remains difficult: SFT token F1 is 20.63%, and EM is 14.55%.

## Runtime and memory interpretation

| Method | Generation seconds | Examples/second | Mean input tokens | Mean generated tokens | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 1111.392 | 0.906 | 202.368 | 19.028 | 1.406 |
| Random | 500.598 | 2.012 | 866.523 | 5.829 | 1.609 |
| Similar | 474.051 | 2.124 | 843.460 | 5.244 | 1.610 |
| SFT | 547.384 | 1.840 | 202.368 | 4.564 | 1.431 |

Generation times cover `model.generate` only. Recorded selection and prompt construction bring the corresponding component sums to 1114.340, 517.646, 524.421, and 550.332 seconds. Similarity index preparation adds 65.636 seconds with a cache miss, yielding 590.057 seconds of recorded components for similar. Its original build time is already included in preparation and is not counted twice.

Similarity selection has the shortest generation time but exceeds random selection once these additional components are included. Baseline produces much longer outputs, so its longer elapsed time is not an isolated measurement of intrinsic model speed. Model loading, file operations, scoring, and other unrecorded overhead are excluded.

The completed training call takes 7,511.617 seconds (125.19 minutes), including scheduled in-training evaluation and saving. Adding at least 23.199 seconds from the discarded session gives a cumulative lower bound of 7,534.817 seconds. Model loading, the separate final validation (180.202 seconds), and setup are excluded. Recorded peak training allocation is 4,302,184,448 bytes, or approximately 4.01 GiB. All memory figures measure PyTorch allocated bytes, not reserved memory or total device occupancy.

## Provenance and limitations

During development, a model-generation-default issue was corrected before final test scoring. All affected validation, count selection, and test inference were rerun using explicit greedy overrides; the completed SFT adapter was unchanged and reused. Historical sampling results are excluded from the published result table. FP16 AMP overflow handling distinguishes a skipped update from a successful update and checks that adapter weights actually change.

The original analysis verified all 134 files in its export manifest and recomputed 4,028 per-sample evaluation records. Per-sample scores agree exactly; cross-runtime aggregate rounding differences were at most `1.11e-16`. These audits establish artifact consistency, not independent proof of chronology or an independent GPU reproduction. They refer to the archived experiment, not an assertion that the public package contains every original artifact or has an unchanged source fingerprint.

Conclusions are limited to one model, seed, dataset split, and resource setting. There are no repeated-seed or significance studies. Strict answer formatting influences the scores, and normalized token overlap does not measure general semantic correctness. Exact-overlap audits cannot exclude semantic leakage. The fixed non-thinking greedy policy does not establish Qwen3's best achievable performance.

Future changes should be selected using validation data. Since these test results have already been inspected, a new unseen evaluation set is needed to support claims about later improvements. Keep the recorded historical results separate from fresh runs and report any changed prompts, dependencies, data, or decoding settings.
