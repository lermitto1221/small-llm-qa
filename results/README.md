# Recorded experiment results

These are aggregate extracts from the completed Tesla T4 experiment, not
scores from the synthetic demo or from a new execution of this public copy.
Values in JSON and CSV are fractions in [0, 1] unless a field explicitly
says otherwise. README tables convert scores to percentages.

- `metrics.json`: four methods' task metrics and unweighted composite score.
- `summary.csv`: compact version of the main result table.
- `statistics.json`: original aggregate inference statistics.
- `training_summary.json`: selected training counters and resource statistics.
- `validation_selection.json`: final greedy demonstration-count selection.
- `paired_outcomes.json`: aggregate improvements and regressions.
- `loss_history.csv`, `loss_curve.png`: unchanged saved training history/plot.
- `provenance.json`: source identifiers, original artifact hashes, and scope.

The composite score is the equal-weight average of mean QA token F1,
mean passage-QA token F1, and math numerical accuracy. Percentage-point
differences are computed from unrounded scores. Timing and GPU memory
scopes are explained in `../docs/experiment.md`.

Curated JSON files are derived summaries and do not have the same bytes as
their full original artifacts. Hashes labelled original identify archived
evidence; `release_manifest.json` at the repository root identifies this
distribution. Complete predictions, references, adapter weights, and the
original private/frozen run directory are not included. Their absence means
these aggregates alone cannot independently reproduce per-example scoring.
