# Terminal-Bench 2.1 — Aggregate Results

Dataset: **Terminal-Bench 2.1**, 89 tasks. Results use the original official
verifier through the external evaluation harness. The public repository includes
only this README summary, not per-task tables, raw scoring records, trajectories,
audit reports or audit scripts.

## Reported results

| System and evaluation condition | PASS | FAIL | UNSCORED | Tasks |
| --- | ---: | ---: | ---: | ---: |
| Managed D12-only, original per-task limits | 62 | 25 | 2 | 89 |
| Full Fusion, original per-task limits | 60 | 29 | 0 | 89 |
| Full Fusion, retained coverage with longer budgets and repeated attempts | 76 | 13 | 0 | 89 |
| LongCat-2.0 + Terminus-2, historical run | 45 | 42 | 2 | 89 |
| HY3 + Terminus-2, historical run | 51 | 36 | 2 | 89 |
| DeepSeek V4 Flash 0731 + Terminus-2, historical run | 61 | 27 | 1 | 89 |

## Evaluation conditions

- PASS and FAIL follow numeric official verifier results. A missing score is
  UNSCORED and is not converted into a failure.
- The original-limit Fusion comparison uses the same D12, Claude Code 2.1.220,
  Harbor 0.15.0 and managed execution/Recovery. Managed D12-only disables Memo,
  reviewers and auxiliary Compact together; it is a system ablation, not a bare
  model. Each condition starts with one attempt per task, running one task at a time.
- The 76/89 retained coverage includes later attempts with 3× or 5× Agent time
  limits. Verifier and per-request output limits are not multiplied. This is
  additional-budget coverage, not single-attempt accuracy or an equal-budget
  advantage. Full Fusion's original-limit result is 60/89, versus D12-only's 62/89.
- Historical external runs use Terminus-2 2.0.0, 10,800 seconds per Agent task,
  two tasks per model in parallel, temperature 0.7, top_p 1 and max output 8,192.
  Their exact task-source revision was not independently established. They are
  descriptive model-plus-agent comparisons, not a controlled model-only ranking.

These are the retained v0.9.4.8-era results, not results for the subsequent
v0.10.0.1 research implementation. Task selection, tool execution and scoring
remain outside the generic Fusion-MOA Gateway.
