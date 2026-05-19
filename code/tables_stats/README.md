# Stage 4: Tables and statistics regeneration

This directory contains scripts that consume Stage 3 modeling outputs and produce the manuscript's tables, lead-time analyses, and bootstrap confidence intervals.

## Scripts

| Script | Purpose |
|---|---|
| `E1_time_anchored_loso.py` | Re-extract LOSO metrics into table form (wrapper around E1 predictions parquet) |
| `E2_episode_onset_lead_recall.py` | Compute standard cow-day Lead-3-day recall and the stricter episode-onset Lead-3 recall (0.431) |
| `E3_common_label_sanity.py` | Tabulate the cohort-vs-cow-history baseline sanity analysis (7/7 vs 2/7) |
| `bootstrap_ci.py` | B=500 cow-clustered nonparametric bootstrap; produces Table 1 paired-cow CIs and Table 2 cow-clustered CIs |
| `regenerate_tables.py` | Master script: regenerates main Tables 1/2/3 + Supplementary Tables S1-S5 in docx format |

## Bootstrap design

`bootstrap_ci.py` implements two distinct bootstrap procedures:

### Table 1 (paired heat-vs-cool delta)
- Within-cow paired Δ values are resampled at the cow level (with replacement)
- 95% percentile intervals over B=500 bootstrap iterations
- Preserves within-cow paired structure

### Table 2 (LOSO performance metrics)
- For each held-out farm, the cow IDs in the test window are resampled with replacement
- The trained LightGBM model is held fixed (no retraining inside the bootstrap loop)
- PR-Lift, Recall@10%, Lift@10%, standard Lead-3 recall are recomputed on each resampled set
- 95% percentile intervals over B=500 iterations
- Captures held-out evaluation-sample uncertainty (not full training-set variability)

Episode-onset Lead-3 CIs are not produced by this script because episode boundaries interact non-trivially with cow-level resampling.

## Episode-onset Lead-3 recall definition

The standard cow-day Lead-3-day recall counts a true positive if a cow flagged on day t experiences an anomaly within days t to t+3. This metric is inflated by within-episode persistence (an ongoing multi-day episode keeps generating positive labels for the same cow).

The episode-onset Lead-3 recall, computed by `E2_episode_onset_lead_recall.py`, restricts the evaluation to the **first day of each anomaly episode**. An episode is defined as a maximal run of consecutive cow-days with anomaly = True (after smoothing). The cow is counted as "advance warned" if it appeared on a Top-10% risk list ≥1 day before the episode-onset day.

This stricter metric averages 0.431 across the 8 farms — meaningfully lower than the 0.559 standard cow-day recall, but still indicating non-trivial advance-warning value.

## Output formats

`regenerate_tables.py` produces Word `.docx` files with three-line table formatting (no vertical rules, no shading) consistent with Elsevier journal style. Output paths can be configured at the top of the script.
