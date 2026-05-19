# Stage 3: Modeling

This directory contains the 5 modeling scripts that produce the manuscript's primary results.

All scripts use **LightGBM regression** (lightgbm 3.3, `random_state=42`).

## Scripts

| Script | Manuscript section | Output |
|---|---|---|
| `E1_time_anchored_loso_regression.py` | §3.3 | LOSO predictions parquet (8 held-out farms); mean PR-Lift 8.03, mean Recall@10% 0.481 |
| `E3_common_label_baseline_regression.py` | §3.4, Supp Fig S5 | Cohort vs cow-history baseline comparison; 7/7 vs 2/7 framework-dependent finding |
| `E4_prospective_feature_time_anchored_LOSO.py` | §3.6 | Prospective-only feature variant; 99.9% PR-Lift retention |
| `v41_sensitivity_5farms.py` | §3.5, Table 3 | Rumination data fusion gain (5 farms with collar coverage) |
| `in_farm_v41_nx06_rum_window.py` | §3.5 | Window-restricted rumination analysis on the reference farm |

## Time-anchored LOSO design (critical)

The "time-anchored" component is essential to the manuscript's deployment-realism argument. For each held-out farm:

1. Identify the held-out farm's test window (typically the most recent 6-12 months of records)
2. Restrict all *other* farms' training data to **dates strictly preceding** the held-out farm's test window start
3. Train a LightGBM model on the restricted other-farm data
4. Evaluate on the held-out farm's test window

This avoids temporal leakage (using future data from other farms to predict past held-out farm events). Standard cross-validation without this anchor inflates performance estimates by an unknown amount.

## Feature variants

| Variant | Description | Result |
|---|---|---|
| Full | All 7 climate features (focal-day + lag-1 + lag-7 cumulative) + cow features + cohort features | Reported in Tables 2 + Fig 4 |
| Prospective-only (E4) | Focal-day climate variables removed; lag features re-derived using only prior days | 99.9% retention of Full PR-Lift |
| Cohort-baseline + cow-history-baseline (E3) | Two anomaly framework definitions | Framework-dependent superiority |

## Compute requirements

- Memory: 32 GB recommended (training set after time-anchor cuts is ~10-15M cow-days)
- Time: ~30-60 min for E1 per held-out farm on 8-core machine; ~5-8 hours for full E1 sweep
- Disk: predictions parquet ~500 MB per held-out farm
