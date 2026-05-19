# Derived data

This directory contains aggregated summary CSVs and bootstrap confidence intervals that serve as inputs to the figure-rendering R script and that document the manuscript's key numerical claims.

These files are safe to redistribute (no individual cow-level records).

## Subdirectories

### `tables_csv/` — Figure input CSVs

| File | Used by | Description |
|---|---|---|
| `figure2_summary.csv` | Figure 2 | Per-farm in-farm vs LOSO Lift@10% with 95% CIs |
| `E4_prospective_vs_full_comparison_wide.csv` | Figure 3 | Per-farm prospective-only vs full PR-Lift and Recall@10% |
| `figure4_panelA_topk.csv` | Supp Fig S2 | Top-K recall curves at K = 1%, 5%, 10%, 20% per farm |
| `fig5_reliability_per_farm.csv` | Supp Fig S4 | 50-bin reliability / risk-stratification per farm |
| `fig5_reliability_pooled.csv` | Supp Fig S3 | Pooled cow-day density by predicted risk percentile |
| `common_label_sanity.csv` | Supp Fig S5 | Common-label sanity raw numbers |
| `E3_common_label_baseline_regression_summary.csv` | Supp Fig S5 | E3 regression summary |

### `bootstrap_ci/` — Per-farm 95% bootstrap CIs

Per-farm CSVs with B=500 cow-clustered nonparametric bootstrap confidence intervals for the LOSO performance metrics reported in Table 2.

## License

Released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
