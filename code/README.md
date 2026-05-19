# Code

This directory contains the 22 core scripts that reproduce all manuscript figures, tables, and numerical claims.

The code is organized by pipeline stage. To reproduce the analysis end-to-end, run the stages in order.

## Pipeline stages

| Stage | Directory | Purpose | # files |
|---|---|---|---|
| 1-2 | `pipeline/` | SQL ingestion → quality control → feature engineering | 11 |
| 3 | `modeling/` | LightGBM training (E1/E3/E4) + rumination sensitivity | 5 |
| 4 | `tables_stats/` | Bootstrap CI + table regeneration | 5 |
| 5 | `figures_render/` | R script rendering all 10 figures | 1 |

## Execution order

### Stage 1 — Data ingestion & quality control
```
pipeline/parse_sql_to_parquet.py        # SQL exports → parquet
pipeline/build_unified_session.py       # multi-farm merge
pipeline/save_unified_v4.py             # unified storage
pipeline/add_dim.py                     # days-in-milk computation
pipeline/parse_e_calving.py             # calving event parsing
pipeline/build_daily_enriched.py        # daily enriched table
pipeline/merge_parity_v2.py             # parity merge
pipeline/qc_v2_generate_clean.py        # QC framework → 15.79M cow-day pool
```

### Stage 2 — Feature engineering & paired analysis
```
pipeline/paired_comparison_v2.py        # within-cow paired heat vs cool comparison (Table 1, Figure 1)
pipeline/beta_lmm_v2_thi68plus.py       # β-LMM individual heat-sensitivity coefficients
pipeline/parse_rumination_4farms.py     # SCR collar rumination ingestion (5 farms)
```

### Stage 3 — Modeling
```
modeling/E1_time_anchored_loso_regression.py
    → §3.3 main LOSO model; produces mean PR-Lift 8.03 / Recall@10% 0.481

modeling/E3_common_label_baseline_regression.py
    → §3.4 / Supp Fig S5 common-label sanity (7/7 vs 2/7 framework-dependent finding)

modeling/E4_prospective_feature_time_anchored_LOSO.py
    → §3.6 prospective-feature sensitivity; 99.9% retention with prior-only features

modeling/v41_sensitivity_5farms.py + modeling/in_farm_v41_nx06_rum_window.py
    → §3.5 rumination data fusion sensitivity
```

### Stage 4 — Tables & statistics regeneration
```
tables_stats/E1_time_anchored_loso.py             # E1 metric extraction (table form)
tables_stats/E2_episode_onset_lead_recall.py      # §3.3 episode-onset Lead-3 = 0.431
tables_stats/E3_common_label_sanity.py            # E3 sanity tabulation
tables_stats/bootstrap_ci.py                      # B=500 cow-clustered bootstrap CIs for Tables 1 and 2
tables_stats/regenerate_tables.py                 # master script: Tables 1/2/3 + Supp S1-S5
```

### Stage 5 — Figures
```
figures_render/render_figures_v1_10_final.R       # all 10 figures from CSV + parquet inputs
```

R script reads inputs from `../derived_data/tables_csv/`:
- `figure2_summary.csv` → Figure 2
- `E4_prospective_vs_full_comparison_wide.csv` → Figure 3
- `figure4_panelA_topk.csv` → Supplementary Figure S2
- `fig5_reliability_per_farm.csv` + `fig5_reliability_pooled.csv` → Supplementary Figures S3 + S4
- `common_label_sanity.csv` + `E3_common_label_baseline_regression_summary.csv` → Supplementary Figure S5

Figures 1, 4, 5 require additional inputs from upstream stages (paired comparison output and LOSO predictions parquet).

## Dependencies

See `../requirements.txt` at the repository root.

## Notes on data dependencies

These scripts assume the 8-farm SQL data (parlor exports, calving records, SCR collar exports, ERA5 reanalysis subset) are present in expected paths. Update path constants at the top of each script to match your environment.

The raw SQL exports are subject to data-use agreements with the participating commercial farms and are not redistributed in this repository. See `../docs/data_access.md`.
