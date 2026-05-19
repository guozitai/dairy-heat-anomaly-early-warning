# Manuscript-to-code map

This document maps every figure, table, and key numerical claim in the manuscript to the script that produces it.

## Main figures

| Figure | Description | Producing script(s) | Source data |
|---|---|---|---|
| **Figure 1** | Paired Δ vs prevalence (8 farms) | `code/figures_render/render_figures_v1_10_final.R` | `pipeline/paired_comparison_v2.py` → `paired_comparison_v2.csv` |
| **Figure 2** | In-farm vs LOSO Lift@10% dumbbell + CI | `code/figures_render/render_figures_v1_10_final.R` | `derived_data/tables_csv/figure2_summary.csv` |
| **Figure 3** | Prospective-only vs full-feature scatter (PR-Lift, Recall@10%) | `code/figures_render/render_figures_v1_10_final.R` | `derived_data/tables_csv/E4_prospective_vs_full_comparison_wide.csv` |
| **Figure 4** | Risk-decile enrichment curve (Top-K%) | `code/figures_render/render_figures_v1_10_final.R` | LOSO predictions parquet (Stage 3 output) |
| **Figure 5** | Farm-specific 8-panel risk-ranking scatter | `code/figures_render/render_figures_v1_10_final.R` | LOSO predictions parquet (Stage 3 output) |

## Main tables

| Table | Description | Producing script(s) |
|---|---|---|
| **Table 1** | Within-cow paired heat-stress response (8 farms) | `pipeline/paired_comparison_v2.py` + `tables_stats/bootstrap_ci.py` |
| **Table 2** | LOSO performance with 95% bootstrap CIs (8 farms) | `modeling/E1_time_anchored_loso_regression.py` + `tables_stats/bootstrap_ci.py` |
| **Table 3** | Rumination data fusion gain (5 farms) | `modeling/v41_sensitivity_5farms.py` + `tables_stats/regenerate_tables.py` |

## Supplementary figures

| Figure | Description | Producing script(s) | Source data |
|---|---|---|---|
| **Supp Fig S1** | Three-way decomposition (Lift@10%) | `render_figures_v1_10_final.R` | LOSO predictions parquet |
| **Supp Fig S2** | Top-K recall curves (1/5/10/20%) | `render_figures_v1_10_final.R` | `derived_data/tables_csv/figure4_panelA_topk.csv` |
| **Supp Fig S3** | Pooled cow-day density heatmap | `render_figures_v1_10_final.R` | `derived_data/tables_csv/fig5_reliability_pooled.csv` |
| **Supp Fig S4** | 50-bin reliability / risk-stratification curve | `render_figures_v1_10_final.R` | `derived_data/tables_csv/fig5_reliability_per_farm.csv` |
| **Supp Fig S5** | Common-label sanity (2-panel paired dot) | `render_figures_v1_10_final.R` | `derived_data/tables_csv/common_label_sanity.csv` |

## Supplementary tables

All supplementary tables (S1-S5) are produced by `tables_stats/regenerate_tables.py`.

| Table | Description |
|---|---|
| **Supp Table S1** | Farm characteristics (n_cows, time window, climate, parity composition) |
| **Supp Table S2** | Data flow (15.79M cow-day pipeline counts) |
| **Supp Table S3** | Three-way decomposition (cohort baseline vs cross-farm transfer steps) |
| **Supp Table S4** | Common-label sanity analysis (E3) |
| **Supp Table S5** | Prospective-feature sensitivity (E4) |

## Key numerical values

| Manuscript section | Value | Source |
|---|---|---|
| §3.3 | Mean PR-Lift = 8.03 | `modeling/E1_time_anchored_loso_regression.py` |
| §3.3 | Mean Recall@10% = 0.481 | `modeling/E1_time_anchored_loso_regression.py` |
| §3.3 | Mean Lift@10% = 4.81 | `modeling/E1_time_anchored_loso_regression.py` |
| §3.3 | Episode-onset Lead-3 recall = 0.431 | `tables_stats/E2_episode_onset_lead_recall.py` |
| §3.6 | Prospective PR-Lift retention = 99.9% | `modeling/E4_prospective_feature_time_anchored_LOSO.py` |
| §3.4 | Cohort-defined: 7/7 farms cohort > cow | `tables_stats/E3_common_label_sanity.py` |
| §3.4 | Cow-history-defined: 2/7 farms cohort > cow | `tables_stats/E3_common_label_sanity.py` |
| §4.1 | Co-managed farms Lift@10% 1.28 vs 3.53 | `modeling/E1_time_anchored_loso_regression.py` |
| Table 1 | Paired Δ range across 8 farms | `pipeline/paired_comparison_v2.py` |
| Table 1 | 95% CIs (paired-cow bootstrap) | `tables_stats/bootstrap_ci.py` |
| Table 2 | 95% CIs (cow-clustered bootstrap, B=500) | `tables_stats/bootstrap_ci.py` |
