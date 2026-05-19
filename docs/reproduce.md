# Reproducing the analysis

This document describes how to reproduce the figures, tables, and numerical claims of the manuscript end-to-end, given access to the raw 8-farm parlor data.

## Prerequisites

### Software

- Python 3.12 (tested on Ubuntu 24)
- R 4.4 (for figure rendering)
- See `requirements.txt` for Python deps; see comments in that file for R deps

### Data

You will need the following inputs to fully reproduce the analysis:

- **8-farm parlor SQL exports** — Daily milk-yield records per cow, 2022-03 to 2025-09. Subject to data-use agreements; see [docs/data_access.md](data_access.md).
- **Calving event records** — Cow-level calving dates per farm.
- **SCR collar exports** (5 of 8 farms) — Daily rumination minutes and activity steps.
- **ERA5 reanalysis subset** — Hourly air temperature and relative humidity for the three counties hosting the 8 study farms (Lingwu, Pingluo, Xixia in Ningxia Hui Autonomous Region), available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/).

### Disk space

Working directory: ~50 GB recommended (intermediate parquet files + LOSO prediction outputs).

## Execution

The pipeline is organized in 5 stages; each stage depends on outputs from the previous one. Run them in order.

### Stage 1 — Data ingestion & quality control

```bash
cd code/pipeline
python parse_sql_to_parquet.py          # SQL → parquet
python build_unified_session.py         # multi-farm merge
python save_unified_v4.py               # unified storage
python add_dim.py                       # DIM computation
python parse_e_calving.py               # calving event parsing
python build_daily_enriched.py          # daily enriched table
python merge_parity_v2.py               # parity merge
python qc_v2_generate_clean.py          # QC framework
```

Final output: ~15.79 million quality-controlled cow-day records.

### Stage 2 — Feature engineering

```bash
python paired_comparison_v2.py          # Table 1 + Figure 1 inputs
python beta_lmm_v2_thi68plus.py         # individual heat-sensitivity coefficients
python parse_rumination_4farms.py       # rumination data ingestion
```

### Stage 3 — Modeling (LightGBM)

```bash
cd ../modeling
python E1_time_anchored_loso_regression.py        # ~30-60 min on 8-core machine
python E3_common_label_baseline_regression.py     # ~30-60 min
python E4_prospective_feature_time_anchored_LOSO.py
python v41_sensitivity_5farms.py
python in_farm_v41_nx06_rum_window.py
```

LightGBM `random_state=42` is fixed throughout.

### Stage 4 — Tables & statistics regeneration

```bash
cd ../tables_stats
python E1_time_anchored_loso.py                   # E1 metric extraction
python E2_episode_onset_lead_recall.py            # episode-onset Lead-3 = 0.431
python E3_common_label_sanity.py                  # 7/7 vs 2/7 sanity
python bootstrap_ci.py                            # B=500 cow-clustered CIs (~30 min)
python regenerate_tables.py                       # Tables 1/2/3 + Supp S1-S5
```

### Stage 5 — Figures

```bash
cd ../figures_render
Rscript render_figures_v1_10_final.R              # all 10 figures (PDF + TIFF)
```

## Expected outputs and numerical values

Successful end-to-end execution should reproduce the following key numerical values reported in the manuscript:

| Manuscript element | Expected value |
|---|---|
| §3.3 mean PR-Lift (time-anchored LOSO) | 8.03 |
| §3.3 mean Recall@10% | 0.481 |
| §3.3 mean Lift@10% | 4.81 |
| §3.3 episode-onset Lead-3-day recall | 0.431 |
| §3.6 prospective-feature PR-Lift retention | 99.9% |
| §3.4 common-label sanity (cohort-defined) | 7/7 farms |
| §3.4 common-label sanity (cow-history-defined) | 2/7 farms |
| §4.1 co-managed farms Lift@10% (Farm 5 vs Farm 2) | 1.28 vs 3.53 |

If your reproduced numbers do not match within rounding, please check:
1. Random seed: LightGBM `random_state=42` is fixed in all training scripts
2. Time anchor: training data must be restricted to dates *strictly preceding* the held-out farm's test window
3. Anomaly threshold: 1.5σ below cohort mean for cow's lactation stage
4. QC pipeline output: 15.79M cow-day count should match before modeling begins

## Troubleshooting

For questions about reproduction, please open a GitHub issue or contact the corresponding author.
