# Session-aware milk-yield monitoring: analysis code

Code for the manuscript

> **Session-aware milk-yield monitoring for early warning in dairy herds under heat stress: unrecorded milkings, validation against veterinary records and herd-level response to heat**
>
> Zitai Guo, Shiyu Xu, Wenqi Lou, Li'e Hou, Hong Guo, Yu Zhang, Dengpan Bu

The scripts in `analysis/` are the ones that produced the results. Each begins with a description of what it computes. They are kept as they were run, with these exceptions: `a2_weather.py` requests the centre of each grid cell together with the elevation of the farm instead of the farm coordinates (the series returned are identical; see below), and the descriptions at the head of `a0_clean_milkings.py`, `a0b_event_tables.py`, `b3z_checks.py`, `b8_misc.py` and `b8_records.py` were added or reworded.

## Data

The milking and event records are held under data-use agreements with the participating farms and are not public (see [`../docs/data_access.md`](../docs/data_access.md)). The scripts read them from directories of the authors' server; the paths are set at the head of each script (`W`, `V`, `D0`, `E`, `O`).

- `unified_session.parquet` and `unified_daily_clean.parquet`: individual milking records and cow-day attributes (days in milk, parity, calving date), as written by the ingestion scripts in [`../code/pipeline/`](../code/pipeline/).
- Event tables (`e_disease`, `e_oestrus`, `e_mating`, `e_immune`, `e_trimming`, `e_group_transformation`, `e_first_inspection`, `e_calving`, `e_dried`): written by `a0b_event_tables.py` from the SQL dumps of the herd-management system.
- Weather: public. `a2_weather.py` runs without the farm records and writes the daily tables used in all analyses.

## Weather data

`a2_weather.py` obtains hourly ERA5 values and archived ECMWF IFS forecasts from Open-Meteo for the 0.25° grid cell nearest to each farm. The archive adjusts the air temperature of the cell to the elevation of the requested location. The script therefore requests the cell centre with the elevation of the farm (Supplementary Table S1 of the manuscript); for all five cells this returns the series obtained with the farm coordinates (checked over the whole period for both sources). `a2b_elevation.py` compares the adjusted series with the unadjusted values of the cell.

## Scripts and order of execution

| Step | Script | Content | Manuscript |
|---|---|---|---|
| 1 | `a0_clean_milkings.py` | Cleaning of milking records, merging of records within 60 min, intervals | Sections 2.2, 2.3 |
| 1 | `a0b_event_tables.py` | Event tables to parquet | Section 2.2 |
| 2 | `a1_build.py` | Placeholder identifiers, secretion rates, reference rates, expected yield of each milking | Section 2.3 |
| 2 | `a1c_cowday.py` | Session-aware deviation D of each cow-day; 24-h equivalent yield; deviation of the daily sum | Section 2.3 |
| 2 | `a1d_lagref.py` | Deviation with the reference window t − 21 to t − 9 | Sections 2.3, 2.7 |
| 2 | `a2_weather.py`, `a2b_elevation.py` | THI from ERA5 and archived forecasts; elevation adjustment | Section 2.5, Table S1 |
| 3 | `b0_features.py` | Features, risk sets and outcomes of the two tasks | Sections 2.4, 2.8, 2.9, Table S8 |
| 4 | `b1_audit.py`, `c0_audit_extra.py`, `c0b_missing_pattern.py`, `c10_lowyield.py`, `c11_my24_stats.py`, `b8_misc.py` | Intervals, yield after long gaps, pattern of unrecorded milkings, records removed at cleaning, dispersion of the measures | Section 3.1, Fig. 1, Tables S2a, S2b, S3a |
| 4 | `d1_farm3_intervals.py`, `d1b_farm3_fourth.py` | Modal interval of each shift by farm and month; the fourth daily session on Farm 3 | Sections 2.2, 3.1 |
| 4 | `b6_desc.py` | Farms, cow-days, episodes | Table 1 |
| 4 | `c5_cohort_label.py`, `c5b_cohort_cows.py` | Cohort-referenced label on daily sums | Section 3.2, Table S2b |
| 5 | `c2_regroup.py`, `c3_flow.py`, `b8_records.py` | Disease classes; flow of disease records | Sections 2.6, 3.3, Tables S4, S5 |
| 5 | `c4_validity.py` | Acute drops before veterinary and management records, matched controls | Section 3.3, Table 3, Fig. 2, Table S14 |
| 5 | `d2_boundary_lowyield.py` | The same comparison without the milkings after intervals of 11.5 to 14 h and with regular-interval milkings only; records of 0.5 kg or less before disease records | Sections 3.1, 3.3 |
| 6 | `b3_herd.py`, `b3y_herd_final.py`, `b3x_herd_robust.py`, `b3z_checks.py`, `b3w_completeness.py`, `c7_herd_extra.py` | Herd-level response to heat load: classes, slopes, farms, subgroups, sensitivity analyses | Section 3.4, Figs 3a–c, 5a, Tables S6, S7 |
| 6 | `c8_forecast_local.py` | Archived forecasts against the reanalysis | Section 3.4, Fig. 3d, Table S7d |
| 6 | `d3_forecast_ordered.py` | Forecasts of 2025 corrected with the bias of 2024 | Section 3.4, Table S7d |
| 7 | `c1_ew.py <part> <nparts>` | Early-warning evaluation: rules, logistic regression, gradient boosting, leave-one-farm-out in two test years, bootstrap | Sections 3.5, 3.6, Tables 4, 5, S3b, S3c, S9 to S13, S15, S16, Figs 4, 5b |
| 7 | `c6_merge.py` | Merges the parts of `c1_ew.py` | as above |
| 7 | `c9_complete_days.py <part> <nparts>` | Task 1 on cow-days with three usable milkings | Section 3.5, Table S12 |
| 8 | `../figures/fig1.py` to `fig5.py`, `figS.py` | Figures from the result tables | Figs 1 to 5, S1 to S3 |

`c1_ew.py` was run in six parts (`c1_ew.py 0 6` to `c1_ew.py 5 6`) and `c9_complete_days.py` in three.

## Environment

The analyses were run with Python 3.10.12, pandas 2.2.1, NumPy 1.26.4, LightGBM 4.6.0, scikit-learn 1.7.2, SciPy 1.12.0 and pyarrow 24.0.0. `b3w_completeness.py`, `c8_forecast_local.py`, `d3_forecast_ordered.py`, `a2b_elevation.py` and the figure scripts were run with Python 3.11.2, pandas 2.3.3, NumPy 2.4.6 and matplotlib 3.11.1; the figures are converted from SVG with `rsvg-convert`. See `requirements.txt`.

Random seeds are fixed in the scripts; those of the models, of the sampling of cow-days and of the bootstrap are derived from 20261111.
