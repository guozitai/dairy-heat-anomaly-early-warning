# Stage 1-2: Data ingestion, QC, and feature engineering

This directory contains 11 scripts that transform raw parlor SQL exports into a quality-controlled cow-day analytical dataset and engineered features for downstream modeling.

## Execution order

```bash
1.  parse_sql_to_parquet.py        # SQL exports → parquet (per-farm)
2.  build_unified_session.py       # multi-farm union
3.  save_unified_v4.py             # unified storage (final iteration)
4.  add_dim.py                     # days-in-milk computation
5.  parse_e_calving.py             # calving event parsing
6.  build_daily_enriched.py        # daily enriched table (cow × day)
7.  merge_parity_v2.py             # parity merge
8.  qc_v2_generate_clean.py        # QC framework → 15.79M cow-day pool
9.  paired_comparison_v2.py        # paired heat vs cool comparison → Table 1, Figure 1
10. beta_lmm_v2_thi68plus.py       # β-LMM individual heat-sensitivity coefficients (James-Stein shrinkage)
11. parse_rumination_4farms.py     # SCR collar rumination ingestion (5 farms)
```

## Key outputs

| Script | Output | Used by |
|---|---|---|
| `qc_v2_generate_clean.py` | 15.79M cow-day QC-passed parquet | All Stage 3 modeling scripts |
| `paired_comparison_v2.py` | `paired_comparison_v2.csv` | Table 1; Figure 1 |
| `beta_lmm_v2_thi68plus.py` | β coefficients per cow | E1 / E3 / E4 modeling |
| `parse_rumination_4farms.py` | Rumination parquet | Stage 3 rumination sensitivity (`v41_*`) |

## Quality control framework

The QC framework is defined in `qc_v2_generate_clean.py`. It applies the following filters in sequence:

1. **Date-range filtering** — keep records within 2022-03-01 to 2025-09-30
2. **Cow-DIM eligibility** — require DIM ∈ [1, 450]
3. **Parity validity** — parity must be 1-9
4. **Yield range** — daily yield ∈ [3, 80] kg/cow/day (extreme outliers flagged separately)
5. **Milking count** — daily milkings ∈ [1, 8]
6. **Cow consistency** — cow ID present in calving records
7. **Climate data availability** — corresponding county-day ERA5 record present

QC-flagged records (those failing any filter) are retained separately for the data-infrastructure analysis (§4.1, Supp Fig S3) but excluded from primary modeling.
