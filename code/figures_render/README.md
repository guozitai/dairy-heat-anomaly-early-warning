# Stage 5: Figure rendering

This directory contains a single R script that renders all 10 figures (5 main + 5 supplementary) for the manuscript.

## Script

`render_figures_v1_10_final.R`

## Inputs

The script reads from two locations:

### 1. `derived_data/tables_csv/` (small summary CSVs)
- `figure2_summary.csv` → Figure 2 (LOSO dumbbell)
- `E4_prospective_vs_full_comparison_wide.csv` → Figure 3 (prospective vs full scatter)
- `figure4_panelA_topk.csv` → Supplementary Figure S2 (Top-K curves)
- `fig5_reliability_per_farm.csv` → Supplementary Figure S4 (50-bin reliability)
- `fig5_reliability_pooled.csv` → Supplementary Figure S3 (pooled density)
- `common_label_sanity.csv` + `E3_common_label_baseline_regression_summary.csv` → Supplementary Figure S5 (common-label sanity 2-panel)

### 2. Direct inputs (paired comparison CSV + LOSO predictions parquet)
- `paired_comparison_v2.csv` → Figure 1 (paired Δ vs prevalence)
- LOSO predictions parquet → Figure 4 (risk-decile enrichment) + Figure 5 (8-panel farm-specific scatter) + Supplementary Figure S1 (decomposition)

## Critical: FARM_MAP

Figure 5 uses an 8-panel facet with farm labels "Farm 1" through "Farm 8". The mapping from underlying farm IDs (internal codes, redacted) to anonymized Farm labels is hardcoded in `render_figures_v1_10_final.R`:

```r
FARM_MAP <- c(
  "A" = "Farm 8",   # smallest farm
  "B" = "Farm 7",
  "C" = "Farm 1",   # reference farm (longest analytical window)
  "D" = "Farm 4",
  "E" = "Farm 5",   # acquisition failure case (Section 4.1)
  "F" = "Farm 6",
  "G" = "Farm 3",
  "H" = "Farm 2"    # co-managed with Farm 5 (Section 4.1)
)
```

This mapping is consistent across all figures and tables.

## Outputs

Each figure is produced in two formats:

- **PDF** — vector, for publication
- **TIFF** — raster, 300 DPI, for journals requiring raster

Output directory: `figures/` (relative to repository root).

## R dependencies

```r
install.packages(c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "readr",
  "scales",
  "patchwork",
  "arrow"           # parquet I/O for Figures 4, 5
))
```

Tested with R 4.4 on Ubuntu 24.

## Usage

```bash
cd code/figures_render
Rscript render_figures_v1_10_final.R
```

Or interactively in R / RStudio.
