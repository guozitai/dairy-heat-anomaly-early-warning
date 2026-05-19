# Deployment-oriented early warning of heat-associated production anomalies in commercial dairy farms

This repository contains the code, figures, and tables for the manuscript:

> **Deployment-oriented early warning of heat-associated production anomalies across eight commercial dairy farms**
>
> Zitai Guo¹, Shiyu Xu¹, Wenqi Lou¹, Li'e Hou², Hong Guo², Yu Zhang², Dengpan Bu¹\*
>
> ¹ State Key Laboratory of Animal Nutrition and Feeding, Institute of Animal Sciences, Chinese Academy of Agricultural Sciences, Beijing 100193, China
> ² Ningxia Hui Autonomous Region Animal Husbandry Workstation, Yinchuan 750001, China
>
> \* Corresponding author. E-mail: Budengpan@126.com
>
> Submitted to *Computers and Electronics in Agriculture*, 2026.

## TL;DR

We evaluate heat-associated production-anomaly early warning across 8 commercial dairy farms in Ningxia, China (15.79 million quality-controlled cow-day records, March 2022 – September 2025).

Under **time-anchored leave-one-site-out (LOSO) validation**, LightGBM models achieve:

| Metric | Mean across 8 farms |
|--------|--------------------:|
| Lift@10% (risk-ranking enrichment) | 4.81× baseline |
| Recall@10% (Top-10% capture rate) | 0.481 |
| PR-Lift (prevalence-normalized AUPRC) | 8.03 |
| Episode-onset Lead-3-day recall | 0.431 |

A **strictly prospective-feature variant** — using only prior-day re-derived climate features — retains 99.9% of full-model performance, confirming the cross-farm risk signal does not depend on focal-day reanalysis variables.

A **common-label sanity analysis** shows the cohort-baseline feature advantage operates within a cohort-defined anomaly framework (winning 7/7 farms) rather than as universal feature superiority (winning 2/7 under cow-history-defined labels).

A **co-managed-farm comparison** and **rumination-sensor sensitivity analysis** identify data acquisition stability as a central deployment constraint — not addressable by model choice alone.

## Repository structure

```
.
├── code/                    22 core scripts reproducing all analyses
│   ├── pipeline/            Stage 1-2: SQL ingestion → QC → feature engineering (11 files)
│   ├── modeling/            Stage 3: LightGBM training (E1/E3/E4 + rumination sensitivity) (5 files)
│   ├── tables_stats/        Stage 4: bootstrap CIs + table regeneration (5 files)
│   └── figures_render/      Stage 5: R script rendering all 10 figures (1 file)
├── figures/                 10 final figures (5 main + 5 supplementary, PDF + TIFF)
├── tables/                  Main tables 1-3 + supplementary tables S1-S5 (docx)
├── derived_data/
│   ├── tables_csv/          Figure input CSVs (used by R script)
│   └── bootstrap_ci/        Per-farm 95% bootstrap confidence intervals
├── docs/
│   ├── reproduce.md         How to reproduce the analysis end-to-end
│   ├── manuscript_to_code_map.md   Which .py / .R produces which manuscript element
│   └── data_access.md       How to request access to the 8-farm SQL data
├── requirements.txt         Python dependencies
├── LICENSE                  MIT
├── CITATION.cff             How to cite this work
└── README.md                This file
```

## Quick links

- 🧮 [How to reproduce](docs/reproduce.md)
- 🗺️ [Manuscript-to-code map](docs/manuscript_to_code_map.md)
- 🔒 [Data access policy](docs/data_access.md)
- 📊 [Final figures](figures/) (PDF + TIFF)

## Method summary

We formulate heat-stress early warning as an **anomaly detection problem**: for each cow-day, predict whether the daily milk yield falls substantially below the cohort-expected baseline given lactation stage and recent history. Anomalies are defined as cow-day milk-yield deviations exceeding 1.5σ below the cohort mean for the cow's lactation stage.

### Validation paradigms

1. **In-farm temporal validation** — train on each farm's earlier records, test on a held-out future window of the same farm.
2. **Time-anchored leave-one-site-out (LOSO)** — train on all other farms (restricted to dates *preceding* the held-out farm's test window, avoiding temporal leakage), evaluate on the held-out farm's later records.

### Deployment-relevant evaluation metrics

- **Lift@10%** — capture rate of true anomalies among the daily Top-10% predicted-risk cow-days, normalized by anomaly prevalence
- **Recall@10%** — fraction of all anomalies caught by daily Top-10% screening
- **Episode-onset Lead-3-day recall** — fraction of new anomaly episodes for which the affected cow was on the Top-10% risk list ≥1 day before episode onset (excludes within-episode persistence inflation)
- **PR-Lift** — Precision-Recall AUC normalized by anomaly prevalence

### Prospective-feature defense

To demonstrate deployment feasibility — i.e., that the model does not depend on focal-day climate variables that may be unavailable at decision time — we evaluate a strictly prior-only feature variant. This variant retains 99.9% of full-model PR-Lift across all 8 farms.

## Reproducibility

The code in this repository reproduces all figures, tables, and numerical claims in the manuscript when run against the 8-farm parlor SQL data. See [`docs/reproduce.md`](docs/reproduce.md) for the full execution pipeline.

The 8-farm SQL data are subject to data-use agreements with the participating commercial farms and are not redistributed here. See [`docs/data_access.md`](docs/data_access.md) for access requests.

## Citation

If you use any code or material from this repository, please cite:

```bibtex
@article{guo2026heat,
  title   = {Deployment-oriented early warning of heat-associated production anomalies
             across eight commercial dairy farms},
  author  = {Guo, Zitai and Xu, Shiyu and Lou, Wenqi and Hou, Li'e and
             Guo, Hong and Zhang, Yu and Bu, Dengpan},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026},
  note    = {Submitted}
}
```

See also [`CITATION.cff`](CITATION.cff) for GitHub-native citation metadata.

## License

- Code is released under the [MIT License](LICENSE).
- Figures and tables are released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

- Corresponding author: **Dengpan Bu** (Budengpan@126.com)
- First author / repository maintainer: **Zitai Guo** (State Key Laboratory of Animal Nutrition and Feeding, CAAS)

## Acknowledgements

We thank the staff at the eight participating commercial dairy farms for sustained data infrastructure cooperation throughout the study period.
