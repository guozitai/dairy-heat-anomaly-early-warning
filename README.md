# Session-aware milk-yield monitoring for early warning in dairy herds under heat stress

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code accompanying the manuscript

> **Session-aware milk-yield monitoring for early warning in dairy herds under heat stress: unrecorded milkings, validation against veterinary records and herd-level response to heat**
>
> Zitai Guo¹, Shiyu Xu¹, Wenqi Lou¹, Li'e Hou², Hong Guo², Yu Zhang², Dengpan Bu¹\*
>
> ¹ State Key Laboratory of Animal Nutrition and Feeding, Institute of Animal Science, Chinese Academy of Agricultural Sciences, Beijing 100193, China
> ² Ningxia Hui Autonomous Region Animal Husbandry Workstation, Yinchuan 750001, China
>
> \* Corresponding author. E-mail: budengpan@126.com

## Summary

Routine milking records of eight commercial dairy farms (41.1 million milkings) were analysed together with veterinary records and gridded weather data. Fewer than three daily milkings were recorded on 45% of cow-days although the cows had been milked, so that the sum of the recorded milkings placed 21.8% of cow-days more than 20% below the cow's own level, against 5.3% for a deviation computed from each milking and the interval since the previous recorded milking (the session-aware deviation).

The code computes this deviation, tests it against veterinary and management records, estimates the response of herd yield to heat load, and evaluates daily lists of cows at risk, formed by simple rules and by models, on farms and in years not used for fitting.

## Contents

| Path | Content |
|---|---|
| [`session_aware/`](session_aware/) | Analysis and figure scripts of the manuscript; its README lists which script produces which result |
| [`code/pipeline/`](code/pipeline/) | Ingestion of the parlour SQL exports into the tables read by `session_aware/analysis/a0_clean_milkings.py` |
| [`docs/data_access.md`](docs/data_access.md) | Access to the farm records, which are not public |

## Data

The milking and event records are subject to data-use agreements with the participating farms and are not redistributed. The weather data are public (ERA5 and archived ECMWF IFS forecasts through Open-Meteo); `session_aware/analysis/a2_weather.py` retrieves them.

## Earlier version

The directories `code/modeling/`, `code/tables_stats/`, `code/figures_render/`, `figures/`, `tables/` and `derived_data/`, and the remaining files in `docs/`, belong to an earlier version of the study (release v1.0, [DOI 10.5281/zenodo.20281308](https://doi.org/10.5281/zenodo.20281308)), in which daily milk yield was the sum of the milkings recorded on a day. They are superseded by `session_aware/` and are kept for reference. The README of that version is [`docs/README_v1.md`](docs/README_v1.md).

## License

Code is released under the [MIT License](LICENSE).

## Contact

- Corresponding author: Dengpan Bu (budengpan@126.com)
- Repository maintainer: Zitai Guo
