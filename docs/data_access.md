# Data access policy

## Why the raw data are not in this repository

The 8-farm parlor SQL exports underpinning this study are subject to **data-use agreements** with the participating commercial dairy farms. These agreements restrict redistribution of the cow-level milking records to third parties.

For this reason, this GitHub repository contains:

✅ All code (Python and R) sufficient to reproduce the analysis given access to the data
✅ Derived summary CSVs that are not individually identifying (e.g., farm-level Lift@10% values, anonymized bootstrap CI summaries)
✅ Final figures and tables (publication-ready)

❌ Raw parlor SQL exports
❌ Cow-level prediction parquet files
❌ ERA5 county-level reanalysis subset (available separately from Copernicus)

## How to request access

Researchers who wish to reproduce the analysis or extend it with new methods may request access to the raw data through the following channels:

1. **Open an issue** on this GitHub repository describing the proposed use
2. **Contact the corresponding author** (Dengpan Bu, Budengpan@126.com) with:
   - A brief description of the proposed analysis
   - Institutional affiliation and PI
   - Confirmation that the data will be used only for the stated purpose and not redistributed
3. **Sign a data-sharing agreement** consistent with the original participating farms' data-use agreements
4. We will then coordinate transfer (typically via secure file transfer or shared institutional storage)

Standard review time: ~2 weeks.

## Available secondary data

The following items are included in this repository and are safe to redistribute:

- `derived_data/tables_csv/` — Figure-input CSVs (farm-level aggregated summaries, no individual cow records)
- `derived_data/bootstrap_ci/` — Per-farm 95% bootstrap CI summaries
- `figures/` — Publication-ready figures (PDF + TIFF)
- `tables/` — Publication-ready tables (docx)

## ERA5 climate data

The ERA5 reanalysis data used for THI computation are publicly available from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) (Hersbach et al., 2020). We extracted the following variables at 0.25° spatial resolution, hourly temporal resolution, for the three counties hosting the 8 study farms (Lingwu, Pingluo, Xixia in Ningxia Hui Autonomous Region, China):

- 2m_temperature (`t2m`)
- 2m_dewpoint_temperature (`d2m`) — for relative humidity computation

Time window: 2022-01-01 to 2025-12-31.

Users can independently retrieve the same subset; we do not redistribute the raw ERA5 NetCDF files in this repository.

## Citation if you use this code

Please cite the manuscript:

> Guo Z, Xu S, Lou W, Bu D. *Deployment-oriented early warning of heat-associated production anomalies across eight commercial dairy farms in northwestern China.* Computers and Electronics in Agriculture (submitted), 2026.
