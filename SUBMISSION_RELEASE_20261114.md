Current submission code snapshot, 14 November 2026

Includes GRU/Transformer, Chronos-Bolt, model summaries and isolated target-day masking. This release extends the previous public commit 033aa6d with the current comparison and masking scripts.

Historical scripts retain the original execution paths and run order. Input data are commercially restricted and are not included. Adjust W/V/D0/E and output directories for an authorized local data layout. See the existing session_aware README and input contracts in each script. a2_weather.py retains the publicly redacted grid-centre/elevation version; b3z_checks.py/b8_misc.py/b8_records.py differ from the local executed scripts only by introductory comments.

The masking script uses sess_v3.parquet, cowday_v2.parquet and cowtab.parquet. It does not overwrite original data. Outputs containing cid/day-level observations are private and are intentionally excluded from the submission archive. Only aggregate tables are supplied. Syntax checks do not imply a full rerun of the original analysis.
