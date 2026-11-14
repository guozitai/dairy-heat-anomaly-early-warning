# Additional coverage, timestamp and repeated-deletion checks

These analyses extend the session-aware measurement evaluation. Existing farm-year model predictions are reused; no models are retrained.

1. `session_aware/analysis/g1_common_support.py`: reproduce both Task 2 rule estimates on the original risk sets, then rebuild daily 5% lists and case eligibility on cow-days where both D and the 24-h equivalent deviation are available. Results distinguish full-procedure coverage from common-support ranking.
2. `g2_record_time.py`: audit exported event/creation timestamps and rebuild D-rule eligibility using disease records available by calendar-day end. This does not reconstruct historical edits, symptom recognition or availability of all model features.
3. `g3_serial_masking.py`: same 4,180 target cow-days as the isolated masking experiment; delete records over 3/9-day blocks and recompute intervals and both historical references. Score availability and original-positive retention are separate outcomes. The script writes private paired data only to the authorized data-server output directory; those files are not distributed.
4. `g4_case_boundary.py`: combine detections for cases spanning test-year boundaries before pooled distinct-case summaries. Within-farm-year comparisons are unchanged. The `ew_pooled_unique.csv` and `ew_by_class_pooled_unique.csv` outputs supersede the pooled outputs of the historical `c6_merge.py` for the current manuscript. That historical merger is retained unchanged for provenance.

Input paths are at each script's head and require authorized farm data. Outputs are written to a separate directory, without changing source parquet files. Only anonymized aggregate CSV/JSON outputs are distributed in `derived_data/v12_checks/`.

At common support, mean case detection is 44.49% versus 43.62%; the eight-farm paired interval spans zero. With nine days of random deletion, availability is 83.88% and original-drop retention 49.14%. Timestamp-based D-rule eligibility changes mean detection from 45.606% to 45.575%. These numerical checks do not establish clinical lead time or arbitrary-missingness robustness.
