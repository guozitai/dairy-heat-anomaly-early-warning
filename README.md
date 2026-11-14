# Session-aware milk-yield monitoring

Code accompanying **Unrecorded milkings and session-aware milk-yield monitoring in dairy herds under heat stress**.

The analysis examines incomplete milking records, a cow- and session-specific yield deviation, individual screening against veterinary records, and herd-level response to heat load.

## Current analysis

- `session_aware/analysis/`: session measurement, validation, herd analysis and prediction comparisons.
- `e1_fm.py`, `e2_dl.py`, `e3_summary.py`: Chronos-Bolt, GRU/Transformer and comparison summaries.
- `f1_masking_check.py`: controlled deletion of records on isolated complete-record days, including the numerical 30% yield-reduction test.
- `session_aware/manuscript/`: archived manuscript and figure-generation scripts.

See `SUBMISSION_RELEASE_20261114.md` for execution paths and data requirements. Historical code and documentation remain for provenance.

Commercial farm and cow-level data are not included. Scripts retain their original data paths and require an authorized local copy of the inputs. No trained-model superiority or clinical deployment claim is implied by the model comparisons. The complete-record masking reference is a numerical reference, not a clinical ground truth.
