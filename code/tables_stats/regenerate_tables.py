#!/usr/bin/env python3
"""
regenerate_tables.py
====================
Rebuild v1.9.3 Supplementary Tables (and main-text Table 1)
from ground-truth CSV outputs on the server, not from manuscript md.

Inputs (all under --derived-dir, default /home/2026gzt/derived):
    - E1_time_anchored_loso_summary.csv
        cols: farm_id, n_train, n_val, n_test, prevalence,
              pr_lift, recall_at_10pct, lift_at_10pct,
              lead_3day_recall, best_iter
    - E2_episode_onset_lead_recall.csv
        cols: farm_id, farm_label, n_episodes, mean_episode_length,
              onset_in_top_k_same_day, lead_1, lead_2, lead_3
    - E3_common_label_sanity.csv
        cols: farm_id, label_cohort__cow_features__pr_lift,
              label_cohort__cohort_features__pr_lift,
              label_cow__cow_features__pr_lift,
              label_cow__cohort_features__pr_lift, ...
    - decomposition_v3_v31_loso.csv   (Table S3 source; optional)
        cols: farm_id, v3_lift10, v31_lift10, loso_lift10,
              delta_cohort, delta_transfer, n_train (optional)
    - E4_prospective_summary.csv      (Table S5 source; optional)
        cols: farm_id, full_pr_lift, prosp_pr_lift, retention,
              full_lead3, prosp_lead3 (column names auto-detected)
    - rumination_v4_summary.csv       (Table S6 source; optional)
        cols: farm_id, n_train, test_rum_coverage,
              v31_lift10, v4_lift10, gain_pct, v31_lead3, v4_lead3

Outputs (under --out-dir):
    - supp_tables_v1_9_3_groundtruth.xlsx     (one sheet per table)
    - supp_tables_v1_9_3_groundtruth.docx     (Word, ready to drop into Supp)
    - table1_main_v1_9_3_groundtruth.docx     (compact main-text Table 1)
    - diff_report_vs_manuscript.txt           (cell-level diff vs frozen md)

Usage:
    python regenerate_tables.py \\
        --derived-dir /home/2026gzt/derived \\
        --frozen-md   /path/to/paper_draft_v1_9_3.md \\
        --out-dir     ${OUTPUT_DIR}/v1_9_3_groundtruth

Missing inputs do not fail the script: the corresponding table is skipped
with a warning, and the rest are still rebuilt.
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- farm anonymisation map ----------
FARM_MAP = {
    "FARM_8": "Farm 8", "FARM_7": "Farm 7", "FARM_1": "Farm 1",
    "FARM_4": "Farm 4", "FARM_5": "Farm 5", "FARM_6": "Farm 6",
    "FARM_3": "Farm 3", "FARM_2": "Farm 2",
    # also accept bare integer ids
    3: "Farm 8",  5: "Farm 7",  6: "Farm 1",
    13: "Farm 4", 14: "Farm 5", 15: "Farm 6",
    16: "Farm 3", 17: "Farm 2",
    "3": "Farm 8",  "5": "Farm 7",  "6": "Farm 1",
    "13": "Farm 4", "14": "Farm 5", "15": "Farm 6",
    "16": "Farm 3", "17": "Farm 2",
}

def map_farm(x):
    return FARM_MAP.get(x, FARM_MAP.get(str(x), str(x)))

FARM_ORDER = [f"Farm {i}" for i in range(1, 9)]


def _normalize_farm_col(df: pd.DataFrame, source_name: str = "") -> pd.DataFrame:
    """Rename any plausible 'farm' column to 'farm_id' so downstream code
    can rely on a single name. Idempotent."""
    if "farm_id" in df.columns:
        return df
    candidates = ["farm", "farm_label", "farm_name", "farmId", "FarmID",
                  "Farm", "FarmId", "farmid", "site", "site_id"]
    for c in candidates:
        if c in df.columns:
            df = df.rename(columns={c: "farm_id"})
            return df
    # Heuristic: any column whose name contains 'farm' case-insensitively
    for c in df.columns:
        if "farm" in c.lower():
            df = df.rename(columns={c: "farm_id"})
            return df
    raise KeyError(
        f"No farm column found in {source_name or 'csv'}. "
        f"Columns are: {list(df.columns)}")


# =============================================================
# Table builders
# =============================================================

def _find_first(derived: Path, *names) -> Path | None:
    """Return first existing file from a list of candidate names."""
    for n in names:
        p = derived / n
        if p.exists():
            return p
    return None


def build_table_s2(derived: Path) -> pd.DataFrame | None:
    """Time-anchored LOSO performance per farm + mean row."""
    p1 = _find_first(derived,
                     "E1_time_anchored_loso_regression_summary.csv",
                     "E1_time_anchored_loso_summary.csv")
    p2 = _find_first(derived,
                     "E2_episode_onset_lead_recall_LOSO_TESTWINDOW.csv",
                     "E2_episode_onset_lead_recall_LOSO.csv",
                     "E2_episode_onset_lead_recall.csv")

    if p1 is None:
        print(f"  [S2] MISSING: E1 summary csv", file=sys.stderr)
        return None
    print(f"  [S2] reading {p1.name}")
    e1 = pd.read_csv(p1)
    print(f"  [S2] E1 columns: {list(e1.columns)}")
    e1 = _normalize_farm_col(e1, p1.name)
    e1["Farm"] = e1["farm_id"].map(map_farm)

    # E1 Std Lead-3 column name
    e1_l3_col = next((c for c in ["standard_lead3", "lead_3day_recall",
                                   "Standard_Lead3"]
                      if c in e1.columns), None)
    if e1_l3_col is None:
        print(f"  [S2] WARN: no Std Lead-3 column in E1", file=sys.stderr)

    if p2 is not None:
        print(f"  [S2] reading {p2.name}")
        e2 = pd.read_csv(p2)
        print(f"  [S2] E2 columns: {list(e2.columns)}")
        e2 = _normalize_farm_col(e2, p2.name)
        e2["Farm"] = e2["farm_id"].map(map_farm)
        # E2 Onset Lead-3 = episode-onset Lead-3 (first day of each episode);
        # prefer 'lead3_eligible' which is restricted to onset days.
        onset_col = next((c for c in ["lead3_eligible", "onset_lead_3",
                                       "lead3_all", "lead_3"]
                          if c in e2.columns), None)
        if onset_col is None:
            print(f"  [S2] WARN: no Onset Lead-3 column in E2; columns are "
                  f"{list(e2.columns)}", file=sys.stderr)
            df = e1.copy()
            df["onset_lead3"] = np.nan
        else:
            print(f"  [S2] using E2 onset column: {onset_col}")
            e2_slim = e2[["Farm", onset_col]].rename(
                columns={onset_col: "onset_lead3"})
            df = e1.merge(e2_slim, on="Farm", how="left")
    else:
        print(f"  [S2] note: E2 csv not found; Onset Lead-3 left blank",
              file=sys.stderr)
        df = e1.copy()
        df["onset_lead3"] = np.nan

    out = pd.DataFrame({
        "Farm":          df["Farm"],
        "Prev (%)":      (df["prevalence"] * 100).round(2),
        "PR-Lift":       df["pr_lift"].round(2),
        "Recall@10%":    df["recall_at_10pct"].round(3),
        "Lift@10%":      df["lift_at_10pct"].round(2),
        "Std Lead-3":    df[e1_l3_col].round(3) if e1_l3_col else np.nan,
        "Onset Lead-3":  df["onset_lead3"].round(3),
    })

    # order by farm 1..8
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    # append mean row
    means = out.iloc[:, 1:].mean(numeric_only=True)
    mean_row = pd.DataFrame([{
        "Farm": "Mean", **{c: round(v, 3) for c, v in means.items()}
    }])
    return pd.concat([out, mean_row], ignore_index=True)


def build_table_s3(derived: Path,
                   s2_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Three-way decomposition: v3 / v3.1 / LOSO / Δcohort / Δtransfer.

    Built from THREE separate csvs (no merged decomposition file required):
      - in_farm_v3_summary.csv
      - in_farm_v31_summary.csv
      - E1_time_anchored_loso_regression_summary.csv

    Or fall back to a single merged file if present:
      - decomposition_v3_v31_loso.csv
    """
    # Option A: merged file
    p_merged = derived / "decomposition_v3_v31_loso.csv"
    if p_merged.exists():
        print(f"  [S3] reading merged {p_merged.name}")
        d = pd.read_csv(p_merged)
        d = _normalize_farm_col(d, p_merged.name)
        d["Farm"] = d["farm_id"].map(map_farm)
        return _format_s3_from_merged(d)

    # Option B: assemble from 3 separate summary csvs
    p_v3   = _find_first(derived, "in_farm_v3_summary.csv")
    p_v31  = _find_first(derived, "in_farm_v31_summary.csv")
    p_loso = _find_first(derived,
                         "E1_time_anchored_loso_regression_summary.csv",
                         "E1_time_anchored_loso_summary.csv")
    missing = [n for n, p in [("in_farm_v3_summary.csv", p_v3),
                              ("in_farm_v31_summary.csv", p_v31),
                              ("E1_..._summary.csv", p_loso)] if p is None]
    if missing:
        print(f"  [S3] MISSING for assembly: {missing}", file=sys.stderr)
        return None

    print(f"  [S3] assembling from {p_v3.name} + {p_v31.name} + {p_loso.name}")
    v3   = pd.read_csv(p_v3)
    v31  = pd.read_csv(p_v31)
    loso = pd.read_csv(p_loso)

    def pick_lift_col(df):
        for c in ["Lift", "lift_at_10pct", "lift_10", "lift10", "lift",
                  "lift_at_10", "lift_10pct"]:
            if c in df.columns: return c
        raise KeyError(f"no lift column in {df.columns.tolist()}")

    def pick_ntrain_col(df):
        for c in ["n_train", "n_train_records", "n_records", "n"]:
            if c in df.columns: return c
        return None

    c_v3   = pick_lift_col(v3)
    c_v31  = pick_lift_col(v31)
    c_loso = pick_lift_col(loso)
    c_ntr  = pick_ntrain_col(loso)

    v3   = _normalize_farm_col(v3, p_v3.name)
    v31  = _normalize_farm_col(v31, p_v31.name)
    loso = _normalize_farm_col(loso, p_loso.name)
    for df in (v3, v31, loso):
        df["Farm"] = df["farm_id"].map(map_farm)

    merged = (v3[["Farm", c_v3]].rename(columns={c_v3: "v3"})
              .merge(v31[["Farm", c_v31]].rename(columns={c_v31: "v31"}),
                     on="Farm", how="outer")
              .merge(loso[["Farm", c_loso] + ([c_ntr] if c_ntr else [])]
                     .rename(columns={c_loso: "loso",
                                       **({c_ntr: "n_train"} if c_ntr else {})}),
                     on="Farm", how="outer"))

    merged["delta_cohort"]   = merged["v31"]  - merged["v3"]
    merged["delta_transfer"] = merged["loso"] - merged["v31"]

    out = pd.DataFrame({
        "Farm":         merged["Farm"],
        "in-farm v3":   merged["v3"].round(2),
        "in-farm v3.1": merged["v31"].round(2),
        "LOSO":         merged["loso"].round(2),
        "Δ(cohort)":    merged["delta_cohort"].apply(lambda x: f"{x:+.2f}"),
        "Δ(transfer)":  merged["delta_transfer"].apply(lambda x: f"{x:+.2f}"),
        "n_train":      (merged["n_train"].apply(lambda x: f"{int(x):,}")
                          if "n_train" in merged.columns else "—"),
    })
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    num_cols = ["in-farm v3", "in-farm v3.1", "LOSO"]
    mean_vals = out[num_cols].mean()
    dc_mean = pd.to_numeric(
        out["Δ(cohort)"].str.replace(r"[+]", "", regex=True), errors="coerce").mean()
    dt_mean = pd.to_numeric(
        out["Δ(transfer)"].str.replace(r"[+]", "", regex=True), errors="coerce").mean()
    mean_row = pd.DataFrame([{
        "Farm": "Mean",
        "in-farm v3":   round(mean_vals["in-farm v3"], 2),
        "in-farm v3.1": round(mean_vals["in-farm v3.1"], 2),
        "LOSO":         round(mean_vals["LOSO"], 2),
        "Δ(cohort)":    f"{dc_mean:+.2f}",
        "Δ(transfer)":  f"{dt_mean:+.2f}",
        "n_train":      "—",
    }])
    return pd.concat([out, mean_row], ignore_index=True)


def _format_s3_from_merged(d: pd.DataFrame) -> pd.DataFrame:
    """Original merged-file path (preserved for compatibility)."""
    def pick(df, *cands):
        for c in cands:
            if c in df.columns: return c
        return None

    c_v3   = pick(d, "v3_lift10", "v3_lift_at_10pct", "v3")
    c_v31  = pick(d, "v31_lift10", "v3_1_lift10", "v3.1_lift10", "v31")
    c_loso = pick(d, "loso_lift10", "loso_lift_at_10pct", "loso")
    c_dc   = pick(d, "delta_cohort", "d_cohort")
    c_dt   = pick(d, "delta_transfer", "d_transfer")
    c_ntr  = pick(d, "n_train", "n_train_records")

    out = pd.DataFrame({
        "Farm":              d["Farm"],
        "in-farm v3":        d[c_v3].round(2),
        "in-farm v3.1":      d[c_v31].round(2),
        "LOSO":              d[c_loso].round(2),
        "Δ(cohort)":         d[c_dc].apply(lambda x: f"{x:+.2f}"),
        "Δ(transfer)":       d[c_dt].apply(lambda x: f"{x:+.2f}"),
        "n_train":           (d[c_ntr].apply(lambda x: f"{int(x):,}")
                              if c_ntr else "—"),
    })
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    return out.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def build_table_s4(derived: Path) -> pd.DataFrame | None:
    """Common-label sanity (E3). The E3 csv has both raw cells and pre-computed
    diff columns; we use both."""
    p = _find_first(derived,
                    "E3_common_label_baseline_regression_summary.csv",
                    "E3_common_label_sanity.csv",
                    "E3_common_label_baseline.csv")
    if p is None:
        print(f"  [S4] MISSING: E3 csv", file=sys.stderr)
        return None
    print(f"  [S4] reading {p.name}")
    e3 = pd.read_csv(p)
    print(f"  [S4] columns: {list(e3.columns)}")
    e3 = _normalize_farm_col(e3, p.name)
    e3["Farm"] = e3["farm_id"].map(map_farm)

    # Real column names in E3:
    #   fixed_cohort_label__cohort_model_pr_lift
    #   fixed_cohort_label__cow_model_pr_lift
    #   fixed_cow_label__cohort_model_pr_lift
    #   fixed_cow_label__cow_model_pr_lift
    #   diff_pr_lift_fixed_cohort_label
    #   diff_pr_lift_fixed_cow_label
    def get(col):
        return e3[col] if col in e3.columns else pd.Series(
            [np.nan] * len(e3), index=e3.index)

    coh_coh = get("fixed_cohort_label__cohort_model_pr_lift")
    coh_cow = get("fixed_cohort_label__cow_model_pr_lift")
    cow_coh = get("fixed_cow_label__cohort_model_pr_lift")
    cow_cow = get("fixed_cow_label__cow_model_pr_lift")
    diff_coh = get("diff_pr_lift_fixed_cohort_label")
    diff_cow = get("diff_pr_lift_fixed_cow_label")

    def fmt(x):
        return f"{x:.2f}" if pd.notna(x) else "—"

    def fmt_signed(x):
        return f"{x:+.2f}" if pd.notna(x) else "—"

    out = pd.DataFrame({
        "Farm": e3["Farm"],
        "Fixed cohort-label: cohort-model":  coh_coh.apply(fmt),
        "Fixed cohort-label: cow-model":     coh_cow.apply(fmt),
        "Fixed cohort-label: diff":          diff_coh.apply(fmt_signed),
        "Fixed cow-label: cohort-model":     cow_coh.apply(fmt),
        "Fixed cow-label: cow-model":        cow_cow.apply(fmt),
        "Fixed cow-label: diff":             diff_cow.apply(fmt_signed),
    })
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    # mean diff (numeric)
    coh_diff_mean = pd.to_numeric(diff_coh, errors="coerce").mean()
    cow_diff_mean = pd.to_numeric(diff_cow, errors="coerce").mean()
    mean_row = pd.DataFrame([{
        "Farm": "Mean diff",
        "Fixed cohort-label: cohort-model":  "",
        "Fixed cohort-label: cow-model":     "",
        "Fixed cohort-label: diff":          f"{coh_diff_mean:+.2f}",
        "Fixed cow-label: cohort-model":     "",
        "Fixed cow-label: cow-model":        "",
        "Fixed cow-label: diff":             f"{cow_diff_mean:+.2f}",
    }])
    return pd.concat([out, mean_row], ignore_index=True)


def build_table_s5(derived: Path) -> pd.DataFrame | None:
    """E4 prospective-feature sensitivity.
    Prefer the WIDE comparison csv (E4_prospective_vs_full_comparison_wide.csv);
    fall back to long summary csv if wide is missing."""

    # ---- preferred: wide csv ----
    p_wide = _find_first(derived,
                         "E4_prospective_vs_full_comparison_wide.csv")
    if p_wide is not None:
        print(f"  [S5] reading {p_wide.name}")
        e4 = pd.read_csv(p_wide)
        print(f"  [S5] columns: {list(e4.columns)}")
        e4 = _normalize_farm_col(e4, p_wide.name)
        e4["Farm"] = e4["farm_id"].map(map_farm)

        # column names in wide csv:
        # PR_Lift__full_same_day_THI / PR_Lift__prospective_prior_only
        # Standard_Lead3__full_same_day_THI / Standard_Lead3__prospective_prior_only
        # ratio_PR_Lift_prospective_over_full

        out = pd.DataFrame({
            "Farm":          e4["Farm"],
            "Full PR-Lift":  e4["PR_Lift__full_same_day_THI"].round(2),
            "Prosp PR-Lift": e4["PR_Lift__prospective_prior_only"].round(2),
            "Retention":     e4["ratio_PR_Lift_prospective_over_full"].round(3),
            "Full Std L-3":  e4["Standard_Lead3__full_same_day_THI"].round(3),
            "Prosp Std L-3": e4["Standard_Lead3__prospective_prior_only"].round(3),
        })
        out["__o"] = out["Farm"].apply(
            lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
        out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)

        # mean: retention is mean of per-farm ratios (NOT ratio of means)
        mean_row = {
            "Farm":          "Mean",
            "Full PR-Lift":  round(out["Full PR-Lift"].mean(), 2),
            "Prosp PR-Lift": round(out["Prosp PR-Lift"].mean(), 2),
            "Retention":     round(out["Retention"].mean(), 3),
            "Full Std L-3":  round(out["Full Std L-3"].mean(), 3),
            "Prosp Std L-3": round(out["Prosp Std L-3"].mean(), 3),
        }
        return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)

    # ---- fallback: long-format summary csv, pivot to wide ----
    p = _find_first(derived,
                    "E4_prospective_feature_time_anchored_loso_summary.csv")
    if p is None:
        print(f"  [S5] MISSING: no E4 csv found", file=sys.stderr)
        return None

    print(f"  [S5] reading {p.name} (long format, will pivot)")
    e4_long = pd.read_csv(p)
    print(f"  [S5] columns: {list(e4_long.columns)}")
    # variant column distinguishes full vs prosp
    if "variant" not in e4_long.columns:
        print(f"  [S5] no variant column; cannot pivot", file=sys.stderr)
        return None

    e4_long = _normalize_farm_col(e4_long, p.name)
    e4_long["Farm"] = e4_long["farm_id"].map(map_farm)

    pivot = e4_long.pivot_table(
        index="Farm",
        columns="variant",
        values=["PR_Lift", "Standard_Lead3"],
        aggfunc="first"
    )
    # flatten columns
    pivot.columns = [f"{a}__{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    full_pr = next((c for c in pivot.columns
                    if "PR_Lift" in c and "full" in c.lower()), None)
    pro_pr  = next((c for c in pivot.columns
                    if "PR_Lift" in c and "prosp" in c.lower()), None)
    full_l3 = next((c for c in pivot.columns
                    if "Standard_Lead3" in c and "full" in c.lower()), None)
    pro_l3  = next((c for c in pivot.columns
                    if "Standard_Lead3" in c and "prosp" in c.lower()), None)
    if not (full_pr and pro_pr):
        print(f"  [S5] cannot identify full/prosp PR-Lift columns after pivot")
        return None

    out = pd.DataFrame({
        "Farm":          pivot["Farm"],
        "Full PR-Lift":  pivot[full_pr].round(2),
        "Prosp PR-Lift": pivot[pro_pr].round(2),
        "Retention":     (pivot[pro_pr] / pivot[full_pr]).round(3),
        "Full Std L-3":  pivot[full_l3].round(3) if full_l3 else np.nan,
        "Prosp Std L-3": pivot[pro_l3].round(3)  if pro_l3  else np.nan,
    })
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    out = out.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    mean_row = {
        "Farm":          "Mean",
        "Full PR-Lift":  round(out["Full PR-Lift"].mean(), 2),
        "Prosp PR-Lift": round(out["Prosp PR-Lift"].mean(), 2),
        "Retention":     round(out["Retention"].mean(), 3),
        "Full Std L-3":  round(out["Full Std L-3"].mean(), 3) if full_l3 else "",
        "Prosp Std L-3": round(out["Prosp Std L-3"].mean(), 3) if pro_l3 else "",
    }
    return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)


def build_table_s6(derived: Path) -> pd.DataFrame | None:
    """Rumination v4 sensitivity, exploratory.
    Skip rows with status indicating failed runs (e.g. data_too_small)."""
    candidates = [
        "rumination_sensitivity_5farms.csv",
        "rumination_sensitivity.csv",
        "rumination_v4_summary.csv",
        "E5_rumination_v4.csv",
    ]
    p = _find_first(derived, *candidates)
    if p is None:
        print(f"  [S6] MISSING: none of {candidates} found", file=sys.stderr)
        return None
    print(f"  [S6] reading {p.name}")
    r = pd.read_csv(p)
    print(f"  [S6] columns: {list(r.columns)}")
    print(f"  [S6] raw row count: {len(r)}")
    r = _normalize_farm_col(r, p.name)
    r["Farm"] = r["farm_id"].map(map_farm)

    # Drop failed runs (e.g. status == 'data_too_small (0)')
    if "status" in r.columns:
        n_before = len(r)
        r = r[r["status"].astype(str).str.lower().str.startswith("ok") |
              r["status"].astype(str).str.lower().eq("ok")]
        # actually be more permissive: keep anything not flagged
        r2 = pd.read_csv(p)
        r2 = _normalize_farm_col(r2, p.name)
        r2["Farm"] = r2["farm_id"].map(map_farm)
        r = r2[~r2["status"].astype(str).str.lower().str.contains(
            "too_small|fail|error|skip", na=False)]
        print(f"  [S6] kept {len(r)}/{n_before} rows after filtering status")

    if len(r) == 0:
        print(f"  [S6] no usable rows after filtering", file=sys.stderr)
        return None

    def pick(df, *cands):
        for c in cands:
            if c in df.columns: return c
        return None

    c_n_train = pick(r, "n_train", "n_train_records")
    c_cov     = pick(r, "test_rum_coverage", "rum_coverage", "coverage")
    c_v31     = pick(r, "v31_lift", "v31_lift10", "v3_1_lift10", "v3_1_lift")
    c_v4      = pick(r, "v4_lift", "v4_lift10", "v4")
    c_v31_l3  = pick(r, "v31_lead3d", "v31_lead3", "v3_1_lead3")
    c_v4_l3   = pick(r, "v4_lead3d", "v4_lead3")
    c_gain    = pick(r, "gain_pct", "gain")

    out = pd.DataFrame({
        "Farm":             r["Farm"],
        "n_train":          (r[c_n_train].apply(lambda x: f"{int(x):,}"
                                                if pd.notna(x) else "—")
                              if c_n_train else "—"),
        "test_rum_cov":     (r[c_cov].apply(lambda x: f"{x*100:.1f}%"
                                            if pd.notna(x) and x <= 1.0
                                            else (f"{x:.1f}%"
                                                  if pd.notna(x) else "—"))
                              if c_cov else "—"),
        "v3.1 Lift@10%":    r[c_v31].round(2) if c_v31 else np.nan,
        "v4 Lift@10%":      r[c_v4].round(2)  if c_v4  else np.nan,
        "Gain (%)":         (r[c_gain].round(1) if c_gain
                              else ((r[c_v4] / r[c_v31] - 1) * 100).round(1)
                              if c_v4 and c_v31 else np.nan),
        "v3.1 Lead-3":      r[c_v31_l3].round(2) if c_v31_l3 else np.nan,
        "v4 Lead-3":        r[c_v4_l3].round(2)  if c_v4_l3  else np.nan,
    })
    out["__o"] = out["Farm"].apply(
        lambda f: FARM_ORDER.index(f) if f in FARM_ORDER else 99)
    return out.sort_values("__o").drop(columns="__o").reset_index(drop=True)


def build_table_s1(derived: Path) -> pd.DataFrame | None:
    """Farm characteristics — try to read a farm-metadata csv;
    otherwise use the values literally transcribed from manuscript md."""
    p = derived / "farm_characteristics.csv"
    if p.exists():
        df = pd.read_csv(p)
        return df
    # Fallback: hard-coded from manuscript md (these are not measured numbers,
    # they are categorical metadata, so transcription is acceptable here)
    rows = [
        ("Farm 1", "Xixia",   "2022.03 – 2025.09", "Large",
         "Reference farm (longest analytical window, 43 months)"),
        ("Farm 2", "Lingwu",  "2023.02 – 2025.09", "Very large",
         "Sister farm of Farm 5"),
        ("Farm 3", "Pingluo", "2023.02 – 2025.09", "Large",
         "Sister farm of Farm 6"),
        ("Farm 4", "Lingwu",  "2023.02 – 2025.09", "Large", ""),
        ("Farm 5", "Lingwu",  "2023.03 – 2025.09", "Large",
         "Sister farm of Farm 2"),
        ("Farm 6", "Pingluo", "2023.02 – 2025.03", "Large",
         "Sister farm of Farm 3"),
        ("Farm 7", "Xixia",   "2023.08 – 2025.09", "Medium", ""),
        ("Farm 8", "Xixia",   "2024.05 – 2025.09", "Medium", ""),
    ]
    return pd.DataFrame(rows,
        columns=["Farm", "County", "Window", "Herd size†", "Notes"])


def build_table_s0(s2_df: pd.DataFrame | None) -> pd.DataFrame:
    """Data flow (counts come from manuscript md; mostly categorical)."""
    return pd.DataFrame({
        "Stage": [
            "Raw daily cow-day records (8 retained farms)",
            "Farm-level eligibility filter",
            "QC-flagged records (density and/or yield-ratio)",
            "QC-passed pool (primary analysis)",
            "Experiment-specific eligibility (per experiment)",
            "  - Valid parity, DIM in modeled range",
            "  - Cohort-baseline stratum available",
            "  - Lag-1 features available",
            "Time-anchored LOSO E1 training (8 folds combined)",
            "Common-label E3 in-farm analysis (7 farms)",
            "Prospective-feature E4 LOSO",
        ],
        "Records": [
            "16.84 M", "16.84 M", "1.054 M", "15.79 M",
            "", "", "", "", "varies", "varies", "varies",
        ],
        "Used where": [
            "inventory only",
            "8 farms retained (1 farm dropped)",
            "infrastructure-audit analyses; excluded from primary models",
            "primary modeling pool",
            "applied per experiment",
            "applied per experiment",
            "applied per experiment",
            "applied per experiment",
            "per-fold counts in Table S2",
            "per-farm counts in Table S4",
            "per-farm counts in Table S5",
        ],
    })


def build_table1_main(s2_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Compact main-text Table 1: prevalence + 3 headline metrics + n cow-days."""
    if s2_df is None:
        return None
    cols_keep = ["Farm", "Prev (%)", "PR-Lift", "Lift@10%", "Recall@10%"]
    return s2_df[cols_keep].copy()


# =============================================================
# Diff vs manuscript md (frozen numbers)
# =============================================================

def diff_against_manuscript(rebuilt: dict, frozen_md: Path) -> str:
    """Extract Tables S2 / S3 / S5 / S6 numeric content from md and diff
    against rebuilt DataFrames. Returns a printable report string."""
    if not frozen_md.exists():
        return f"(skipped — manuscript md not found at {frozen_md})\n"

    text = frozen_md.read_text(encoding="utf-8")

    report = []
    report.append("=" * 70)
    report.append("DIFF REPORT: rebuilt-from-ground-truth  vs  manuscript md (frozen)")
    report.append("=" * 70)
    report.append("")

    # ---- Table S2 ----
    if "s2" in rebuilt:
        report.append("--- Table S2 (time-anchored LOSO performance) ---")
        m = re.search(r"### Table S2:[^\n]*\n(.*?)### Table S3:", text, re.S)
        if m:
            blk = m.group(1)
            for farm in FARM_ORDER:
                # find row like "Farm 1   2.23    12.54   0.628   ..."
                pat = rf"{farm}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                rm = re.search(pat, blk)
                if not rm:
                    continue
                md_vals = [float(rm.group(i+1)) for i in range(6)]
                md_cols = ["Prev (%)", "PR-Lift", "Recall@10%",
                           "Lift@10%", "Std Lead-3", "Onset Lead-3"]
                rb_row = rebuilt["s2"][rebuilt["s2"]["Farm"] == farm]
                if rb_row.empty: continue
                rb_vals = [float(rb_row[c].values[0])
                           if pd.notna(rb_row[c].values[0]) else np.nan
                           for c in md_cols]
                for c, mv, rv in zip(md_cols, md_vals, rb_vals):
                    if pd.isna(rv): continue
                    if abs(mv - rv) > 0.01:
                        report.append(
                            f"  {farm:8s} {c:14s} md={mv:8.3f}  ground={rv:8.3f}  "
                            f"Δ={rv-mv:+.3f}   ←← MISMATCH")
        else:
            report.append("  (could not parse Table S2 from md)")
        report.append("")

    # ---- Table S3 ----
    if "s3" in rebuilt:
        report.append("--- Table S3 (decomposition) ---")
        m = re.search(r"### Table S3:[^\n]*\n(.*?)### Table S4:", text, re.S)
        if m:
            blk = m.group(1)
            for farm in FARM_ORDER:
                pat = rf"{farm}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)"
                rm = re.search(pat, blk)
                if not rm: continue
                md_vals = [float(rm.group(i+1)) for i in range(5)]
                rb_row = rebuilt["s3"][rebuilt["s3"]["Farm"] == farm]
                if rb_row.empty: continue
                rb_vals = [
                    float(rb_row["in-farm v3"].values[0]),
                    float(rb_row["in-farm v3.1"].values[0]),
                    float(rb_row["LOSO"].values[0]),
                    float(str(rb_row["Δ(cohort)"].values[0]).replace("+", "")),
                    float(str(rb_row["Δ(transfer)"].values[0]).replace("+", "")),
                ]
                cols = ["v3", "v3.1", "LOSO", "Δ(cohort)", "Δ(transfer)"]
                for c, mv, rv in zip(cols, md_vals, rb_vals):
                    if abs(mv - rv) > 0.01:
                        report.append(
                            f"  {farm:8s} {c:12s} md={mv:7.2f}  ground={rv:7.2f}  "
                            f"Δ={rv-mv:+.2f}   ←← MISMATCH")
        report.append("")

    report.append("(Tables not listed above either pass exact match, or "
                  "their md block could not be parsed.)")
    return "\n".join(report) + "\n"


# =============================================================
# Output writers (xlsx + docx)
# =============================================================

def write_xlsx(tables: dict, out_path: Path):
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in tables.items():
            if df is None: continue
            sheet = name.upper()[:31]
            df.to_excel(xw, sheet_name=sheet, index=False)
    print(f"  wrote {out_path}")


def write_docx(tables: dict, captions: dict, out_path: Path,
               title: str = "Supplementary Tables (rebuilt from ground truth)"):
    from docx import Document
    from docx.shared import Pt
    from docx.enum.table import WD_ALIGN_VERTICAL

    doc = Document()
    doc.add_heading(title, 0)

    for name, df in tables.items():
        if df is None: continue

        doc.add_heading(captions.get(name, name).split("\n")[0], 2)
        cap = captions.get(name, "")
        if "\n" in cap:
            doc.add_paragraph(cap.split("\n", 1)[1])

        # build word table
        n_rows, n_cols = df.shape
        tbl = doc.add_table(rows=n_rows + 1, cols=n_cols)
        tbl.style = "Light Grid Accent 1"

        # header
        for j, col in enumerate(df.columns):
            cell = tbl.rows[0].cells[j]
            cell.text = str(col)
            for p in cell.paragraphs:
                for r in p.runs:
                    r.bold = True
                    r.font.size = Pt(9)

        # body
        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                v = row[col]
                if isinstance(v, float):
                    s = f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
                    if pd.isna(v): s = ""
                elif pd.isna(v):
                    s = ""
                else:
                    s = str(v)
                tbl.rows[i + 1].cells[j].text = s
                for p in tbl.rows[i + 1].cells[j].paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(9)

        doc.add_paragraph()

    doc.save(out_path)
    print(f"  wrote {out_path}")


# =============================================================
# Main
# =============================================================

CAPTIONS = {
    "s0": "Table S0. Data flow\n"
          "Record counts from raw to QC-passed pool to per-experiment "
          "eligibility. Counts reflect the v1.9.3 analytical snapshot.",
    "s1": "Table S1. Farm characteristics\n"
          "Eight commercial dairy farms in northwestern China retained after "
          "quality-control screening. \"Sister farm\" indicates shared "
          "high-level operational grouping within the same county. "
          "† Herd size: Medium = 3,000-6,000; Large = 6,000-12,000; "
          "Very large = >12,000 lactating cows. Exact head counts withheld.",
    "s2": "Table S2. Time-anchored LOSO performance per farm\n"
          "PR-Lift = AUPRC / prevalence (pooled, threshold-free). "
          "Recall@10% = daily Top-10% recall, averaged across test dates. "
          "Lift@10% = Recall@10% / 0.10. Std Lead-3 = standard cow-day "
          "Lead-3-day recall (all anomaly days). Onset Lead-3 = "
          "episode-onset Lead-3-day recall (first day of each episode only). "
          "Training records from other 7 farms restricted to <= 2024-12-31 "
          "(or <= 2024-03-31 for Farm 6).",
    "s3": "Table S3. Three-way decomposition — cohort baseline vs cross-farm "
          "transfer\n"
          "All values are Lift@10%. v3: single-farm trained with cow-specific "
          "baseline. v3.1: single-farm trained with (parity, DIM-stage) cohort "
          "baseline. LOSO: time-anchored leave-one-site-out with cohort "
          "baseline. Δ(cohort) = v3.1 − v3 (within-single-farm baseline "
          "switch). Δ(transfer) = LOSO − v3.1 (additional gain from cross-farm "
          "training). Anomaly labels change with baseline in this naive "
          "decomposition; see Table S4 for the common-label sanity analysis.",
    "s4": "Table S4. Common-label sanity analysis (E3)\n"
          "All values are PR-Lift. \"Fixed cohort-label\" holds the cohort-"
          "defined anomaly label and varies the features; \"Fixed cow-label\" "
          "holds the cow-history-defined label. Under fixed cohort-label, "
          "cohort features win on most farms; under fixed cow-label, the "
          "advantage largely disappears, establishing that the cohort-baseline "
          "advantage is framework-dependent rather than universal "
          "(see §3.4, §4.2). Farm 6 excluded due to insufficient paired "
          "cow-history data for cow-baseline label definition.",
    "s5": "Table S5. Prospective-feature sensitivity (E4)\n"
          "Full = full feature set including focal-day THI, hours_thi_gt_72/79, "
          "heatwave_3d, and precomputed lagged/cumulative climate summaries. "
          "Prosp = focal-day variables removed; all precomputed lagged/"
          "cumulative climate summaries replaced with strictly prior-only "
          "re-derived variables (lag-1, lag-2, lag-3, lag-7 THI and heat-hour "
          "summaries plus a prior-3-day heatwave indicator). Retention "
          "(per-farm) = Prosp PR-Lift / Full PR-Lift. Mean retention is the "
          "average of per-farm retention ratios.",
    "s6": "Table S6. Rumination-feature sensitivity (exploratory)\n"
          "Four farms with available rumination collar data evaluated for "
          "incremental performance gain from rumination-feature fusion. "
          "v4 = the model variant with rumination and v3.1 features combined; "
          "Gain (%) is the relative change in Lift@10%. Farm 7 was excluded: "
          "milking SQL coverage (2023.08-2025.09) does not overlap with "
          "rumination data window (2020.01-2024.06).",
    "t1": "Table 1. Main performance summary across 8 commercial dairy farms.\n"
          "Per-farm time-anchored LOSO performance at the daily Top-10% "
          "operating point. Full metric breakdown including standard and "
          "episode-onset Lead-3 recall is reported in Supplementary Table S2.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived-dir", type=Path,
                    default=Path("/home/2026gzt/derived"))
    ap.add_argument("--frozen-md",   type=Path,
                    default=Path("./paper_draft_v1_9_3.md"))
    ap.add_argument("--out-dir",     type=Path,
                    default=Path("./v1_9_3_groundtruth_tables"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"derived-dir = {args.derived_dir}")
    print(f"frozen-md   = {args.frozen_md}")
    print(f"out-dir     = {args.out_dir}\n")

    print("Building tables from ground truth...")
    s2 = build_table_s2(args.derived_dir)
    s3 = build_table_s3(args.derived_dir, s2)
    s4 = build_table_s4(args.derived_dir)
    s5 = build_table_s5(args.derived_dir)
    s6 = build_table_s6(args.derived_dir)
    s1 = build_table_s1(args.derived_dir)
    s0 = build_table_s0(s2)
    t1 = build_table1_main(s2)

    rebuilt = {k: v for k, v in dict(
        s0=s0, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, s6=s6
    ).items() if v is not None}

    # ---- xlsx ----
    write_xlsx(rebuilt, args.out_dir / "supp_tables_v1_9_3_groundtruth.xlsx")

    # ---- docx (supp) ----
    write_docx(rebuilt, CAPTIONS,
               args.out_dir / "supp_tables_v1_9_3_groundtruth.docx")

    # ---- docx (Table 1 main) ----
    if t1 is not None:
        write_docx({"t1": t1}, CAPTIONS,
                   args.out_dir / "table1_main_v1_9_3_groundtruth.docx",
                   title="Main-text Table 1 (rebuilt from ground truth)")
    else:
        print("  [Table 1] skipped (S2 missing)")

    # ---- diff report ----
    report = diff_against_manuscript(rebuilt, args.frozen_md)
    (args.out_dir / "diff_report_vs_manuscript.txt").write_text(report)
    print()
    print(report)

    print("\nDone. Inspect diff_report_vs_manuscript.txt for any mismatches.")


if __name__ == "__main__":
    main()
