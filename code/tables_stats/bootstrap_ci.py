#!/usr/bin/env python3
"""
bootstrap_ci.py  (v2)
=====================
Cow-clustered nonparametric bootstrap 95% CI for time-anchored LOSO metrics,
using metric functions byte-identical to E1_time_anchored_loso.py so point
estimates exactly match Table S2.

Fixes vs v1:
  - top_k_recall: GLOBAL recall under per-day Top-K ranking
        sum(picked anomalies) / sum(all anomalies)
        with picked decided within each day.
        NOT per-day-averaged recall.
  - lead_k_recall: checks t-1 .. t-k only (NOT same-day t).
  - lead_k_recall vectorised via cow -> set[day_int] dict
        (~100x faster than iterrows; needed for B=500 to be feasible).
  - reset_index after iloc[boot_idx] to avoid duplicate-index errors.
  - serial (workers=1) to sidestep multiprocessing fork deadlocks.

Usage:
    # sanity check (B=20, ~1-2 min)
    python bootstrap_ci.py --pred <parquet> --B 20

    # final (B=500, ~20-40 min serial)
    nohup python bootstrap_ci.py --pred <parquet> --B 500 \\
          --out ./bootstrap_ci_final > bootstrap.log 2>&1 &
"""

from __future__ import annotations
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


FARM_ORDER = [f"Farm {i}" for i in range(1, 9)]


# =============================================================
# Metric primitives — byte-identical to E1_time_anchored_loso.py
# =============================================================

def pr_lift(y_true, y_score):
    """AUPRC / prevalence."""
    prev = y_true.mean()
    if prev == 0:
        return np.nan
    return average_precision_score(y_true, y_score) / prev


def top_k_recall(y_true, y_score, k_frac=0.10, by_day=None):
    """Daily Top-K% recall, GLOBAL flavour.

    Rank within each day, then return
        sum(picked anomalies) / sum(all anomalies).
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "s": np.asarray(y_score)})
    if by_day is not None:
        df["day"] = np.asarray(by_day)
        df["rank"] = df.groupby("day")["s"].rank(ascending=False, method="first")
        df["n_day"] = df.groupby("day")["y"].transform("count")
        df["k_threshold"] = (df["n_day"] * k_frac).clip(lower=1).round()
        df["picked"] = df["rank"] <= df["k_threshold"]
    else:
        n = len(df)
        k = max(1, int(round(n * k_frac)))
        df = df.sort_values("s", ascending=False)
        df["picked"] = False
        df.iloc[:k, df.columns.get_loc("picked")] = True
    n_pos = df["y"].sum()
    if n_pos == 0:
        return np.nan
    return df.loc[df["picked"], "y"].sum() / n_pos


def lead_k_recall(df, k_days=3, k_frac=0.10):
    """Standard (non-episode-onset) Lead-K-day recall.

    Same semantics as E1 (checks t-1 .. t-k, NOT same-day t),
    but vectorised via cow -> set[day_int] for speed.

    df must have columns: cow_id, day, y, s
    """
    df = df.reset_index(drop=True).copy()
    df["day"] = pd.to_datetime(df["day"])
    df["day_int"] = (df["day"].astype("int64") // 86_400_000_000_000).astype(np.int64)

    df["rank"] = df.groupby("day")["s"].rank(ascending=False, method="first")
    df["n_day"] = df.groupby("day")["y"].transform("count")
    df["k_threshold"] = (df["n_day"] * k_frac).clip(lower=1).round()
    df["picked"] = df["rank"] <= df["k_threshold"]

    picked_by_cow = (
        df.loc[df["picked"], ["cow_id", "day_int"]]
        .groupby("cow_id")["day_int"]
        .apply(set)
        .to_dict()
    )

    anomalies = df.loc[df["y"] == 1, ["cow_id", "day_int"]]
    if len(anomalies) == 0:
        return np.nan

    cows = anomalies["cow_id"].values
    days = anomalies["day_int"].values

    hits = 0
    for i in range(len(anomalies)):
        cow = cows[i]
        t   = days[i]
        picked_days = picked_by_cow.get(cow)
        if not picked_days:
            continue
        for lag in range(1, k_days + 1):
            if (t - lag) in picked_days:
                hits += 1
                break
    return hits / len(anomalies)


def metrics_for_farm(fdf: pd.DataFrame) -> dict:
    """fdf needs: cow_id, day, event, risk_score."""
    if len(fdf) == 0 or fdf["event"].sum() == 0:
        return dict(pr_lift=np.nan, recall_10=np.nan,
                    lift_10=np.nan, std_lead3=np.nan)
    y_true  = fdf["event"].values
    y_score = fdf["risk_score"].values
    prl = pr_lift(y_true, y_score)
    rec = top_k_recall(y_true, y_score, k_frac=0.10,
                       by_day=fdf["day"].values)
    lft = rec / 0.10 if not np.isnan(rec) else np.nan
    df_l3 = fdf[["cow_id", "day", "event", "risk_score"]].rename(
        columns={"event": "y", "risk_score": "s"})
    l3 = lead_k_recall(df_l3, k_days=3, k_frac=0.10)
    return dict(pr_lift=prl, recall_10=rec, lift_10=lft, std_lead3=l3)


# =============================================================
# Bootstrap loop  (serial)
# =============================================================

def run_bootstrap_serial(pred: pd.DataFrame, B: int, seed: int = 42,
                         log_every: int = 10, workers: int = 1):
    """Bootstrap loop.

    workers == 1: serial across farms and replicates.
    workers >  1: farm-level parallel — each worker owns ONE farm and runs
                  all B replicates for it. With 8 farms, workers=8 saturates
                  parallelism; more workers don't help.  Each worker is
                  independent (no shared mutable state), avoiding the
                  fork-deadlock issues of replicate-level parallelism.
    """

    # Per-farm working snapshots
    farm_work = {}
    for farm, fdf in pred.groupby("farm_label"):
        fdf = fdf.reset_index(drop=True)
        cow_to_rows = {cow: sub.index.to_numpy()
                       for cow, sub in fdf.groupby("cow_id")}
        farm_work[farm] = dict(rows=fdf,
                                cow_to_rows=cow_to_rows,
                                cows=np.array(list(cow_to_rows.keys())))
        print(f"  {farm}: {len(fdf):,} cow-days, "
              f"{len(cow_to_rows):,} unique cows")

    if workers <= 1:
        return _run_serial(farm_work, B, seed, log_every)
    else:
        return _run_parallel_per_farm(farm_work, B, seed, workers)


def _bootstrap_one_farm(farm: str, w: dict, B: int, seed: int) -> list:
    """Run B replicates for a single farm. Returns list of metric dicts."""
    rng  = np.random.default_rng(seed)
    cows = w["cows"]
    rows = w["rows"]
    c2r  = w["cow_to_rows"]
    out = []
    for b in range(B):
        picked = rng.choice(cows, size=len(cows), replace=True)
        boot_idx = np.concatenate([c2r[c] for c in picked])
        boot_df = rows.iloc[boot_idx].reset_index(drop=True)
        out.append(metrics_for_farm(boot_df))
    return out


def _run_serial(farm_work, B, seed, log_every):
    rng = np.random.default_rng(seed)
    boot_results = []
    t0 = time.time()
    for b in range(B):
        rep = {}
        for farm, w in farm_work.items():
            cows = w["cows"]
            picked_cows = rng.choice(cows, size=len(cows), replace=True)
            boot_idx = np.concatenate([w["cow_to_rows"][c] for c in picked_cows])
            boot_df = w["rows"].iloc[boot_idx].reset_index(drop=True)
            rep[farm] = metrics_for_farm(boot_df)
        boot_results.append(rep)
        if (b + 1) % log_every == 0 or b == 0 or b == B - 1:
            elapsed = time.time() - t0
            eta = elapsed * (B - b - 1) / (b + 1)
            print(f"  [{b+1:4d}/{B}]  elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                  flush=True)
    print(f"\nBootstrap done. {B} replicates in {time.time() - t0:.1f}s.")
    return boot_results


def _run_parallel_per_farm(farm_work, B, seed, workers):
    """Each worker = one farm. Returns list[dict[farm, metric_dict]] of len B."""
    from joblib import Parallel, delayed

    farms = list(farm_work.keys())
    print(f"\nFarm-level parallel: {len(farms)} farms × B={B} replicates "
          f"on {workers} workers")
    print("(each worker holds one farm's data; ~zero cross-worker overhead)",
          flush=True)
    t0 = time.time()

    # Different seed per farm so streams are independent
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(farms))

    per_farm_lists = Parallel(n_jobs=workers, backend="loky", verbose=10)(
        delayed(_bootstrap_one_farm)(farm, farm_work[farm], B, int(s))
        for farm, s in zip(farms, seeds)
    )

    # transpose: list of B dicts {farm: metrics}
    boot_results = []
    for b in range(B):
        rep = {farm: per_farm_lists[i][b] for i, farm in enumerate(farms)}
        boot_results.append(rep)

    elapsed = time.time() - t0
    print(f"\nBootstrap done. {B} replicates × {len(farms)} farms in "
          f"{elapsed:.1f}s  ({elapsed/B:.2f}s per replicate equivalent).")
    return boot_results


def summarise_ci(results, point_metrics):
    rows = []
    for farm in FARM_ORDER:
        for m in ["pr_lift", "recall_10", "lift_10", "std_lead3"]:
            vals = np.array([r[farm][m] for r in results
                             if farm in r and not np.isnan(r[farm][m])])
            if len(vals) == 0: continue
            ci_lo = float(np.percentile(vals, 2.5))
            ci_hi = float(np.percentile(vals, 97.5))
            point = point_metrics[farm].get(m, np.nan)
            rows.append(dict(
                Farm=farm, metric=m,
                point=round(float(point), 4),
                ci_lo=round(ci_lo, 4),
                ci_hi=round(ci_hi, 4),
                B_used=int(len(vals)),
            ))
    return pd.DataFrame(rows)


def widen_for_table_s2(ci_df: pd.DataFrame) -> pd.DataFrame:
    def fmt(p, lo, hi, d=2):
        return f"{p:.{d}f}  [{lo:.{d}f}, {hi:.{d}f}]"
    rows = []
    for farm in FARM_ORDER:
        farm_ci = ci_df[ci_df["Farm"] == farm].set_index("metric")
        if farm_ci.empty: continue
        def get(m, d=2):
            if m not in farm_ci.index: return ""
            r = farm_ci.loc[m]
            return fmt(r["point"], r["ci_lo"], r["ci_hi"], d=d)
        rows.append(dict(
            Farm=farm,
            **{"PR-Lift [95% CI]":    get("pr_lift",   2)},
            **{"Recall@10% [95% CI]": get("recall_10", 3)},
            **{"Lift@10% [95% CI]":   get("lift_10",   2)},
            **{"Std Lead-3 [95% CI]": get("std_lead3", 3)},
        ))
    return pd.DataFrame(rows)


# =============================================================
# Main
# =============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8,
                    help="Farm-level parallel workers (default 8 = saturate "
                         "the 8 farms). Set 1 for serial fallback.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("./bootstrap_ci"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions: {args.pred}")
    pred = pd.read_parquet(args.pred)
    keep = ["farm_label", "cow_id", "day", "event", "risk_score"]
    missing = [c for c in keep if c not in pred.columns]
    if missing:
        raise SystemExit(f"parquet missing columns: {missing}; "
                         f"have: {list(pred.columns)}")
    pred = pred[keep].copy().dropna()
    pred["event"]      = pred["event"].astype(int)
    pred["day"]        = pd.to_datetime(pred["day"])
    pred["farm_label"] = pred["farm_label"].astype(str)
    print(f"  Rows: {len(pred):,}  cows: {pred['cow_id'].nunique():,}  "
          f"farms: {pred['farm_label'].nunique()}")

    # auto-detect risk-score direction
    sample = pred.sample(min(200000, len(pred)), random_state=0)
    def quick_lift(df):
        r = top_k_recall(df["event"].values, df["risk_score"].values,
                         k_frac=0.10, by_day=df["day"].values)
        return (r / 0.10) if not np.isnan(r) else np.nan
    lift_desc = quick_lift(sample)
    sample_flipped = sample.copy()
    sample_flipped["risk_score"] = -sample_flipped["risk_score"]
    lift_asc = quick_lift(sample_flipped)
    if not np.isnan(lift_asc) and lift_asc > lift_desc:
        print(f"  Flipping risk_score sign "
              f"(lift_asc={lift_asc:.2f} > lift_desc={lift_desc:.2f})")
        pred["risk_score"] = -pred["risk_score"]
    else:
        print(f"  Using risk_score as-is (lift_desc={lift_desc:.2f})")

    # point estimates — should match Table S2
    print("\nComputing point estimates per farm...")
    point_metrics = {}
    for farm, fdf in pred.groupby("farm_label"):
        m = metrics_for_farm(fdf.reset_index(drop=True))
        point_metrics[farm] = m
        print(f"  {farm}: PR-Lift={m['pr_lift']:.2f}  "
              f"Recall@10%={m['recall_10']:.3f}  "
              f"Lift@10%={m['lift_10']:.2f}  "
              f"Std L-3={m['std_lead3']:.3f}", flush=True)

    print(f"\nRunning bootstrap (B={args.B}, workers={args.workers})...")
    results = run_bootstrap_serial(pred, B=args.B, seed=args.seed,
                                    workers=args.workers)

    ci_df = summarise_ci(results, point_metrics)
    long_path = args.out / "bootstrap_ci_per_farm.csv"
    ci_df.to_csv(long_path, index=False)
    print(f"\nWrote long-format CI: {long_path}")

    wide = widen_for_table_s2(ci_df)
    wide_path = args.out / "bootstrap_ci_summary_wide.csv"
    wide.to_csv(wide_path, index=False)
    print(f"Wrote wide-format:    {wide_path}")

    print("\nSample of long-format output:")
    print(ci_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
