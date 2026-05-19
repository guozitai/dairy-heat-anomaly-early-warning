#!/usr/bin/env python3
"""
E1: Time-anchored LOSO experiment.

Defense against GPT-2 Attack 2 (LOSO time leakage):
The current retrospective LOSO uses other-farm data from the entire study window,
including the would-be test period. This conflicts with deployment-oriented framing.

Time-anchored LOSO simulates real cold-start deployment:
- For each held-out farm:
    - Test window = the farm's original test window (unchanged)
    - Training data from OTHER farms = restricted to records <= 2024-12-31
      (NOT 2025-04 to 2025-09, which is the test season we're "deploying to")

Reports: PR-Lift, Recall@10%, Lift@10%, Lead-3-day recall per farm.

Run after main pipeline; reuses existing data parquets.
Assumes you have:
  - /home/2026gzt/derived/unified_daily_clean.parquet  (or wherever main data lives)
  - lightgbm trained per existing LOSO config
  - Features and label definition unchanged from v3.1 (cohort baseline)

USER ACTION REQUIRED:
1. Update DATA_PATH, MODEL_CONFIG, FEATURES below to match your existing pipeline
2. Run on JupyterLab server (<server IP redacted>)
3. Output saved to /home/2026gzt/derived/E1_time_anchored_loso_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import json

# =============================================================
# CONFIG — ADJUST TO YOUR PIPELINE
# =============================================================
DATA_PATH = "/home/2026gzt/derived/unified_daily_clean.parquet"
OUT_DIR = Path("/home/2026gzt/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 8 retained farms — match your farm IDs to Farm 1-8 labels
# Adjust to your actual farm_id values
FARM_MAP = {
    "FARM_1": "Farm 1",
    "FARM_2": "Farm 2",
    "FARM_3": "Farm 3",
    "FARM_4": "Farm 4",
    "FARM_5": "Farm 5",
    "FARM_6": "Farm 6",
    "FARM_7": "Farm 7",
    "FARM_8": "Farm 8",
}

# Time-anchor cutoff: other-farm training data restricted to <= this date
# Simulates "we don't have other-farm data from the test season"
TRAINING_CUTOFF = pd.Timestamp("2024-12-31")

# Test windows per farm — adjust based on your existing LOSO test windows
# Format: { farm_id : (test_start, test_end) }
TEST_WINDOWS = {
    "FARM_1": ("2025-04-01", "2025-09-30"),
    "FARM_2": ("2025-04-01", "2025-09-30"),
    "FARM_3": ("2025-04-01", "2025-09-30"),
    "FARM_4": ("2025-04-01", "2025-09-30"),
    "FARM_5": ("2025-04-01", "2025-09-30"),
    "FARM_6": ("2024-08-12", "2025-09-30"),  # Farm 6: shorter window if applicable
    "FARM_7": ("2025-04-01", "2025-09-30"),
    "FARM_8": ("2025-04-01", "2025-09-30"),
}

# Features — must match your v3.1 feature set
FEATURES = [
    # climate
    "thi_max", "thi_max_lag1", "thi_max_lag3_avg",
    "hours_thi_gt_72", "hours_thi_gt_79", "thi_max_lag7_cum", "heatwave_3d",
    # cow / production
    "dim", "parity_group",
    "expected_milk_cohort",       # cohort baseline
    "milk_lag1", "ratio_lag1",
]

LABEL_COL = "anomaly_label"   # 1 if r = actual/expected < threshold
THRESHOLD = 0.85              # Replace with the actual threshold used in your v3.1

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": 16,
}

NUM_BOOST_ROUND = 1500
EARLY_STOPPING = 50


# =============================================================
# METRIC HELPERS
# =============================================================
def pr_lift(y_true, y_score):
    """AUPRC / prevalence."""
    prev = y_true.mean()
    if prev == 0:
        return np.nan
    return average_precision_score(y_true, y_score) / prev


def top_k_recall(y_true, y_score, k_frac=0.10, by_day=None):
    """
    Daily Top-K% recall.
    If by_day is provided, rank within each day separately.
    Otherwise pooled ranking.
    """
    df = pd.DataFrame({"y": y_true, "s": y_score})
    if by_day is not None:
        df["day"] = by_day
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
    """
    Standard (non-episode-onset) Lead-K-day recall.
    For each anomaly cow-day, check if that cow appeared in daily Top-K%
    on any of days t-1, ..., t-k.
    df must have columns: cow_id, day, y, s
    """
    df = df.copy().sort_values(["cow_id", "day"])
    df["day"] = pd.to_datetime(df["day"])
    df["rank"] = df.groupby("day")["s"].rank(ascending=False, method="first")
    df["n_day"] = df.groupby("day")["y"].transform("count")
    df["k_threshold"] = (df["n_day"] * k_frac).clip(lower=1).round()
    df["picked"] = df["rank"] <= df["k_threshold"]

    anomalies = df.loc[df["y"] == 1, ["cow_id", "day"]].copy()
    picked_set = set(
        zip(df.loc[df["picked"], "cow_id"], df.loc[df["picked"], "day"])
    )
    hits = 0
    for _, row in anomalies.iterrows():
        cow = row["cow_id"]
        t = row["day"]
        for lag in range(1, k_days + 1):
            if (cow, t - pd.Timedelta(days=lag)) in picked_set:
                hits += 1
                break
    return hits / len(anomalies) if len(anomalies) > 0 else np.nan


# =============================================================
# MAIN
# =============================================================
def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["day"] = pd.to_datetime(df["day"])

    # Filter to 8 retained farms
    df = df[df["farm_id"].isin(FARM_MAP.keys())].copy()
    print(f"Loaded {len(df):,} rows from {df['farm_id'].nunique()} farms")

    results = []

    for held_out_farm in FARM_MAP.keys():
        print(f"\n{'='*60}")
        print(f"Held-out farm: {held_out_farm} ({FARM_MAP[held_out_farm]})")
        print('='*60)

        test_start, test_end = TEST_WINDOWS[held_out_farm]
        test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)

        # Test set: held-out farm's test window
        test_mask = (
            (df["farm_id"] == held_out_farm)
            & (df["day"] >= test_start)
            & (df["day"] <= test_end)
        )
        df_test = df.loc[test_mask].copy()

        # Training: OTHER farms only, time-anchored to <= cutoff
        train_mask = (
            (df["farm_id"] != held_out_farm)
            & (df["day"] <= TRAINING_CUTOFF)
        )
        df_train = df.loc[train_mask].copy()

        # Validation: small slice from training for early stopping
        # (last 30 days before cutoff, from training farms)
        val_start = TRAINING_CUTOFF - pd.Timedelta(days=30)
        df_val = df_train.loc[df_train["day"] > val_start].copy()
        df_train = df_train.loc[df_train["day"] <= val_start].copy()

        print(f"  train: {len(df_train):,} rows (up to {TRAINING_CUTOFF.date()})")
        print(f"  val:   {len(df_val):,} rows")
        print(f"  test:  {len(df_test):,} rows ({test_start.date()} to {test_end.date()})")
        print(f"  test prevalence: {df_test[LABEL_COL].mean():.4f}")

        # Drop rows with missing features
        X_train = df_train[FEATURES].copy()
        X_val = df_val[FEATURES].copy()
        X_test = df_test[FEATURES].copy()

        y_train = df_train[LABEL_COL].astype(int).values
        y_val = df_val[LABEL_COL].astype(int).values
        y_test = df_test[LABEL_COL].astype(int).values

        train_ds = lgb.Dataset(X_train, label=y_train)
        val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

        model = lgb.train(
            LGBM_PARAMS,
            train_ds,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )

        # Predict
        y_score = model.predict(X_test, num_iteration=model.best_iteration)

        # Metrics
        prev = y_test.mean()
        pr_lift_val = pr_lift(y_test, y_score)
        recall_10 = top_k_recall(y_test, y_score, k_frac=0.10, by_day=df_test["day"].values)
        lift_at_10 = recall_10 / 0.10 if not np.isnan(recall_10) else np.nan

        # Lead-3 recall (standard, not episode-onset — that's E2)
        df_test_pred = df_test[["cow_id", "day"]].copy()
        df_test_pred["y"] = y_test
        df_test_pred["s"] = y_score
        lead3 = lead_k_recall(df_test_pred, k_days=3, k_frac=0.10)

        result = {
            "held_out_farm": held_out_farm,
            "farm_label": FARM_MAP[held_out_farm],
            "n_train": len(df_train),
            "n_val": len(df_val),
            "n_test": len(df_test),
            "prevalence": prev,
            "pr_lift": pr_lift_val,
            "recall_at_10pct": recall_10,
            "lift_at_10pct": lift_at_10,
            "lead_3day_recall": lead3,
            "best_iter": model.best_iteration,
        }
        results.append(result)

        print(f"  PR-Lift          = {pr_lift_val:.3f}")
        print(f"  Recall@10%       = {recall_10:.3f}")
        print(f"  Lift@10%         = {lift_at_10:.3f}")
        print(f"  Lead-3-day recall = {lead3:.3f}")

    # Save
    out = pd.DataFrame(results)
    out_path = OUT_DIR / "E1_time_anchored_loso_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print('='*60)
    print(out.to_string(index=False))
    print(f"\nMean PR-Lift          : {out['pr_lift'].mean():.3f}")
    print(f"Mean Recall@10%       : {out['recall_at_10pct'].mean():.3f}")
    print(f"Mean Lift@10%         : {out['lift_at_10pct'].mean():.3f}")
    print(f"Mean Lead-3-day recall : {out['lead_3day_recall'].mean():.3f}")

    # Compare against retrospective LOSO (if you have it)
    print("\n---")
    print("Compare these to your existing retrospective LOSO results.")
    print("Expected: PR-Lift may drop 10-20%, Lead-3 recall similar.")
    print("If results hold up, deployment narrative strengthens.")
    print("If results drop, write honestly:")
    print("  'deployment-realistic time-anchored LOSO yields lower but still")
    print("   actionable performance compared to retrospective pooled estimation'")


if __name__ == "__main__":
    main()
