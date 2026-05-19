#!/usr/bin/env python3
"""
E3: Common-label cohort vs cow baseline sanity check.

Defense against GPT-2 Attack 5 (cohort baseline may "win" because the LABEL changed
along with the baseline, not because the baseline itself is better).

The current decomposition (v3 vs v3.1) probably defines:
  v3:   r_cow = actual / cow_specific_baseline; label = (r_cow < threshold)
  v3.1: r_coh = actual / cohort_baseline;       label = (r_coh < threshold)

If labels differ, we are comparing two DIFFERENT TASKS, not two baselines.
Any "cohort > cow" claim is then about task choice, not about baseline quality.

This experiment:
  1. FIX one label set (recommended: cohort-baseline label) for ALL comparisons
  2. Train two models on this fixed label:
       v3-fixed-label:   uses cow_specific_baseline as a FEATURE
       v3.1-fixed-label: uses cohort_baseline       as a FEATURE
  3. Compare PR-Lift / Recall@10% / Lift@10%
  4. Also run with the OPPOSITE fixed label as robustness check

If v3.1 still wins under fixed label → mechanism claim is supported (cohort baseline
   gives the model a better-calibrated reference value)
If v3.1 only wins under v3.1-label and not under v3-label → the "win" was driven by
   label-task switching, not baseline merit
If results are mixed → soften mechanism claim to "label + baseline jointly matter"

USER ACTION REQUIRED:
1. Update DATA_PATH below
2. Ensure df contains BOTH baselines as columns: expected_milk_cow, expected_milk_cohort
3. Run on JupyterLab server
4. Output: /home/2026gzt/derived/E3_common_label_sanity.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import average_precision_score

# =============================================================
# CONFIG
# =============================================================
DATA_PATH = "/home/2026gzt/derived/unified_daily_clean.parquet"
OUT_DIR = Path("/home/2026gzt/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FARM_MAP = {
    "FARM_1": "Farm 1", "FARM_2": "Farm 2", "FARM_3": "Farm 3", "FARM_4": "Farm 4",
    "FARM_5": "Farm 5", "FARM_6": "Farm 6", "FARM_7": "Farm 7", "FARM_8": "Farm 8",
}

# Threshold for r < t → anomaly
THRESHOLD = 0.85

# Test windows per farm
TEST_WINDOWS = {
    "FARM_1": ("2025-04-01", "2025-09-30"),
    "FARM_2": ("2025-04-01", "2025-09-30"),
    "FARM_3": ("2025-04-01", "2025-09-30"),
    "FARM_4": ("2025-04-01", "2025-09-30"),
    "FARM_5": ("2025-04-01", "2025-09-30"),
    "FARM_6": ("2024-08-12", "2025-09-30"),
    "FARM_7": ("2025-04-01", "2025-09-30"),
    "FARM_8": ("2025-04-01", "2025-09-30"),
}

# Shared (non-baseline) features
SHARED_FEATURES = [
    "thi_max", "thi_max_lag1", "thi_max_lag3_avg",
    "hours_thi_gt_72", "hours_thi_gt_79", "thi_max_lag7_cum", "heatwave_3d",
    "dim", "parity_group",
    "milk_lag1", "ratio_lag1",
]
COW_BASELINE_COL = "expected_milk_cow"
COHORT_BASELINE_COL = "expected_milk_cohort"

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
# HELPERS
# =============================================================
def pr_lift(y_true, y_score):
    prev = y_true.mean()
    if prev == 0:
        return np.nan
    return average_precision_score(y_true, y_score) / prev


def top_k_recall_by_day(y_true, y_score, days, k_frac=0.10):
    df = pd.DataFrame({"y": y_true, "s": y_score, "day": days})
    df["rank"] = df.groupby("day")["s"].rank(ascending=False, method="first")
    df["n_day"] = df.groupby("day")["y"].transform("count")
    df["k_threshold"] = (df["n_day"] * k_frac).clip(lower=1).round()
    df["picked"] = df["rank"] <= df["k_threshold"]
    n_pos = df["y"].sum()
    if n_pos == 0:
        return np.nan
    return df.loc[df["picked"], "y"].sum() / n_pos


def make_label(actual, baseline, threshold=THRESHOLD):
    """Anomaly label: r = actual / baseline < threshold."""
    r = actual / baseline
    return (r < threshold).astype(int)


def train_eval(df_train, df_val, df_test, features, label_col):
    X_tr, X_va, X_te = df_train[features], df_val[features], df_test[features]
    y_tr, y_va, y_te = df_train[label_col].values, df_val[label_col].values, df_test[label_col].values
    tr_ds = lgb.Dataset(X_tr, label=y_tr)
    va_ds = lgb.Dataset(X_va, label=y_va, reference=tr_ds)
    model = lgb.train(
        LGBM_PARAMS, tr_ds,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[va_ds],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    s = model.predict(X_te, num_iteration=model.best_iteration)
    return {
        "pr_lift": pr_lift(y_te, s),
        "recall_10": top_k_recall_by_day(y_te, s, df_test["day"].values, 0.10),
        "best_iter": model.best_iteration,
        "test_prev": y_te.mean(),
    }


# =============================================================
# MAIN
# =============================================================
def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df["day"] = pd.to_datetime(df["day"])
    df = df[df["farm_id"].isin(FARM_MAP.keys())].copy()
    print(f"Loaded {len(df):,} rows from {df['farm_id'].nunique()} farms")

    # Compute both label variants up front
    if "actual_milk" not in df.columns:
        raise ValueError("Need 'actual_milk' column for label computation")
    df["label_cow_baseline"] = make_label(df["actual_milk"], df[COW_BASELINE_COL])
    df["label_cohort_baseline"] = make_label(df["actual_milk"], df[COHORT_BASELINE_COL])

    print(f"  label_cow_baseline prev:    {df['label_cow_baseline'].mean():.4f}")
    print(f"  label_cohort_baseline prev: {df['label_cohort_baseline'].mean():.4f}")

    # Single-farm in-farm comparison (this is where the v3 vs v3.1 result was originally
    # reported, so it's the most relevant comparison to validate)
    results = []
    for farm_id in FARM_MAP.keys():
        print(f"\n{farm_id} ({FARM_MAP[farm_id]})")
        test_start, test_end = pd.Timestamp(TEST_WINDOWS[farm_id][0]), pd.Timestamp(TEST_WINDOWS[farm_id][1])
        df_farm = df.loc[df["farm_id"] == farm_id].copy()

        test_mask = (df_farm["day"] >= test_start) & (df_farm["day"] <= test_end)
        df_test = df_farm.loc[test_mask].copy()
        df_train_all = df_farm.loc[~test_mask].copy()

        # Val: last 30 days before test_start
        val_start = test_start - pd.Timedelta(days=30)
        df_val = df_train_all.loc[df_train_all["day"] >= val_start].copy()
        df_train = df_train_all.loc[df_train_all["day"] < val_start].copy()

        if len(df_train) < 5000 or len(df_test) < 1000:
            print(f"  too few rows (train={len(df_train)}, test={len(df_test)}), skipping")
            continue

        features_cow = SHARED_FEATURES + [COW_BASELINE_COL]
        features_cohort = SHARED_FEATURES + [COHORT_BASELINE_COL]

        row = {"farm_id": farm_id, "farm_label": FARM_MAP[farm_id]}

        # ---- Common label A: cohort-baseline label (the one used in v3.1 paper)
        for fset_name, fset in [("cow_features", features_cow), ("cohort_features", features_cohort)]:
            res = train_eval(df_train, df_val, df_test, fset, "label_cohort_baseline")
            row[f"label_cohort__{fset_name}__pr_lift"] = res["pr_lift"]
            row[f"label_cohort__{fset_name}__recall10"] = res["recall_10"]
            row[f"label_cohort__{fset_name}__lift10"] = (
                res["recall_10"] / 0.10 if not np.isnan(res["recall_10"]) else np.nan
            )

        # ---- Common label B: cow-baseline label (the one v3 originally used)
        for fset_name, fset in [("cow_features", features_cow), ("cohort_features", features_cohort)]:
            res = train_eval(df_train, df_val, df_test, fset, "label_cow_baseline")
            row[f"label_cow__{fset_name}__pr_lift"] = res["pr_lift"]
            row[f"label_cow__{fset_name}__recall10"] = res["recall_10"]
            row[f"label_cow__{fset_name}__lift10"] = (
                res["recall_10"] / 0.10 if not np.isnan(res["recall_10"]) else np.nan
            )

        # Print summary
        print(f"  Label = cohort-baseline (v3.1's label):")
        print(f"    cow-features    PR-Lift = {row['label_cohort__cow_features__pr_lift']:.3f}")
        print(f"    cohort-features PR-Lift = {row['label_cohort__cohort_features__pr_lift']:.3f}")
        print(f"  Label = cow-baseline (v3's label):")
        print(f"    cow-features    PR-Lift = {row['label_cow__cow_features__pr_lift']:.3f}")
        print(f"    cohort-features PR-Lift = {row['label_cow__cohort_features__pr_lift']:.3f}")

        results.append(row)

    out = pd.DataFrame(results)
    out_path = OUT_DIR / "E3_common_label_sanity.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved: {out_path}")
    print('='*70)
    print(out.to_string(index=False))

    print()
    print("INTERPRETATION:")
    print("  For each label fixed, compare cow-features vs cohort-features performance.")
    print()
    print("  Under FIXED cohort-baseline label:")
    cohort_label_diff = out["label_cohort__cohort_features__pr_lift"] - out["label_cohort__cow_features__pr_lift"]
    print(f"    PR-Lift diff (cohort-features minus cow-features):")
    print(f"      mean = {cohort_label_diff.mean():+.3f}")
    print(f"      farms where cohort-features wins: {(cohort_label_diff > 0).sum()}/{len(cohort_label_diff)}")
    print()
    print("  Under FIXED cow-baseline label:")
    cow_label_diff = out["label_cow__cohort_features__pr_lift"] - out["label_cow__cow_features__pr_lift"]
    print(f"    PR-Lift diff (cohort-features minus cow-features):")
    print(f"      mean = {cow_label_diff.mean():+.3f}")
    print(f"      farms where cohort-features wins: {(cow_label_diff > 0).sum()}/{len(cow_label_diff)}")
    print()
    print("DECISION:")
    print("  If cohort-features wins under BOTH label fixings (>=6/8 farms in each):")
    print("    → mechanism claim (cohort baseline as better reference) is supported")
    print("  If cohort-features only wins when its own label is used:")
    print("    → the original v3 vs v3.1 win was a task-redefinition effect, not")
    print("      a baseline merit effect. Soften §4.2 accordingly.")


if __name__ == "__main__":
    main()
