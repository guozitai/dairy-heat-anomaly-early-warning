#!/usr/bin/env python3
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path("${DATA_ROOT}")
DER = BASE / "derived"

DATA_PATH = DER / "unified_daily_clean.parquet"

OUT_SUMMARY = DER / "E1_time_anchored_loso_regression_summary.csv"
OUT_PRED = DER / "E1_time_anchored_loso_regression_predictions.parquet"

N_THREADS = 85
SEED = 20260514

FARM_MAP = {
    6: "Farm 1",
    17: "Farm 2",
    16: "Farm 3",
    13: "Farm 4",
    14: "Farm 5",
    15: "Farm 6",
    5: "Farm 7",
    3: "Farm 8",
}

TEST_WINDOWS = {
    6:  ("2025-04-01", "2025-09-30"),
    17: ("2025-04-01", "2025-09-30"),
    16: ("2025-04-01", "2025-09-30"),
    13: ("2025-04-01", "2025-09-30"),
    14: ("2025-04-01", "2025-09-30"),
    15: ("2024-04-01", "2024-09-30"),
    5:  ("2025-04-01", "2025-09-30"),
    3:  ("2025-04-01", "2025-09-30"),
}

CLIMATE_FEATURES = [
    "thi_max",
    "thi_max_lag1",
    "thi_max_lag3_avg",
    "hours_thi_gt_72",
    "hours_thi_gt_79",
    "thi_max_lag7_cum",
    "heatwave_3d",
]

FEATURES = [
    "thi_max",
    "thi_max_lag1",
    "thi_max_lag3_avg",
    "hours_thi_gt_72",
    "hours_thi_gt_79",
    "thi_max_lag7_cum",
    "heatwave_3d",
    "DIM",
    "parity_group",
    "expected_milk",
    "milk_lag1",
    "ratio_lag1",
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "verbose": -1,
    "num_threads": N_THREADS,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
}

NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 200


def pick_existing_columns(path):
    pf = pq.ParquetFile(path)
    cols = pf.schema.names

    required = [
        "farm_id", "bullNumber", "milkDate", "daily_milk_total",
        "DIM", "inferred_parity",
    ] + CLIMATE_FEATURES

    # qc_flag is optional but preferred if present
    if "qc_flag" in cols and "qc_flag" not in required:
        required.append("qc_flag")

    missing = [c for c in required if c not in cols]
    if missing:
        print("\nERROR: missing required columns in unified_daily_clean.parquet:")
        print(missing)
        print("\nExisting columns:")
        print(cols)
        raise SystemExit(1)

    return required


def add_dim_bin(df):
    # manuscript bins DIM 60–330 into 9 bins of 30 days
    df = df.copy()
    df = df[(df["DIM"] >= 60) & (df["DIM"] <= 330)].copy()
    df["DIM_bin"] = ((df["DIM"] - 60) // 30).astype("int16")
    df["DIM_bin"] = df["DIM_bin"].clip(0, 8)
    return df


def build_expected_from_train(df_train):
    tmp = add_dim_bin(df_train)
    g = (
        tmp.groupby(["parity_group", "DIM_bin"], observed=True)["daily_milk_total"]
        .agg(["median", "count"])
        .reset_index()
    )
    g = g[g["count"] >= 200].copy()
    g = g.rename(columns={"median": "expected_milk"})
    return g[["parity_group", "DIM_bin", "expected_milk", "count"]]


def attach_expected(df, baseline):
    x = add_dim_bin(df)
    x = x.merge(
        baseline[["parity_group", "DIM_bin", "expected_milk"]],
        on=["parity_group", "DIM_bin"],
        how="left",
        sort=False,
    )
    x = x.dropna(subset=["expected_milk"]).copy()
    x = x[x["expected_milk"] > 0].copy()
    x["ratio"] = x["daily_milk_total"] / x["expected_milk"]
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.dropna(subset=["ratio"])
    return x


def add_lag_features(df):
    df = df.sort_values(["farm_id", "bullNumber", "milkDate"]).copy()
    grp = df.groupby(["farm_id", "bullNumber"], sort=False)
    df["milk_lag1"] = grp["daily_milk_total"].shift(1)
    df["ratio_lag1"] = grp["ratio"].shift(1)

    # only keep lag if previous record is exactly one day earlier
    prev_day = grp["milkDate"].shift(1)
    gap = (df["milkDate"] - prev_day).dt.days
    df.loc[gap != 1, ["milk_lag1", "ratio_lag1"]] = np.nan

    return df


def make_events_from_train_thresholds(df_train, df_apply):
    # parity-specific threshold = mean - 1.5 sd on training ratio distribution
    th = (
        df_train.groupby("parity_group", observed=True)["ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    th["threshold"] = th["mean"] - 1.5 * th["std"]
    th = th[["parity_group", "threshold"]]

    x = df_apply.merge(th, on="parity_group", how="left", sort=False)
    x = x.dropna(subset=["threshold"]).copy()
    x["event"] = (x["ratio"] < x["threshold"]).astype("int8")
    return x, th


def pr_lift(y_true, risk_score):
    prev = float(np.mean(y_true))
    if prev <= 0:
        return np.nan
    return average_precision_score(y_true, risk_score) / prev


def daily_recall_at_k(y_true, risk_score, days, k_frac=0.10):
    d = pd.DataFrame({"y": y_true, "s": risk_score, "day": days})
    d = d.sort_values(["day", "s"], ascending=[True, False]).reset_index(drop=True)
    d["rank"] = d.groupby("day").cumcount() + 1
    d["n_day"] = d.groupby("day")["y"].transform("size")
    d["k"] = np.ceil(d["n_day"] * k_frac).clip(lower=1)
    d["picked"] = d["rank"] <= d["k"]
    n_pos = d["y"].sum()
    if n_pos == 0:
        return np.nan
    return d.loc[d["picked"], "y"].sum() / n_pos


def standard_lead3(df_pred, k_frac=0.10):
    d = df_pred[["cow_id", "day", "event", "risk_score"]].copy()
    d = d.sort_values(["day", "risk_score"], ascending=[True, False]).reset_index(drop=True)
    d["rank"] = d.groupby("day").cumcount() + 1
    d["n_day"] = d.groupby("day")["event"].transform("size")
    d["k"] = np.ceil(d["n_day"] * k_frac).clip(lower=1)
    d["picked"] = d["rank"] <= d["k"]

    picked = d.loc[d["picked"], ["cow_id", "day"]].drop_duplicates()
    picked = picked.rename(columns={"day": "lead_day"})
    picked["_picked"] = 1

    ev = d.loc[d["event"] == 1, ["cow_id", "day"]].drop_duplicates().reset_index(drop=True)
    if len(ev) == 0:
        return np.nan

    ev["_idx"] = np.arange(len(ev))
    hit = np.zeros(len(ev), dtype=bool)

    for lag in [1, 2, 3]:
        tmp = ev[["_idx", "cow_id", "day"]].copy()
        tmp["lead_day"] = tmp["day"] - pd.Timedelta(days=lag)
        m = tmp[["_idx", "cow_id", "lead_day"]].merge(
            picked, on=["cow_id", "lead_day"], how="left", sort=False
        )
        hit |= m["_picked"].fillna(0).to_numpy(dtype=np.int8).astype(bool)

    return float(hit.mean())


def main():
    print("=" * 80, flush=True)
    print("E1 time-anchored LOSO regression", flush=True)
    print("DATA:", DATA_PATH, flush=True)
    print("threads:", N_THREADS, flush=True)
    print("=" * 80, flush=True)

    cols = pick_existing_columns(DATA_PATH)
    print("Reading columns:", cols, flush=True)

    df = pd.read_parquet(DATA_PATH, columns=cols)
    df["milkDate"] = pd.to_datetime(df["milkDate"])
    df["farm_id"] = df["farm_id"].astype(int)
    df["bullNumber"] = df["bullNumber"].astype(str)
    # Manuscript parity_group ∈ {1, 2, 3, 4+}; source table has inferred_parity
    df["inferred_parity"] = pd.to_numeric(df["inferred_parity"], errors="coerce")
    bad_parity = (~np.isfinite(df["inferred_parity"])).sum()
    if bad_parity > 0:
        print(f"Dropping rows with missing inferred_parity: {bad_parity:,}", flush=True)
    df = df[np.isfinite(df["inferred_parity"])].copy()
    df["parity_group"] = df["inferred_parity"].clip(lower=1, upper=4).astype("int16")

    df["DIM"] = pd.to_numeric(df["DIM"], errors="coerce")
    df["daily_milk_total"] = pd.to_numeric(df["daily_milk_total"], errors="coerce")

    for c in CLIMATE_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["farm_id"].isin(FARM_MAP.keys())].copy()

    # Use quality-controlled records only if qc_flag is available.
    # Accept both boolean and 0/1 forms.
    if "qc_flag" in df.columns:
        before_qc = len(df)
        df = df[(df["qc_flag"] == 0) | (df["qc_flag"] == False)].copy()
        print(f"QC filter: {before_qc:,} -> {len(df):,} rows", flush=True)

    df = df.dropna(subset=["milkDate", "bullNumber", "daily_milk_total", "DIM", "parity_group"])
    df = df[df["daily_milk_total"] > 0].copy()

    print(f"Loaded rows: {len(df):,}", flush=True)
    print(df.groupby("farm_id")["milkDate"].agg(["min", "max", "count"]).sort_index(), flush=True)

    all_summaries = []
    pred_tables = []

    for held_out in FARM_MAP.keys():
        farm_label = FARM_MAP[held_out]
        test_start = pd.Timestamp(TEST_WINDOWS[held_out][0])
        test_end = pd.Timestamp(TEST_WINDOWS[held_out][1])

        # Strict time anchor: other-farm training cannot use data after deployment start.
        # For 2025 tests, cutoff is 2024-12-31; for Farm 6 2024 test, cutoff is 2024-03-31.
        cutoff = min(pd.Timestamp("2024-12-31"), test_start - pd.Timedelta(days=1))
        val_start = cutoff - pd.Timedelta(days=30)

        print("\n" + "=" * 80, flush=True)
        print(f"Held-out {held_out} {farm_label}", flush=True)
        print(f"Test window: {test_start.date()} to {test_end.date()}", flush=True)
        print(f"Training cutoff for other farms: {cutoff.date()}", flush=True)

        # Baseline and model training data: other farms only, <= cutoff
        other_time = df[(df["farm_id"] != held_out) & (df["milkDate"] <= cutoff)].copy()
        if len(other_time) == 0:
            print("No training rows; skipping", flush=True)
            continue

        baseline_train_raw = other_time[other_time["milkDate"] <= val_start].copy()
        baseline = build_expected_from_train(baseline_train_raw)
        print(f"Baseline strata: {len(baseline)}", flush=True)

        # Need held-out history up to test_end so lag1 on test days can be computed
        held_history = df[
            (df["farm_id"] == held_out)
            & (df["milkDate"] <= test_end)
            & (df["milkDate"] >= test_start - pd.Timedelta(days=10))
        ].copy()

        # Fold frame: other train/val + held history/test
        fold_raw = pd.concat([other_time, held_history], ignore_index=True)

        fold = attach_expected(fold_raw, baseline)
        fold = add_lag_features(fold)

        train_base = fold[
            (fold["farm_id"] != held_out)
            & (fold["milkDate"] <= val_start)
        ].copy()
        val_base = fold[
            (fold["farm_id"] != held_out)
            & (fold["milkDate"] > val_start)
            & (fold["milkDate"] <= cutoff)
        ].copy()
        test_base = fold[
            (fold["farm_id"] == held_out)
            & (fold["milkDate"] >= test_start)
            & (fold["milkDate"] <= test_end)
        ].copy()

        # Create event labels using train thresholds; apply same thresholds to val/test
        train_labeled, thresholds = make_events_from_train_thresholds(train_base, train_base)
        val_labeled, _ = make_events_from_train_thresholds(train_base, val_base)
        test_labeled, _ = make_events_from_train_thresholds(train_base, test_base)

        # log-ratio target for regression
        for x in [train_labeled, val_labeled, test_labeled]:
            x["ratio_clip"] = x["ratio"].clip(0.05, 3.0)
            x["log_ratio"] = np.log(x["ratio_clip"])

        # Drop rows with missing required features/target
        train_labeled = train_labeled.dropna(subset=FEATURES + ["log_ratio", "event"]).copy()
        val_labeled = val_labeled.dropna(subset=FEATURES + ["log_ratio", "event"]).copy()
        test_labeled = test_labeled.dropna(subset=FEATURES + ["event"]).copy()

        print(f"Rows train={len(train_labeled):,} val={len(val_labeled):,} test={len(test_labeled):,}", flush=True)
        print(f"Prevalence train={train_labeled['event'].mean():.4f} val={val_labeled['event'].mean():.4f} test={test_labeled['event'].mean():.4f}", flush=True)
        print("Thresholds by parity:", thresholds.to_dict("records"), flush=True)

        if len(train_labeled) < 10000 or len(val_labeled) < 1000 or len(test_labeled) < 1000:
            print("Too few rows; skipping", flush=True)
            continue

        X_train = train_labeled[FEATURES]
        y_train = train_labeled["log_ratio"].astype("float32")
        X_val = val_labeled[FEATURES]
        y_val = val_labeled["log_ratio"].astype("float32")
        X_test = test_labeled[FEATURES]

        train_ds = lgb.Dataset(
            X_train,
            label=y_train,
            categorical_feature=["parity_group"],
            free_raw_data=True,
        )
        val_ds = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_ds,
            categorical_feature=["parity_group"],
            free_raw_data=True,
        )

        model = lgb.train(
            LGB_PARAMS,
            train_ds,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_ds],
            valid_names=["val"],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        pred_log_ratio = model.predict(X_test, num_iteration=model.best_iteration)
        ratio_pred = np.exp(pred_log_ratio)
        risk_score = -ratio_pred

        y_true = test_labeled["event"].astype(int).to_numpy()
        prev = float(y_true.mean())
        pr_lift_val = pr_lift(y_true, risk_score)
        recall10 = daily_recall_at_k(
            y_true,
            risk_score,
            test_labeled["milkDate"].to_numpy(),
            k_frac=0.10,
        )
        lift10 = recall10 / 0.10 if not np.isnan(recall10) else np.nan

        pred_out = pd.DataFrame({
            "held_out_farm": held_out,
            "farm_id": test_labeled["farm_id"].astype(int).to_numpy(),
            "farm_label": farm_label,
            "cow_id": test_labeled["bullNumber"].astype(str).to_numpy(),
            "day": test_labeled["milkDate"].to_numpy(),
            "parity_group": test_labeled["parity_group"].astype(int).to_numpy(),
            "DIM": test_labeled["DIM"].to_numpy(),
            "daily_milk_total": test_labeled["daily_milk_total"].to_numpy(),
            "expected_milk": test_labeled["expected_milk"].to_numpy(),
            "ratio": test_labeled["ratio"].to_numpy(),
            "ratio_pred": ratio_pred,
            "risk_score": risk_score,
            "event": y_true,
        })

        lead3 = standard_lead3(pred_out, k_frac=0.10)

        summary = {
            "held_out_farm": held_out,
            "farm_label": farm_label,
            "train_cutoff": str(cutoff.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_train": len(train_labeled),
            "n_val": len(val_labeled),
            "n_test": len(test_labeled),
            "prevalence": prev,
            "pr_lift": pr_lift_val,
            "recall_at_10pct": recall10,
            "lift_at_10pct": lift10,
            "standard_lead3": lead3,
            "best_iter": model.best_iteration,
        }

        all_summaries.append(summary)
        pred_tables.append(pred_out)

        pd.DataFrame(all_summaries).to_csv(OUT_SUMMARY, index=False)

        print(f"RESULT {held_out} {farm_label}", flush=True)
        print(f"  PR-Lift      = {pr_lift_val:.3f}", flush=True)
        print(f"  Recall@10%   = {recall10:.3f}", flush=True)
        print(f"  Lift@10%     = {lift10:.3f}", flush=True)
        print(f"  Std Lead-3   = {lead3:.3f}", flush=True)
        print(f"  best_iter    = {model.best_iteration}", flush=True)
        print(f"Partial summary saved: {OUT_SUMMARY}", flush=True)

        # free memory
        del train_ds, val_ds, model
        del train_labeled, val_labeled, test_labeled, train_base, val_base, test_base, fold, fold_raw

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\n" + "=" * 80, flush=True)
    print("Final summary", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print("\nMEANS:", flush=True)
    for c in ["pr_lift", "recall_at_10pct", "lift_at_10pct", "standard_lead3"]:
        print(f"{c:20s} {summary_df[c].mean():.3f}", flush=True)

    if pred_tables:
        pred_df = pd.concat(pred_tables, ignore_index=True)
        pred_df.to_parquet(OUT_PRED, index=False)
        print(f"\nSaved predictions: {OUT_PRED}", flush=True)

    print(f"Saved summary: {OUT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
