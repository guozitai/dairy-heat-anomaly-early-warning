#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E4: Prospective-feature time-anchored LOSO regression.

Purpose
-------
Defend against the criticism that focal-day THI variables are not truly
available at the time of early-warning deployment.

This script compares two time-anchored LOSO variants:

1. full_same_day_THI
   Uses the original full climate feature set, including focal-day THI:
   - thi_max
   - hours_thi_gt_72
   - hours_thi_gt_79
   - heatwave_3d
   plus lagged/cumulative features.

2. prospective_prior_only
   Removes all focal-day climate variables and uses only exposure information
   derivable before the focal day:
   - prior-day THI
   - prior 3-day THI average
   - prior 1/3/7-day heat-hour summaries
   - prior 3-day heatwave indicator

The model formulation follows E1:
- LightGBM regression predicts log(ratio)
- ratio_pred = exp(pred_log_ratio)
- risk_score = -ratio_pred
- lower predicted ratio = higher risk

Outputs
-------
${DATA_ROOT}/derived/E4_prospective_feature_time_anchored_loso_summary.csv
${DATA_ROOT}/derived/E4_prospective_vs_full_comparison_wide.csv
${DATA_ROOT}/derived/E4_prospective_feature_time_anchored_loso_predictions.parquet
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score


# ============================================================
# 0. Paths and global settings
# ============================================================

BASE = Path("${DATA_ROOT}")
DER = BASE / "derived"

DATA_PATH = DER / "unified_daily_clean.parquet"

OUT_SUMMARY = DER / "E4_prospective_feature_time_anchored_loso_summary.csv"
OUT_WIDE = DER / "E4_prospective_vs_full_comparison_wide.csv"
OUT_PRED = DER / "E4_prospective_feature_time_anchored_loso_predictions.parquet"

N_THREADS = 85
SEED = 20260514
K_FRAC = 0.10
THRESHOLD_SD = 1.5

RUN_FULL_COMPARATOR = True

FARM_MAP = {
    6:  "Farm 1",
    17: "Farm 2",
    16: "Farm 3",
    13: "Farm 4",
    14: "Farm 5",
    15: "Farm 6",
    5:  "Farm 7",
    3:  "Farm 8",
}

FARM_ORDER = [6, 17, 16, 13, 14, 15, 5, 3]

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


# ============================================================
# 1. Feature sets
# ============================================================

FULL_CLIMATE_FEATURES = [
    "thi_max",
    "thi_max_lag1",
    "thi_max_lag3_avg",
    "hours_thi_gt_72",
    "hours_thi_gt_79",
    "thi_max_lag7_cum",
    "heatwave_3d",
]

PROSPECTIVE_CLIMATE_FEATURES = [
    # Fully derived prior-only features.
    # Do not reuse precomputed lag/cumulative columns here.
    "thi_max_lag1_calc",
    "thi_max_lag2_calc",
    "thi_max_lag3_calc",
    "thi_max_lag3_avg_prior_calc",
    "hours_thi_gt72_lag1",
    "hours_thi_gt79_lag1",
    "hours_thi_gt72_lag3_sum",
    "hours_thi_gt79_lag3_sum",
    "hours_thi_gt72_lag7_sum",
    "hours_thi_gt79_lag7_sum",
    "prior_heatwave_3d",
]

BASE_FEATURES = [
    "DIM",
    "parity_group",
    "expected_milk",
    "milk_lag1",
    "ratio_lag1",
]

FEATURE_SETS = {
    "prospective_prior_only": PROSPECTIVE_CLIMATE_FEATURES + BASE_FEATURES,
}

if RUN_FULL_COMPARATOR:
    FEATURE_SETS = {
        "full_same_day_THI": FULL_CLIMATE_FEATURES + BASE_FEATURES,
        "prospective_prior_only": PROSPECTIVE_CLIMATE_FEATURES + BASE_FEATURES,
    }


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


# ============================================================
# 2. Helpers
# ============================================================

def read_main_table():
    needed_cols = [
        "farm_id",
        "bullNumber",
        "milkDate",
        "daily_milk_total",
        "DIM",
        "inferred_parity",
        "qc_flag",

        "thi_max",
        "thi_max_lag1",
        "thi_max_lag3_avg",
        "hours_thi_gt_72",
        "hours_thi_gt_79",
        "thi_max_lag7_cum",
        "heatwave_3d",
    ]

    pf = pq.ParquetFile(DATA_PATH)
    existing = set(pf.schema.names)
    cols = [c for c in needed_cols if c in existing]

    required = [
        "farm_id",
        "bullNumber",
        "milkDate",
        "daily_milk_total",
        "DIM",
        "inferred_parity",
        "thi_max",
        "thi_max_lag1",
        "thi_max_lag3_avg",
        "hours_thi_gt_72",
        "hours_thi_gt_79",
        "thi_max_lag7_cum",
        "heatwave_3d",
    ]
    missing = [c for c in required if c not in existing]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    print("=" * 80, flush=True)
    print("E4 prospective-feature time-anchored LOSO", flush=True)
    print(f"DATA: {DATA_PATH}", flush=True)
    print(f"threads={N_THREADS}", flush=True)
    print("=" * 80, flush=True)
    print("Reading columns:", cols, flush=True)

    df = pd.read_parquet(DATA_PATH, columns=cols)

    df["milkDate"] = pd.to_datetime(df["milkDate"])
    df["farm_id"] = df["farm_id"].astype(int)

    # HARD FILTER: only the 8 retained farms.
    before_farm = len(df)
    df = df[df["farm_id"].isin(FARM_MAP.keys())].copy()
    print(f"Retained-farm filter: {before_farm:,} -> {len(df):,}", flush=True)

    df["bullNumber"] = df["bullNumber"].astype(str)
    df["daily_milk_total"] = pd.to_numeric(df["daily_milk_total"], errors="coerce")
    df["DIM"] = pd.to_numeric(df["DIM"], errors="coerce")
    df["inferred_parity"] = pd.to_numeric(df["inferred_parity"], errors="coerce")

    for c in [
        "thi_max",
        "thi_max_lag1",
        "thi_max_lag3_avg",
        "hours_thi_gt_72",
        "hours_thi_gt_79",
        "thi_max_lag7_cum",
        "heatwave_3d",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "qc_flag" in df.columns:
        before_qc = len(df)
        df = df[(df["qc_flag"] == 0) | (df["qc_flag"] == False)].copy()
        print(f"QC filter: {before_qc:,} -> {len(df):,}", flush=True)

    bad_parity = (~np.isfinite(df["inferred_parity"])).sum()
    if bad_parity > 0:
        print(f"Dropping missing inferred_parity: {bad_parity:,}", flush=True)
    df = df[np.isfinite(df["inferred_parity"])].copy()

    df["parity_group"] = df["inferred_parity"].clip(lower=1, upper=4).astype("int8")

    df = df.dropna(subset=[
        "farm_id",
        "bullNumber",
        "milkDate",
        "daily_milk_total",
        "DIM",
        "parity_group",
        "thi_max",
        "hours_thi_gt_72",
        "hours_thi_gt_79",
    ]).copy()

    df = df[df["daily_milk_total"] > 0].copy()

    print(f"Loaded usable rows: {len(df):,}", flush=True)
    print(df.groupby("farm_id")["milkDate"].agg(["min", "max", "count"]).sort_index(), flush=True)

    return df


def add_prior_climate_features(df):
    print("\nDeriving strict prior-only climate features...", flush=True)

    daily = (
        df[[
            "farm_id",
            "milkDate",
            "thi_max",
            "hours_thi_gt_72",
            "hours_thi_gt_79",
        ]]
        .drop_duplicates(subset=["farm_id", "milkDate"])
        .sort_values(["farm_id", "milkDate"])
        .reset_index(drop=True)
    )

    g = daily.groupby("farm_id", sort=False)

    daily["thi_max_lag1_calc"] = g["thi_max"].shift(1)
    daily["thi_max_lag2_calc"] = g["thi_max"].shift(2)
    daily["thi_max_lag3_calc"] = g["thi_max"].shift(3)

    daily["thi_max_lag3_avg_prior_calc"] = g["thi_max"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
    )

    daily["hours_thi_gt72_lag1"] = g["hours_thi_gt_72"].shift(1)
    daily["hours_thi_gt79_lag1"] = g["hours_thi_gt_79"].shift(1)

    daily["hours_thi_gt72_lag3_sum"] = g["hours_thi_gt_72"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )
    daily["hours_thi_gt79_lag3_sum"] = g["hours_thi_gt_79"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
    )

    daily["hours_thi_gt72_lag7_sum"] = g["hours_thi_gt_72"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )
    daily["hours_thi_gt79_lag7_sum"] = g["hours_thi_gt_79"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).sum()
    )

    daily["prior_heatwave_3d"] = g["thi_max"].transform(
        lambda s: (
            (s.shift(1) > 72)
            .rolling(window=3, min_periods=3)
            .sum()
            .eq(3)
            .astype(float)
        )
    )

    prior_cols = [
        "farm_id",
        "milkDate",
        "thi_max_lag1_calc",
        "thi_max_lag2_calc",
        "thi_max_lag3_calc",
        "thi_max_lag3_avg_prior_calc",
        "hours_thi_gt72_lag1",
        "hours_thi_gt79_lag1",
        "hours_thi_gt72_lag3_sum",
        "hours_thi_gt79_lag3_sum",
        "hours_thi_gt72_lag7_sum",
        "hours_thi_gt79_lag7_sum",
        "prior_heatwave_3d",
    ]

    df = df.merge(
        daily[prior_cols],
        on=["farm_id", "milkDate"],
        how="left",
        sort=False,
    )

    print("Prior-only climate features added.", flush=True)
    return df


def build_expected_from_train(df_train):
    tmp = df_train[(df_train["DIM"] >= 60) & (df_train["DIM"] <= 330)].copy()
    tmp["DIM_bin"] = ((tmp["DIM"] - 60) // 30).astype("int16").clip(0, 8)

    baseline = (
        tmp.groupby(["parity_group", "DIM_bin"], observed=True)["daily_milk_total"]
        .agg(["median", "count"])
        .reset_index()
    )

    baseline = baseline[baseline["count"] >= 200].copy()
    baseline = baseline.rename(columns={"median": "expected_milk"})

    return baseline[["parity_group", "DIM_bin", "expected_milk", "count"]]


def attach_expected(df, baseline):
    x = df[(df["DIM"] >= 60) & (df["DIM"] <= 330)].copy()
    x["DIM_bin"] = ((x["DIM"] - 60) // 30).astype("int16").clip(0, 8)

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

    prev_day = grp["milkDate"].shift(1)
    gap = (df["milkDate"] - prev_day).dt.days

    df.loc[gap != 1, ["milk_lag1", "ratio_lag1"]] = np.nan

    return df


def make_events_from_train_thresholds(df_train, df_apply):
    th = (
        df_train.groupby("parity_group", observed=True)["ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    th["threshold"] = th["mean"] - THRESHOLD_SD * th["std"]

    x = df_apply.merge(
        th[["parity_group", "threshold"]],
        on="parity_group",
        how="left",
        sort=False,
    )

    x = x.dropna(subset=["threshold"]).copy()
    x["event"] = (x["ratio"] < x["threshold"]).astype("int8")

    return x, th[["parity_group", "threshold"]]


def pr_lift(y_true, risk_score):
    y_true = np.asarray(y_true).astype(int)
    risk_score = np.asarray(risk_score)

    prev = float(np.mean(y_true))
    if prev <= 0:
        return np.nan, np.nan

    auprc = average_precision_score(y_true, risk_score)
    return auprc, auprc / prev


def get_daily_topk_picks(df_pred, k_frac=0.10):
    d = df_pred[["cow_id", "day", "event", "risk_score"]].copy()
    d = d.dropna(subset=["risk_score"])
    d = d.sort_values(["day", "risk_score"], ascending=[True, False]).reset_index(drop=True)

    d["rank"] = d.groupby("day").cumcount() + 1
    d["n_day"] = d.groupby("day")["event"].transform("size")
    d["k"] = np.ceil(d["n_day"] * k_frac).clip(lower=1)
    d["picked"] = d["rank"] <= d["k"]

    return d


def daily_topk_metrics(df_pred, k_frac=0.10):
    d = get_daily_topk_picks(df_pred, k_frac=k_frac)

    n_pos = int(d["event"].sum())
    n_pick = int(d["picked"].sum())
    n_hit = int(d.loc[d["picked"], "event"].sum())

    recall = np.nan if n_pos == 0 else n_hit / n_pos
    precision = np.nan if n_pick == 0 else n_hit / n_pick
    lift_at_k = np.nan if pd.isna(recall) else recall / k_frac

    per_day = (
        d[d["picked"]]
        .groupby("day")
        .agg(
            picked_cowdays=("cow_id", "size"),
            picked_unique_cows=("cow_id", "nunique"),
            picked_events=("event", "sum"),
        )
        .reset_index()
    )

    return {
        "recall_at_10": recall,
        "precision_at_10": precision,
        "lift_at_10": lift_at_k,
        "mean_picked_cowdays_per_day": per_day["picked_cowdays"].mean(),
        "mean_picked_unique_cows_per_day": per_day["picked_unique_cows"].mean(),
        "total_events": n_pos,
        "total_picked": n_pick,
        "total_hits": n_hit,
    }


def standard_lead3(df_pred, k_frac=0.10):
    d = get_daily_topk_picks(df_pred, k_frac=k_frac)

    picked = (
        d.loc[d["picked"], ["cow_id", "day"]]
        .drop_duplicates()
        .rename(columns={"day": "lead_day"})
    )
    picked["_picked"] = 1

    ev = (
        d.loc[d["event"] == 1, ["cow_id", "day"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(ev) == 0:
        return np.nan

    ev["_idx"] = np.arange(len(ev))
    hit = np.zeros(len(ev), dtype=bool)

    for lag in [1, 2, 3]:
        tmp = ev[["_idx", "cow_id", "day"]].copy()
        tmp["lead_day"] = tmp["day"] - pd.Timedelta(days=lag)

        m = tmp.merge(
            picked,
            on=["cow_id", "lead_day"],
            how="left",
            sort=False,
        )

        hit |= m["_picked"].fillna(0).to_numpy(dtype=np.int8).astype(bool)

    return float(hit.mean())


def episode_onset_lead_metrics(df_pred, k_frac=0.10):
    d = get_daily_topk_picks(df_pred, k_frac=k_frac)
    d = d.sort_values(["cow_id", "day"]).copy()

    min_day = d["day"].min()

    grp = d.groupby("cow_id", sort=False)
    d["prev_day"] = grp["day"].shift(1)
    d["prev_event"] = grp["event"].shift(1).fillna(0).astype(int)
    d["day_gap"] = (d["day"] - d["prev_day"]).dt.days

    d["is_onset"] = (
        (d["event"] == 1)
        & (
            (d["prev_event"] == 0)
            | (d["day_gap"] != 1)
        )
    )

    onset = d.loc[d["is_onset"], ["cow_id", "day", "picked"]].copy()
    onset = onset.reset_index(drop=True)

    if len(onset) == 0:
        return {
            "onset_n": 0,
            "same_day_onset_top10": np.nan,
            "onset_lead1_all": np.nan,
            "onset_lead2_all": np.nan,
            "onset_lead3_all": np.nan,
            "onset_lead1_eligible": np.nan,
            "onset_lead2_eligible": np.nan,
            "onset_lead3_eligible": np.nan,
        }

    picked = (
        d.loc[d["picked"], ["cow_id", "day"]]
        .drop_duplicates()
        .rename(columns={"day": "lead_day"})
    )
    picked["_picked"] = 1

    out = {
        "onset_n": int(len(onset)),
        "same_day_onset_top10": float(onset["picked"].mean()),
    }

    for k in [1, 2, 3]:
        onset_k = onset[["cow_id", "day"]].copy()
        onset_k["_idx"] = np.arange(len(onset_k))

        hit = np.zeros(len(onset_k), dtype=bool)

        for lag in range(1, k + 1):
            tmp = onset_k[["_idx", "cow_id", "day"]].copy()
            tmp["lead_day"] = tmp["day"] - pd.Timedelta(days=lag)

            m = tmp.merge(
                picked,
                on=["cow_id", "lead_day"],
                how="left",
                sort=False,
            )

            hit |= m["_picked"].fillna(0).to_numpy(dtype=np.int8).astype(bool)

        eligible = onset_k["day"] >= (min_day + pd.Timedelta(days=k))

        out[f"onset_lead{k}_all"] = float(hit.mean())
        out[f"onset_lead{k}_eligible"] = float(hit[eligible].mean()) if eligible.any() else np.nan

    return out


def prepare_fold_data(df_all, held_out_farm):
    test_start, test_end = map(pd.Timestamp, TEST_WINDOWS[held_out_farm])

    if test_start.year == 2025:
        cutoff = pd.Timestamp("2024-12-31")
    elif held_out_farm == 15:
        cutoff = pd.Timestamp("2024-03-31")
    else:
        cutoff = test_start - pd.Timedelta(days=1)

    val_end = cutoff
    val_start = cutoff - pd.Timedelta(days=29)
    train_end = val_start - pd.Timedelta(days=1)

    print("\n" + "=" * 80, flush=True)
    print(f"Held-out: {held_out_farm} {FARM_MAP[held_out_farm]}", flush=True)
    print(f"Test window: {test_start.date()} to {test_end.date()}", flush=True)
    print(f"Train end: {train_end.date()}", flush=True)
    print(f"Val window: {val_start.date()} to {val_end.date()}", flush=True)

    is_other = df_all["farm_id"] != held_out_farm
    is_held = df_all["farm_id"] == held_out_farm

    raw_train = df_all[
        is_other
        & (df_all["milkDate"] <= train_end)
    ].copy()

    raw_val = df_all[
        is_other
        & (df_all["milkDate"] >= val_start)
        & (df_all["milkDate"] <= val_end)
    ].copy()

    raw_test_context = df_all[
        is_held
        & (df_all["milkDate"] >= test_start - pd.Timedelta(days=10))
        & (df_all["milkDate"] <= test_end)
    ].copy()

    print(f"Raw train rows: {len(raw_train):,}", flush=True)
    print(f"Raw val rows:   {len(raw_val):,}", flush=True)
    print(f"Raw test+context rows: {len(raw_test_context):,}", flush=True)

    baseline = build_expected_from_train(raw_train)

    raw_all_fold = pd.concat(
        [
            raw_train.assign(_split="train"),
            raw_val.assign(_split="val"),
            raw_test_context.assign(_split="test_context"),
        ],
        ignore_index=True,
    )

    x = attach_expected(raw_all_fold, baseline)
    x = add_lag_features(x)

    train = x[x["_split"] == "train"].copy()
    val = x[x["_split"] == "val"].copy()
    test = x[
        (x["_split"] == "test_context")
        & (x["milkDate"] >= test_start)
        & (x["milkDate"] <= test_end)
    ].copy()

    train, thresholds = make_events_from_train_thresholds(train, train)
    val, _ = make_events_from_train_thresholds(train, val)
    test, _ = make_events_from_train_thresholds(train, test)

    print(f"Prepared train rows: {len(train):,}", flush=True)
    print(f"Prepared val rows:   {len(val):,}", flush=True)
    print(f"Prepared test rows:  {len(test):,}", flush=True)
    print(f"Test prevalence: {test['event'].mean():.4f}", flush=True)

    return train, val, test, thresholds, cutoff, test_start, test_end


def train_and_predict(train, val, test, features, variant_name):
    train = train.copy()
    val = val.copy()
    test = test.copy()

    eps = 1e-6
    train["target"] = np.log(train["ratio"].clip(lower=eps))
    val["target"] = np.log(val["ratio"].clip(lower=eps))

    for d in [train, val, test]:
        d["parity_group"] = d["parity_group"].astype("category")

    missing_features = [f for f in features if f not in train.columns]
    if missing_features:
        raise ValueError(f"Missing features for {variant_name}: {missing_features}")

    train = train.dropna(subset=["target"]).copy()
    val = val.dropna(subset=["target"]).copy()

    dtrain = lgb.Dataset(
        train[features],
        label=train["target"],
        categorical_feature=["parity_group"],
        free_raw_data=True,
    )

    dval = lgb.Dataset(
        val[features],
        label=val["target"],
        categorical_feature=["parity_group"],
        reference=dtrain,
        free_raw_data=True,
    )

    print(f"\nTraining variant: {variant_name}", flush=True)
    print(f"Features ({len(features)}): {features}", flush=True)
    print(f"Rows train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    pred_log = model.predict(test[features], num_iteration=model.best_iteration)
    ratio_pred = np.exp(pred_log)
    risk_score = -ratio_pred

    pred = pd.DataFrame({
        "variant": variant_name,
        "farm_id": test["farm_id"].astype(int).values,
        "farm": test["farm_id"].map(FARM_MAP).values,
        "cow_id": test["bullNumber"].astype(str).values,
        "day": test["milkDate"].values,
        "parity_group": test["parity_group"].astype(str).values,
        "DIM": test["DIM"].values,
        "daily_milk_total": test["daily_milk_total"].values,
        "expected_milk": test["expected_milk"].values,
        "ratio": test["ratio"].values,
        "ratio_pred": ratio_pred,
        "risk_score": risk_score,
        "event": test["event"].astype(int).values,
    })

    return pred, model.best_iteration


def summarize_predictions(pred, held_out_farm, cutoff, test_start, test_end, best_iter):
    y = pred["event"].astype(int).values
    s = pred["risk_score"].values

    auprc, prlift = pr_lift(y, s)
    topk = daily_topk_metrics(pred, k_frac=K_FRAC)
    lead3 = standard_lead3(pred, k_frac=K_FRAC)
    onset = episode_onset_lead_metrics(pred, k_frac=K_FRAC)

    row = {
        "variant": pred["variant"].iloc[0],
        "farm": FARM_MAP[held_out_farm],
        "held_out_farm": held_out_farm,
        "train_cutoff": cutoff.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
        "n_test": len(pred),
        "prevalence": float(np.mean(y)),
        "AUPRC": auprc,
        "PR_Lift": prlift,
        "Recall_at_10": topk["recall_at_10"],
        "Precision_at_10": topk["precision_at_10"],
        "Lift_at_10": topk["lift_at_10"],
        "Standard_Lead3": lead3,
        "best_iteration": best_iter,
        "mean_picked_cowdays_per_day": topk["mean_picked_cowdays_per_day"],
        "mean_picked_unique_cows_per_day": topk["mean_picked_unique_cows_per_day"],
        "total_events": topk["total_events"],
        "total_picked": topk["total_picked"],
        "total_hits": topk["total_hits"],
    }

    row.update(onset)
    return row


def main():
    df = read_main_table()
    df = add_prior_climate_features(df)

    all_summary = []
    all_predictions = []

    for held_out_farm in FARM_ORDER:
        train, val, test, thresholds, cutoff, test_start, test_end = prepare_fold_data(
            df_all=df,
            held_out_farm=held_out_farm,
        )

        for variant_name, features in FEATURE_SETS.items():
            pred, best_iter = train_and_predict(
                train=train,
                val=val,
                test=test,
                features=features,
                variant_name=variant_name,
            )

            row = summarize_predictions(
                pred=pred,
                held_out_farm=held_out_farm,
                cutoff=cutoff,
                test_start=test_start,
                test_end=test_end,
                best_iter=best_iter,
            )

            all_summary.append(row)
            all_predictions.append(pred)

            pd.DataFrame(all_summary).to_csv(OUT_SUMMARY, index=False)

            print("\nSummary row:", flush=True)
            for k, v in row.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}", flush=True)
                else:
                    print(f"  {k}: {v}", flush=True)
            print(f"Partial summary saved: {OUT_SUMMARY}", flush=True)

    summary = pd.DataFrame(all_summary)
    summary.to_csv(OUT_SUMMARY, index=False)

    if RUN_FULL_COMPARATOR:
        wide = summary.pivot_table(
            index=["farm", "held_out_farm"],
            columns="variant",
            values=[
                "PR_Lift",
                "Recall_at_10",
                "Lift_at_10",
                "Standard_Lead3",
                "onset_lead3_eligible",
            ],
            aggfunc="first",
        )

        wide.columns = [f"{metric}__{variant}" for metric, variant in wide.columns]
        wide = wide.reset_index()

        for metric in [
            "PR_Lift",
            "Recall_at_10",
            "Lift_at_10",
            "Standard_Lead3",
            "onset_lead3_eligible",
        ]:
            full_col = f"{metric}__full_same_day_THI"
            prior_col = f"{metric}__prospective_prior_only"
            if full_col in wide.columns and prior_col in wide.columns:
                wide[f"delta_{metric}_prospective_minus_full"] = wide[prior_col] - wide[full_col]
                wide[f"ratio_{metric}_prospective_over_full"] = wide[prior_col] / wide[full_col]

        wide.to_csv(OUT_WIDE, index=False)
        print(f"\nSaved wide comparison: {OUT_WIDE}", flush=True)

    if all_predictions:
        pred_all = pd.concat(all_predictions, ignore_index=True)
        pred_all.to_parquet(OUT_PRED, index=False)
        print(f"Saved predictions: {OUT_PRED}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"Saved summary: {OUT_SUMMARY}", flush=True)

    mean_cols = [
        "prevalence",
        "AUPRC",
        "PR_Lift",
        "Recall_at_10",
        "Precision_at_10",
        "Lift_at_10",
        "Standard_Lead3",
        "same_day_onset_top10",
        "onset_lead1_eligible",
        "onset_lead2_eligible",
        "onset_lead3_eligible",
    ]

    print("\nMean metrics by variant:", flush=True)
    print(
        summary
        .groupby("variant")[mean_cols]
        .mean(numeric_only=True)
        .round(4),
        flush=True,
    )


if __name__ == "__main__":
    main()
