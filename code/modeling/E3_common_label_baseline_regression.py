#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import pyarrow.parquet as pq

BASE = Path("${DATA_ROOT}")
DER = BASE / "derived"

DATA_PATH = DER / "unified_daily_clean.parquet"
OUT = DER / "E3_common_label_baseline_regression_summary.csv"

N_THREADS = 85
SEED = 20260514
COW_MIN_N = 30
COHORT_MIN_N = 200

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

CLIMATE = [
    "thi_max",
    "thi_max_lag1",
    "thi_max_lag3_avg",
    "hours_thi_gt_72",
    "hours_thi_gt_79",
    "thi_max_lag7_cum",
    "heatwave_3d",
]

SHARED = CLIMATE + ["DIM", "parity_group", "milk_lag1"]

FEATURES_COHORT = SHARED + ["expected_milk_cohort", "ratio_lag1_cohort"]
FEATURES_COW = SHARED + ["expected_milk_cow", "ratio_lag1_cow"]

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


def read_data():
    cols = [
        "farm_id", "bullNumber", "milkDate", "daily_milk_total",
        "DIM", "inferred_parity",
        "qc_flag",
    ] + CLIMATE

    pf = pq.ParquetFile(DATA_PATH)
    existing = set(pf.schema.names)
    cols = [c for c in cols if c in existing]

    required = [
        "farm_id", "bullNumber", "milkDate", "daily_milk_total",
        "DIM", "inferred_parity",
    ] + CLIMATE

    missing = [c for c in required if c not in existing]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    print("Reading columns:", cols, flush=True)
    df = pd.read_parquet(DATA_PATH, columns=cols)

    df["farm_id"] = df["farm_id"].astype(int)
    df["bullNumber"] = df["bullNumber"].astype(str)
    df["milkDate"] = pd.to_datetime(df["milkDate"])
    df["daily_milk_total"] = pd.to_numeric(df["daily_milk_total"], errors="coerce")
    df["DIM"] = pd.to_numeric(df["DIM"], errors="coerce")
    df["inferred_parity"] = pd.to_numeric(df["inferred_parity"], errors="coerce")

    for c in CLIMATE:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df[df["farm_id"].isin(FARM_MAP.keys())].copy()

    if "qc_flag" in df.columns:
        before_qc = len(df)
        df = df[(df["qc_flag"] == 0) | (df["qc_flag"] == False)].copy()
        print(f"QC filter: {before_qc:,} -> {len(df):,}", flush=True)

    bad_parity = (~np.isfinite(df["inferred_parity"])).sum()
    if bad_parity > 0:
        print(f"Dropping missing inferred_parity: {bad_parity:,}", flush=True)
    df = df[np.isfinite(df["inferred_parity"])].copy()

    df["parity_group"] = df["inferred_parity"].clip(lower=1, upper=4).astype("int16")

    df = df.dropna(subset=[
        "farm_id", "bullNumber", "milkDate", "daily_milk_total",
        "DIM", "parity_group",
    ] + CLIMATE).copy()

    df = df[df["daily_milk_total"] > 0].copy()
    df = df[(df["DIM"] >= 60) & (df["DIM"] <= 330)].copy()
    df["DIM_bin"] = ((df["DIM"] - 60) // 30).astype("int16").clip(0, 8)

    print(f"Loaded usable rows: {len(df):,} from raw {before:,}", flush=True)
    print(df.groupby("farm_id")["milkDate"].agg(["min", "max", "count"]).sort_index(), flush=True)
    return df


def make_cohort_baseline(train):
    b = (
        train.groupby(["parity_group", "DIM_bin"], observed=True)["daily_milk_total"]
        .agg(["median", "count"])
        .reset_index()
    )
    b = b[b["count"] >= COHORT_MIN_N].copy()
    b = b.rename(columns={"median": "expected_milk_cohort"})
    return b[["parity_group", "DIM_bin", "expected_milk_cohort"]]


def make_cow_baseline(train):
    b = (
        train.groupby("bullNumber", observed=True)["daily_milk_total"]
        .agg(["median", "count"])
        .reset_index()
    )
    b = b[b["count"] >= COW_MIN_N].copy()
    b = b.rename(columns={"median": "expected_milk_cow"})
    return b[["bullNumber", "expected_milk_cow"]]


def attach_baselines(x, cohort_b, cow_b):
    out = x.merge(
        cohort_b,
        on=["parity_group", "DIM_bin"],
        how="left",
        sort=False,
    )
    out = out.merge(
        cow_b,
        on="bullNumber",
        how="left",
        sort=False,
    )
    out = out.dropna(subset=["expected_milk_cohort", "expected_milk_cow"]).copy()
    out = out[(out["expected_milk_cohort"] > 0) & (out["expected_milk_cow"] > 0)].copy()

    out["ratio_cohort"] = out["daily_milk_total"] / out["expected_milk_cohort"]
    out["ratio_cow"] = out["daily_milk_total"] / out["expected_milk_cow"]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["ratio_cohort", "ratio_cow"]).copy()
    return out


def add_lag_features(x):
    x = x.sort_values(["farm_id", "bullNumber", "milkDate"]).copy()
    g = x.groupby(["farm_id", "bullNumber"], sort=False)
    prev_day = g["milkDate"].shift(1)
    gap = (x["milkDate"] - prev_day).dt.days

    x["milk_lag1"] = g["daily_milk_total"].shift(1)
    x["ratio_lag1_cohort"] = g["ratio_cohort"].shift(1)
    x["ratio_lag1_cow"] = g["ratio_cow"].shift(1)

    bad = gap != 1
    x.loc[bad, ["milk_lag1", "ratio_lag1_cohort", "ratio_lag1_cow"]] = np.nan
    return x


def thresholds(train, ratio_col):
    th = (
        train.groupby("parity_group", observed=True)[ratio_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    th["threshold"] = th["mean"] - 1.5 * th["std"]
    return th[["parity_group", "threshold"]]


def apply_label(x, th, ratio_col, label_col):
    out = x.merge(th, on="parity_group", how="left", sort=False)
    out[label_col] = (out[ratio_col] < out["threshold"]).astype("int8")
    out = out.drop(columns=["threshold"])
    return out


def pr_lift(y, s):
    prev = float(np.mean(y))
    if prev <= 0 or prev >= 1:
        return np.nan
    return average_precision_score(y, s) / prev


def recall_at_10(y, s, days):
    d = pd.DataFrame({"y": y, "s": s, "day": days})
    d = d.sort_values(["day", "s"], ascending=[True, False]).reset_index(drop=True)
    d["rank"] = d.groupby("day").cumcount() + 1
    d["n_day"] = d.groupby("day")["y"].transform("size")
    d["k"] = np.ceil(d["n_day"] * 0.10).clip(lower=1)
    d["picked"] = d["rank"] <= d["k"]
    n_pos = d["y"].sum()
    if n_pos == 0:
        return np.nan
    return float(d.loc[d["picked"], "y"].sum() / n_pos)


def train_regression(train, val, features, target_col, model_name):
    train = train.dropna(subset=features + [target_col]).copy()
    val = val.dropna(subset=features + [target_col]).copy()

    Xtr = train[features]
    ytr = train[target_col].astype("float32")
    Xva = val[features]
    yva = val[target_col].astype("float32")

    ds_tr = lgb.Dataset(
        Xtr,
        label=ytr,
        categorical_feature=["parity_group"],
        free_raw_data=True,
    )
    ds_va = lgb.Dataset(
        Xva,
        label=yva,
        reference=ds_tr,
        categorical_feature=["parity_group"],
        free_raw_data=True,
    )

    print(f"Training {model_name}: train={len(train):,} val={len(val):,}", flush=True)

    model = lgb.train(
        LGB_PARAMS,
        ds_tr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[ds_va],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return model


def eval_score(test, risk_score, label_col, prefix):
    y = test[label_col].astype(int).to_numpy()
    s = risk_score
    out = {
        f"{prefix}_prev": float(np.mean(y)),
        f"{prefix}_pr_lift": pr_lift(y, s),
        f"{prefix}_recall10": recall_at_10(y, s, test["milkDate"].to_numpy()),
    }
    out[f"{prefix}_lift10"] = out[f"{prefix}_recall10"] / 0.10 if not np.isnan(out[f"{prefix}_recall10"]) else np.nan
    return out


def main():
    print("=" * 80, flush=True)
    print("E3 common-label cohort vs cow baseline sanity check", flush=True)
    print("Regression formulation; fixed-label evaluation", flush=True)
    print(f"threads={N_THREADS} COW_MIN_N={COW_MIN_N} COHORT_MIN_N={COHORT_MIN_N}", flush=True)
    print("=" * 80, flush=True)

    df = read_data()
    rows = []

    for farm_id in FARM_ORDER:
        farm_label = FARM_MAP[farm_id]
        test_start = pd.Timestamp(TEST_WINDOWS[farm_id][0])
        test_end = pd.Timestamp(TEST_WINDOWS[farm_id][1])
        val_start = test_start - pd.Timedelta(days=30)

        print("\n" + "=" * 80, flush=True)
        print(f"{farm_id} {farm_label}", flush=True)
        print(f"Test window: {test_start.date()} to {test_end.date()}", flush=True)

        farm = df[df["farm_id"] == farm_id].copy()

        train_raw = farm[farm["milkDate"] < val_start].copy()
        val_raw = farm[(farm["milkDate"] >= val_start) & (farm["milkDate"] < test_start)].copy()
        test_raw = farm[(farm["milkDate"] >= test_start) & (farm["milkDate"] <= test_end)].copy()

        print(f"Raw split: train={len(train_raw):,} val={len(val_raw):,} test={len(test_raw):,}", flush=True)

        if len(train_raw) < 10000 or len(test_raw) < 1000:
            print("Too few rows, skipping", flush=True)
            continue

        cohort_b = make_cohort_baseline(train_raw)
        cow_b = make_cow_baseline(train_raw)

        print(f"Baseline strata: cohort={len(cohort_b)} cow={len(cow_b):,}", flush=True)

        # Include a short pre-test history so lag1 exists for first test days.
        history_raw = farm[(farm["milkDate"] >= test_start - pd.Timedelta(days=10)) & (farm["milkDate"] < test_start)].copy()
        combined_raw = pd.concat([train_raw, val_raw, history_raw, test_raw], ignore_index=True)

        combined = attach_baselines(combined_raw, cohort_b, cow_b)
        combined = add_lag_features(combined)

        train = combined[combined["milkDate"] < val_start].copy()
        val = combined[(combined["milkDate"] >= val_start) & (combined["milkDate"] < test_start)].copy()
        test = combined[(combined["milkDate"] >= test_start) & (combined["milkDate"] <= test_end)].copy()

        print(f"Common-baseline rows: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)
        print(f"Test retained vs raw: {len(test):,}/{len(test_raw):,} = {len(test)/max(len(test_raw),1):.3f}", flush=True)

        if len(train) < 10000 or len(val) < 1000 or len(test) < 1000:
            print("Too few common-baseline rows, skipping", flush=True)
            continue

        th_cohort = thresholds(train, "ratio_cohort")
        th_cow = thresholds(train, "ratio_cow")

        for xname, x in [("train", train), ("val", val), ("test", test)]:
            pass

        train = apply_label(train, th_cohort, "ratio_cohort", "label_cohort")
        val = apply_label(val, th_cohort, "ratio_cohort", "label_cohort")
        test = apply_label(test, th_cohort, "ratio_cohort", "label_cohort")

        train = apply_label(train, th_cow, "ratio_cow", "label_cow")
        val = apply_label(val, th_cow, "ratio_cow", "label_cow")
        test = apply_label(test, th_cow, "ratio_cow", "label_cow")

        train["log_ratio_cohort"] = np.log(train["ratio_cohort"].clip(0.05, 3.0))
        val["log_ratio_cohort"] = np.log(val["ratio_cohort"].clip(0.05, 3.0))
        test["log_ratio_cohort"] = np.log(test["ratio_cohort"].clip(0.05, 3.0))

        train["log_ratio_cow"] = np.log(train["ratio_cow"].clip(0.05, 3.0))
        val["log_ratio_cow"] = np.log(val["ratio_cow"].clip(0.05, 3.0))
        test["log_ratio_cow"] = np.log(test["ratio_cow"].clip(0.05, 3.0))

        train = train.dropna(subset=FEATURES_COHORT + FEATURES_COW + ["log_ratio_cohort", "log_ratio_cow", "label_cohort", "label_cow"]).copy()
        val = val.dropna(subset=FEATURES_COHORT + FEATURES_COW + ["log_ratio_cohort", "log_ratio_cow", "label_cohort", "label_cow"]).copy()
        test = test.dropna(subset=FEATURES_COHORT + FEATURES_COW + ["label_cohort", "label_cow"]).copy()

        print(f"After feature drop: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)
        print("Cohort thresholds:", th_cohort.to_dict("records"), flush=True)
        print("Cow thresholds:", th_cow.to_dict("records"), flush=True)
        print(f"Test prev label_cohort={test['label_cohort'].mean():.4f} label_cow={test['label_cow'].mean():.4f}", flush=True)

        if len(train) < 10000 or len(val) < 1000 or len(test) < 1000:
            print("Too few rows after feature drop, skipping", flush=True)
            continue

        cohort_model = train_regression(train, val, FEATURES_COHORT, "log_ratio_cohort", "cohort-model")
        cow_model = train_regression(train, val, FEATURES_COW, "log_ratio_cow", "cow-model")

        pred_cohort = np.exp(cohort_model.predict(test[FEATURES_COHORT], num_iteration=cohort_model.best_iteration))
        pred_cow = np.exp(cow_model.predict(test[FEATURES_COW], num_iteration=cow_model.best_iteration))

        risk_cohort = -pred_cohort
        risk_cow = -pred_cow

        row = {
            "farm_id": farm_id,
            "farm_label": farm_label,
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "test_retained_fraction": len(test) / max(len(test_raw), 1),
            "n_cows_cow_baseline": len(cow_b),
            "cohort_best_iter": cohort_model.best_iteration,
            "cow_best_iter": cow_model.best_iteration,
        }

        # Fixed cohort label: compare cohort-risk vs cow-risk
        row.update(eval_score(test, risk_cohort, "label_cohort", "fixed_cohort_label__cohort_model"))
        row.update(eval_score(test, risk_cow, "label_cohort", "fixed_cohort_label__cow_model"))

        # Fixed cow label: compare cohort-risk vs cow-risk
        row.update(eval_score(test, risk_cohort, "label_cow", "fixed_cow_label__cohort_model"))
        row.update(eval_score(test, risk_cow, "label_cow", "fixed_cow_label__cow_model"))

        row["diff_pr_lift_fixed_cohort_label"] = (
            row["fixed_cohort_label__cohort_model_pr_lift"]
            - row["fixed_cohort_label__cow_model_pr_lift"]
        )
        row["diff_recall10_fixed_cohort_label"] = (
            row["fixed_cohort_label__cohort_model_recall10"]
            - row["fixed_cohort_label__cow_model_recall10"]
        )
        row["diff_pr_lift_fixed_cow_label"] = (
            row["fixed_cow_label__cohort_model_pr_lift"]
            - row["fixed_cow_label__cow_model_pr_lift"]
        )
        row["diff_recall10_fixed_cow_label"] = (
            row["fixed_cow_label__cohort_model_recall10"]
            - row["fixed_cow_label__cow_model_recall10"]
        )

        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False)

        print("RESULT", farm_id, farm_label, flush=True)
        print(
            f"  Fixed cohort-label PR-Lift: cohort={row['fixed_cohort_label__cohort_model_pr_lift']:.3f} "
            f"cow={row['fixed_cohort_label__cow_model_pr_lift']:.3f} "
            f"diff={row['diff_pr_lift_fixed_cohort_label']:+.3f}",
            flush=True,
        )
        print(
            f"  Fixed cohort-label Recall@10: cohort={row['fixed_cohort_label__cohort_model_recall10']:.3f} "
            f"cow={row['fixed_cohort_label__cow_model_recall10']:.3f} "
            f"diff={row['diff_recall10_fixed_cohort_label']:+.3f}",
            flush=True,
        )
        print(
            f"  Fixed cow-label PR-Lift: cohort={row['fixed_cow_label__cohort_model_pr_lift']:.3f} "
            f"cow={row['fixed_cow_label__cow_model_pr_lift']:.3f} "
            f"diff={row['diff_pr_lift_fixed_cow_label']:+.3f}",
            flush=True,
        )
        print(
            f"  Fixed cow-label Recall@10: cohort={row['fixed_cow_label__cohort_model_recall10']:.3f} "
            f"cow={row['fixed_cow_label__cow_model_recall10']:.3f} "
            f"diff={row['diff_recall10_fixed_cow_label']:+.3f}",
            flush=True,
        )
        print(f"Partial saved: {OUT}", flush=True)

        del cohort_model, cow_model

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print("\n" + "=" * 80, flush=True)
    print("Final E3 summary", flush=True)
    print(out.to_string(index=False), flush=True)

    if len(out) > 0:
        print("\nMEANS / WINS", flush=True)

        for diff in [
            "diff_pr_lift_fixed_cohort_label",
            "diff_recall10_fixed_cohort_label",
            "diff_pr_lift_fixed_cow_label",
            "diff_recall10_fixed_cow_label",
        ]:
            vals = out[diff]
            print(f"{diff:38s} mean={vals.mean():+.3f} wins={(vals > 0).sum()}/{vals.notna().sum()}", flush=True)

        print("\nInterpretation:", flush=True)
        print("  Positive diff = cohort-baseline formulation beats cow-baseline formulation under the same fixed label.", flush=True)
        print("  If cohort wins mainly under fixed cohort-label but not fixed cow-label, soften mechanism claim.", flush=True)
        print("  If cohort wins under both labels, mechanism claim is much stronger.", flush=True)

    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
