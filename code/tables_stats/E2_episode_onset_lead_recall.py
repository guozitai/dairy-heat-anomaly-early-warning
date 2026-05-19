#!/usr/bin/env python3
"""
E2: Episode-onset Lead-K-day recall.

Defense against GPT-2 Attack 4 (Lead-3 may capture persistence, not early warning):
Standard Lead-3 says "anomaly day t had the cow in Top-K% on any of t-1/t-2/t-3."
But within a multi-day anomaly episode, days 2/3/4/... of the episode trivially
satisfy Lead-3 because day 1 (the cow already low-yield) was already flagged.
This conflates "persistence" with "early warning."

Episode-onset Lead-K only evaluates the FIRST day of each anomaly episode:
- Group consecutive anomaly days for a cow into episodes
- For each episode's first day, check if that cow was in daily Top-K%
  during days t-1, t-2, t-3 (lag 1/2/3)
- Report Lead-1 / Lead-2 / Lead-3 recall

This isolates true early warning from persistence.

USES EXISTING PREDICTIONS — no retraining needed.

USER ACTION REQUIRED:
1. Update PRED_DIR to point to your existing LOSO predictions (one CSV per fold
   or one combined CSV with farm_id column)
2. Each prediction file must contain: farm_id, cow_id, day, y_true, y_score
3. Run on JupyterLab server
4. Output: /home/2026gzt/derived/E2_episode_onset_lead_recall.csv

Recommended: also run on time-anchored predictions from E1 once available.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================
# CONFIG — ADJUST
# =============================================================
# Either a single combined predictions CSV with farm_id column,
# OR per-fold CSVs in PRED_DIR
PRED_FILE = "/home/2026gzt/derived/loso_predictions_v3_1.csv"
# Expected columns: farm_id, cow_id, day, y_true (0/1), y_score

OUT_DIR = Path("/home/2026gzt/derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_FRAC = 0.10  # daily Top-K% budget

FARM_MAP = {
    "FARM_1": "Farm 1", "FARM_2": "Farm 2", "FARM_3": "Farm 3", "FARM_4": "Farm 4",
    "FARM_5": "Farm 5", "FARM_6": "Farm 6", "FARM_7": "Farm 7", "FARM_8": "Farm 8",
}


# =============================================================
# CORE LOGIC
# =============================================================
def identify_episodes(df_cow):
    """
    For a single cow's time-sorted anomaly days, group consecutive days
    into episodes. Returns a list of (episode_start_day, episode_length).
    """
    df_cow = df_cow.sort_values("day")
    anomaly_days = df_cow.loc[df_cow["y_true"] == 1, "day"].tolist()
    if not anomaly_days:
        return []

    episodes = []
    ep_start = anomaly_days[0]
    ep_prev = ep_start

    for d in anomaly_days[1:]:
        if (d - ep_prev).days == 1:
            # consecutive — extend current episode
            ep_prev = d
        else:
            # gap — close current episode, start new
            episodes.append((ep_start, (ep_prev - ep_start).days + 1))
            ep_start = d
            ep_prev = d
    episodes.append((ep_start, (ep_prev - ep_start).days + 1))
    return episodes


def episode_onset_lead_recall(df_farm, k_frac=0.10, max_lead=3):
    """
    For one farm's test predictions, compute episode-onset Lead-1/2/3 recall.
    df_farm columns: cow_id, day (datetime), y_true, y_score
    """
    df_farm = df_farm.copy()
    df_farm["day"] = pd.to_datetime(df_farm["day"])

    # Rank within each day to identify Top-K% picks
    df_farm["rank"] = df_farm.groupby("day")["y_score"].rank(ascending=False, method="first")
    df_farm["n_day"] = df_farm.groupby("day")["y_true"].transform("count")
    df_farm["k_threshold"] = (df_farm["n_day"] * k_frac).clip(lower=1).round()
    df_farm["picked"] = df_farm["rank"] <= df_farm["k_threshold"]

    picked_set = set(zip(df_farm.loc[df_farm["picked"], "cow_id"],
                         df_farm.loc[df_farm["picked"], "day"]))

    # Identify episodes per cow
    onsets = []
    for cow_id, df_cow in df_farm.groupby("cow_id"):
        for ep_start, ep_len in identify_episodes(df_cow):
            onsets.append({"cow_id": cow_id, "onset": ep_start, "len": ep_len})

    if not onsets:
        return {
            "n_episodes": 0,
            "mean_episode_length": np.nan,
            "lead_1": np.nan, "lead_2": np.nan, "lead_3": np.nan,
            "onset_in_top_k_same_day": np.nan,
        }

    onsets_df = pd.DataFrame(onsets)

    # Lead-K hits
    results = {"lead_1": 0, "lead_2": 0, "lead_3": 0}
    same_day_hits = 0
    for _, row in onsets_df.iterrows():
        cow = row["cow_id"]
        t = row["onset"]
        # same-day check (sanity)
        if (cow, t) in picked_set:
            same_day_hits += 1
        # lead-K: cow picked on day t-k
        for lead in range(1, max_lead + 1):
            if (cow, t - pd.Timedelta(days=lead)) in picked_set:
                # cumulative: if lead-1 hit, also lead-2, lead-3 hit
                # (because lead-K = picked on AT LEAST ONE day in t-1..t-K)
                pass
        # cleaner cumulative computation:
        for lead_target in range(1, max_lead + 1):
            for lag in range(1, lead_target + 1):
                if (cow, t - pd.Timedelta(days=lag)) in picked_set:
                    results[f"lead_{lead_target}"] += 1
                    break

    n_ep = len(onsets_df)
    return {
        "n_episodes": n_ep,
        "mean_episode_length": onsets_df["len"].mean(),
        "lead_1": results["lead_1"] / n_ep,
        "lead_2": results["lead_2"] / n_ep,
        "lead_3": results["lead_3"] / n_ep,
        "onset_in_top_k_same_day": same_day_hits / n_ep,
    }


# =============================================================
# MAIN
# =============================================================
def main():
    print("Loading predictions...")
    df = pd.read_csv(PRED_FILE)
    df["day"] = pd.to_datetime(df["day"])

    required_cols = {"farm_id", "cow_id", "day", "y_true", "y_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing columns: {missing}")

    print(f"Loaded {len(df):,} rows from {df['farm_id'].nunique()} farms")

    results = []
    for farm_id in FARM_MAP.keys():
        df_farm = df.loc[df["farm_id"] == farm_id].copy()
        if len(df_farm) == 0:
            print(f"  {farm_id}: NO DATA, skipping")
            continue
        print(f"\n{farm_id} ({FARM_MAP[farm_id]})  n={len(df_farm):,}")
        r = episode_onset_lead_recall(df_farm, k_frac=K_FRAC, max_lead=3)
        r["farm_id"] = farm_id
        r["farm_label"] = FARM_MAP[farm_id]
        results.append(r)
        print(f"  episodes:                    {r['n_episodes']:,}")
        print(f"  mean episode length:         {r['mean_episode_length']:.2f}")
        print(f"  onset same-day Top-10% hit:  {r['onset_in_top_k_same_day']:.3f}")
        print(f"  episode-onset Lead-1 recall: {r['lead_1']:.3f}")
        print(f"  episode-onset Lead-2 recall: {r['lead_2']:.3f}")
        print(f"  episode-onset Lead-3 recall: {r['lead_3']:.3f}")

    out = pd.DataFrame(results)
    out_path = OUT_DIR / "E2_episode_onset_lead_recall.csv"
    out.to_csv(out_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved: {out_path}")
    print('='*60)
    print(out.to_string(index=False))
    print(f"\nMean episode-onset Lead-3 recall: {out['lead_3'].mean():.3f}")
    print(f"For comparison, current paper reports standard Lead-3: 0.74")
    print()
    print("INTERPRETATION:")
    print("  - If episode-onset Lead-3 stays >= 0.50: strong early-warning signal,")
    print("    NOT just persistence. Paper claim defensible.")
    print("  - If episode-onset Lead-3 < 0.30: warning system mostly persistence-driven.")
    print("    Must reframe Lead-K as 'continued risk flagging' rather than 'early warning'.")
    print("  - Middle ground 0.30-0.50: mixed. Discuss honestly in §4.5.")


if __name__ == "__main__":
    main()
