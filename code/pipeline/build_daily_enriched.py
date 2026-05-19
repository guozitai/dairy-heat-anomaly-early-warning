"""
build_daily_enriched.py
========================
unified_session.parquet → unified_daily_enriched.parquet

逻辑:
  1. 读 unified_session (4459 万 session)
  2. milkYield outlier 过滤 + 班次时段标记 (morning/midday/evening)
  3. 按 (farm_id, bullNumber, milkDate) 聚合到日级
       - daily_milk_total / daily_milk_morning/midday/evening
       - daily_n_sessions
       - 传感器(blood/cond/flow)全天均值 + 最大值
  4. 读 ERA5 3 个县 hourly → daily (THI + 滞后 + 热浪)
  5. join farms_metadata_v3 (farm_id → county_adcode)
  6. join ERA5 daily (county_adcode + milkDate)
  7. 输出 unified_daily_enriched.parquet
"""

import pandas as pd
import numpy as np
import glob, os, time, gc

t = time.time()

# 配置
UNIFIED_PATH = '${DATA_ROOT}/derived/unified_session.parquet'
FARMS_META   = '${DATA_ROOT}/derived/farms_metadata_v3.csv'
ERA5_DIR     = '${DATA_ROOT}/raw/era5'
OUT_PATH     = '${DATA_ROOT}/derived/unified_daily_enriched.parquet'

# 项目用到的 3 个县
COUNTIES = {
    # County code → name mapping redacted (administrative codes for the three counties)
    # See manuscript Section 2.1 for county-level study area description.
}

# Outlier 阈值
MILKYIELD_MIN = 0.01    # > 0 才算
MILKYIELD_MAX = 50.0    # > 50 kg/session 标 NaN
DAILY_MAX     = 80.0    # > 80 kg/day 标 suspicious

# ========================================================================
# 1. 读 unified_session + 预处理
# ========================================================================

print(f"[{time.time()-t:.0f}s] 读 unified_session...", flush=True)
df = pd.read_parquet(UNIFIED_PATH)
print(f"  shape: {df.shape}", flush=True)

# milkYield → 数字
df['milkYield_num'] = pd.to_numeric(df['milkYield'], errors='coerce')

# outlier
df['milk_valid'] = (df['milkYield_num'] > MILKYIELD_MIN) & (df['milkYield_num'] <= MILKYIELD_MAX)
n_drop = (~df['milk_valid']).sum()
print(f"  outlier (=0 或 >50 kg): {n_drop:,} ({n_drop/len(df)*100:.2f}%)", flush=True)

# 班次时段 (按 milkingStart 的 hour)
df['milkingStart_dt'] = pd.to_datetime(df['milkingStart'], errors='coerce')
df['shift_hour'] = df['milkingStart_dt'].dt.hour

def time_period(h):
    if pd.isna(h):
        return 'unknown'
    if 4 <= h < 11:
        return 'morning'
    elif 11 <= h < 17:
        return 'midday'
    else:  # 17-24 + 0-4
        return 'evening'
df['period'] = df['shift_hour'].apply(time_period)

print(f"[{time.time()-t:.0f}s] 班次时段分布:", flush=True)
print(df['period'].value_counts().to_string(), flush=True)

# 把不计入 daily 的字段提前过滤
# - milkYield > 50: 不算这次 session 的产量,但记录 session 存在
df.loc[~df['milk_valid'], 'milkYield_num'] = np.nan

# 传感器字段转数字
SENSOR_COLS = {
    'avg_blood':  'AverageBlood',
    'max_blood':  'MaxBlood',
    'avg_cond':   'AvgConductivity',
    'max_cond':   'MaxConductivity',
    'avg_flow':   'flowVelocity',
    'peak_flow':  'flowRate',
}
for new, old in SENSOR_COLS.items():
    df[new] = pd.to_numeric(df[old], errors='coerce')

# ========================================================================
# 2. session → daily 聚合
# ========================================================================

print(f"[{time.time()-t:.0f}s] session → daily 聚合...", flush=True)

# 主聚合
agg = df.groupby(['farm_id', 'bullNumber', 'milkDate'], observed=True).agg(
    daily_milk_total=('milkYield_num', 'sum'),
    daily_n_sessions=('milk_valid', 'sum'),
    daily_milk_max=('milkYield_num', 'max'),
    daily_milk_min=('milkYield_num', 'min'),
    daily_avg_blood=('avg_blood', 'mean'),
    daily_max_blood=('max_blood', 'mean'),
    daily_avg_cond=('avg_cond', 'mean'),
    daily_max_cond=('max_cond', 'max'),
    daily_avg_flow=('avg_flow', 'mean'),
    daily_peak_flow=('peak_flow', 'max'),
    source=('source', 'first'),
).reset_index()

print(f"  daily 聚合后: {len(agg):,} 行", flush=True)

# 按 period 拆产量
print(f"[{time.time()-t:.0f}s] 按时段拆产量...", flush=True)
period_milk = (df[df['milk_valid']]
    .groupby(['farm_id', 'bullNumber', 'milkDate', 'period'], observed=True)['milkYield_num']
    .sum()
    .unstack(fill_value=0)
    .reset_index())

# rename 列
period_milk = period_milk.rename(columns={
    'morning': 'daily_milk_morning',
    'midday':  'daily_milk_midday',
    'evening': 'daily_milk_evening',
    'unknown': 'daily_milk_unknown_period',
})

# 补齐可能缺的时段列
for col in ['daily_milk_morning', 'daily_milk_midday', 'daily_milk_evening', 'daily_milk_unknown_period']:
    if col not in period_milk.columns:
        period_milk[col] = 0.0

# 合并
agg = agg.merge(period_milk, on=['farm_id', 'bullNumber', 'milkDate'], how='left')

# QC flag
agg['yield_suspicious'] = (agg['daily_milk_total'] > DAILY_MAX) | (agg['daily_milk_total'] < 1)
agg['n_sessions_anomaly'] = (agg['daily_n_sessions'] < 2) | (agg['daily_n_sessions'] > 4)

print(f"  yield_suspicious: {agg['yield_suspicious'].sum():,} ({agg['yield_suspicious'].mean()*100:.2f}%)", flush=True)
print(f"  n_sessions_anomaly: {agg['n_sessions_anomaly'].sum():,} ({agg['n_sessions_anomaly'].mean()*100:.2f}%)", flush=True)

# 释放内存
del df
gc.collect()

# ========================================================================
# 3. 读 ERA5 hourly → daily
# ========================================================================

print(f"[{time.time()-t:.0f}s] 读 ERA5 hourly + 聚合 daily...", flush=True)

def compute_thi(t_c, rh):
    """NRC 1971 THI"""
    return (1.8 * t_c + 32) - (0.55 - 0.0055 * rh) * (1.8 * t_c - 26)

era5_files = {}
for code in COUNTIES:
    pattern = os.path.join(ERA5_DIR, f'{code}_*.csv')
    matches = glob.glob(pattern)
    if matches:
        era5_files[code] = matches[0]
    else:
        print(f"  WARN: county {code} ERA5 文件未找到", flush=True)

print(f"  找到 {len(era5_files)} 个 ERA5 县文件", flush=True)

all_daily_climate = []
for code, path in era5_files.items():
    print(f"  处理 {code} {os.path.basename(path)[:40]}...", flush=True)
    
    h = pd.read_csv(path)
    # 第一列是 datetime (无 header 名)
    h = h.rename(columns={h.columns[0]: 'datetime'})
    h['datetime'] = pd.to_datetime(h['datetime'], errors='coerce')
    h = h.dropna(subset=['datetime'])
    
    # rename 中文列
    h = h.rename(columns={
        '2米气温(C)': 'temp',
        '2米相对湿度(%)': 'rh',
    })
    h['temp'] = pd.to_numeric(h['temp'], errors='coerce')
    h['rh']   = pd.to_numeric(h['rh'], errors='coerce')
    h['thi']  = compute_thi(h['temp'], h['rh'])
    
    # 过滤到项目时间窗 (2022-01 之后,加 lag 缓冲 → 从 2021-12 开始)
    h = h[h['datetime'] >= '2021-12-01']
    
    # 日级聚合
    h['date'] = h['datetime'].dt.date
    daily = h.groupby('date').agg(
        thi_max=('thi', 'max'),
        thi_mean=('thi', 'mean'),
        thi_min=('thi', 'min'),
        temp_max=('temp', 'max'),
        temp_min=('temp', 'min'),
        rh_max=('rh', 'max'),
        rh_mean=('rh', 'mean'),
        hours_thi_gt_68=('thi', lambda x: (x > 68).sum()),
        hours_thi_gt_72=('thi', lambda x: (x > 72).sum()),
        hours_thi_gt_79=('thi', lambda x: (x > 79).sum()),
    ).reset_index()
    
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date').reset_index(drop=True)
    
    # 滞后特征
    daily['thi_max_lag1']     = daily['thi_max'].shift(1)
    daily['thi_max_lag3_avg'] = daily['thi_max'].shift(1).rolling(3, min_periods=1).mean()
    daily['thi_max_lag7_cum'] = daily['hours_thi_gt_72'].shift(1).rolling(7, min_periods=1).sum()
    
    # 热浪事件: 连续 3 天 max_thi > 72
    daily['hot_day'] = (daily['thi_max'] > 72).astype(int)
    daily['heatwave_3d'] = daily['hot_day'].rolling(3, min_periods=3).sum() == 3
    daily['heatwave_3d'] = daily['heatwave_3d'].astype(int)
    daily = daily.drop(columns=['hot_day'])
    
    daily['county_adcode'] = code
    all_daily_climate.append(daily)

climate = pd.concat(all_daily_climate, ignore_index=True)
print(f"  ERA5 daily 总: {len(climate):,} (期望 ≈ 3 县 × 1500 天 = 4500)", flush=True)

# ========================================================================
# 4. join farm_id → county_adcode
# ========================================================================

print(f"[{time.time()-t:.0f}s] 读 farms_metadata + join...", flush=True)

farms = pd.read_csv(FARMS_META)
farms_map = farms[['farm_id', 'county_adcode']].dropna()
farms_map['farm_id'] = farms_map['farm_id'].astype(int)
farms_map['county_adcode'] = farms_map['county_adcode'].astype(int)

print(f"  farm → county 映射: {len(farms_map)} 场", flush=True)
print(farms_map.to_string(index=False), flush=True)

# join 1: agg + county
agg['farm_id'] = agg['farm_id'].astype(int)
agg = agg.merge(farms_map, on='farm_id', how='left')

# 检查未匹配
unmatched_farms = agg[agg['county_adcode'].isna()]['farm_id'].unique()
if len(unmatched_farms) > 0:
    print(f"  WARN: 未匹配 farm_id: {unmatched_farms}", flush=True)

# join 2: agg + climate
agg['milkDate'] = pd.to_datetime(agg['milkDate'])
climate = climate.rename(columns={'date': 'milkDate'})

print(f"[{time.time()-t:.0f}s] join ERA5 climate...", flush=True)
agg = agg.merge(climate, on=['county_adcode', 'milkDate'], how='left')

# 检查 join 后 NaN 比例
thi_na = agg['thi_max'].isna().sum()
print(f"  thi_max NaN: {thi_na:,} ({thi_na/len(agg)*100:.2f}%)", flush=True)

# ========================================================================
# 5. 保存
# ========================================================================

print(f"[{time.time()-t:.0f}s] 写 parquet...", flush=True)

# 类型清理
agg['daily_n_sessions'] = agg['daily_n_sessions'].astype('Int64')
agg['source'] = agg['source'].astype('string')
agg['yield_suspicious'] = agg['yield_suspicious'].astype('boolean')
agg['n_sessions_anomaly'] = agg['n_sessions_anomaly'].astype('boolean')

agg.to_parquet(OUT_PATH, index=False, compression='snappy')

# ========================================================================
# 6. FINAL REPORT
# ========================================================================

print(f"\n[{time.time()-t:.0f}s] FINAL REPORT", flush=True)
print(f"\n  shape: {agg.shape}", flush=True)
print(f"  场: {sorted(agg['farm_id'].unique())}", flush=True)
print(f"  时间: {agg['milkDate'].min()} → {agg['milkDate'].max()}", flush=True)

print(f"\n  按场 × 是否有气候:", flush=True)
agg['has_climate'] = ~agg['thi_max'].isna()
g = agg.groupby('farm_id').agg(
    n_cow_days=('bullNumber', 'count'),
    n_unique_cows=('bullNumber', 'nunique'),
    n_with_climate=('has_climate', 'sum'),
    mean_daily_milk=('daily_milk_total', 'mean'),
    mean_thi_max=('thi_max', 'mean'),
)
print(g.to_string(), flush=True)

print(f"\n  字段清单:", flush=True)
for i, c in enumerate(agg.columns):
    print(f"    [{i:2}] {c}", flush=True)

print(f"\n  ✅ 保存: {OUT_PATH}", flush=True)
print(f"  ⏱  总用时: {time.time()-t:.0f} 秒", flush=True)
