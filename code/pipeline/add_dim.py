"""
add_dim.py
==========
为 unified_daily_enriched 每行 (farm_id, bullNumber, milkDate) 计算 DIM (days in milk)

DIM = milkDate - 该牛最近一次产犊日期 (在 milkDate 之前)

逻辑:
  1. 读 e_calving.parquet + unified_daily_enriched.parquet
  2. 对每行 daily 数据, 找到该牛在 milkDate 之前最近一次的 calvingDate
  3. DIM = (milkDate - calvingDate).days
  4. 保存 unified_daily_enriched_v2.parquet
"""
import pandas as pd
import time

t = time.time()

print(f"[{time.time()-t:.0f}s] 读 daily + calving...", flush=True)
daily = pd.read_parquet(
    '${DATA_ROOT}/derived/unified_daily_enriched.parquet',
)
calving = pd.read_parquet(
    '${DATA_ROOT}/derived/e_calving.parquet',
)

# 排除 2017-01-01 系统初始化日期 (可疑)
calving = calving[calving['calvingDate'] >= '2017-06-01']
# 只保留 unified 涉及的 farm
target_farms = sorted(daily['farm_id'].unique())
calving = calving[calving['farm_id'].isin(target_farms)]
calving['bullNumber'] = calving['bullNumber'].astype(str)
daily['bullNumber'] = daily['bullNumber'].astype(str)

print(f"  daily: {len(daily):,} 行", flush=True)
print(f"  calving (过滤后): {len(calving):,} 行", flush=True)

# ============== 算 DIM ==============
print(f"\n[{time.time()-t:.0f}s] 计算 DIM...", flush=True)

# 排序 calving 表
calving = calving.sort_values(['farm_id', 'bullNumber', 'calvingDate'])

# 对 daily 也排序
daily = daily.sort_values(['farm_id', 'bullNumber', 'milkDate']).reset_index(drop=True)

# 用 merge_asof 找每个 milkDate 之前最近的 calvingDate
print(f"[{time.time()-t:.0f}s] merge_asof (按 farm × bullNumber 分组)...", flush=True)

daily['milkDate'] = pd.to_datetime(daily['milkDate'])
calving['calvingDate'] = pd.to_datetime(calving['calvingDate'])

# merge_asof 要求 by 分组 + on 时间排序
merged = pd.merge_asof(
    daily,
    calving[['farm_id', 'bullNumber', 'calvingDate', 'parity']],
    by=['farm_id', 'bullNumber'],
    left_on='milkDate',
    right_on='calvingDate',
    direction='backward',  # milkDate 之前最近的 calving
)

# DIM 计算
merged['DIM'] = (merged['milkDate'] - merged['calvingDate']).dt.days

# 异常处理:
#   DIM < 0    → 没匹配到 (该牛产犊记录不在数据库,设 NaN)
#   DIM > 700  → 太长,可能是上一胎数据,设 NaN (奶牛干奶期前 305 天泌乳, 700 天明显是过早匹配)
#   DIM 0-305 → 正常泌乳期
#   DIM 305-450 → 干奶期或晚泌乳

merged.loc[merged['DIM'] < 0, 'DIM'] = pd.NA
merged.loc[merged['DIM'] > 700, 'DIM'] = pd.NA

print(f"[{time.time()-t:.0f}s] DIM 计算完成", flush=True)

n_total = len(merged)
n_dim_ok = merged['DIM'].notna().sum()
print(f"\n  DIM 匹配统计:")
print(f"    总行数: {n_total:,}")
print(f"    DIM 有效: {n_dim_ok:,} ({n_dim_ok/n_total*100:.2f}%)")
print(f"    DIM 缺失 (没匹配到产犊): {n_total - n_dim_ok:,}")

# DIM 分布
dim_ok = merged[merged['DIM'].notna()]['DIM'].astype(int)
print(f"\n  DIM 分布:")
print(f"    min:    {dim_ok.min()}")
print(f"    p10:    {dim_ok.quantile(0.1):.0f}")
print(f"    median: {dim_ok.median():.0f}")
print(f"    p90:    {dim_ok.quantile(0.9):.0f}")
print(f"    max:    {dim_ok.max()}")

# 按场看 DIM 匹配率
print(f"\n  按场 DIM 匹配率:")
g = merged.groupby('farm_id').agg(
    n=('DIM', 'count'),
    matched=('DIM', lambda x: x.notna().sum()),
    median_dim=('DIM', 'median'),
)
g['match_rate'] = (g['matched'] / g['n'] * 100).round(1)
print(g.to_string())

# ============== 保存 v2 ==============
out = '${DATA_ROOT}/derived/unified_daily_enriched_v2.parquet'
merged.to_parquet(out, index=False, compression='snappy')
print(f"\n[{time.time()-t:.0f}s] ✅ 保存: {out}")
print(f"⏱  用时: {time.time()-t:.0f} 秒")
