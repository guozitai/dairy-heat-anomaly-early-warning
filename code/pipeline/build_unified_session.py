"""
build_unified_session.py
========================
合并三批挤奶数据到 unified_session.parquet

策略 2: SQL 为基底, master 补 SQL 之前的数据, y_milk 补 master 之前的早期 + FARM_X (excluded farm)

数据源:
  1. ${DATA_ROOT}/parquet/sql_parquet/*.parquet  (3380万行)
  2. ${DATA_ROOT}/derived/master_session_qc.parquet  (1273万行)
  3. ${DATA_ROOT}/raw/csv_raw/y_YieldMilk.csv  (1088万行)

输出:
  ${DATA_ROOT}/derived/unified_session.parquet
  
策略:
  - SQL 全部保留 (最新最全)
  - master 只保留 MilkDate < 2024-05-01 部分 (主要是 FARM_1 早期)
  - y_milk 只保留 SQL+master 都没覆盖的部分 (主要是 FARM_X (excluded farm) + FARM_1 2022-2023)
  
去重(防止万一):
  key = (farm_id, bullNumber, milkDate, milkingStart)
"""

import pandas as pd
import numpy as np
import glob
import os
import time

t_start = time.time()

print("=" * 70)
print("Build unified_session.parquet")
print("=" * 70)

# 标准字段集(以 SQL 为主, 33 列结构,统一字段名)
UNIFIED_COLS = [
    'farm_id', 'bullNumber', 'milkDate', 'milkingParlor',
    'shift', 'batch', 'platformNumber',
    'idenDate', 'idenNumber',
    'milkingStart', 'milkingEnd',
    'milkYield', 'flowVelocity', 'flowRate', 'peakTime',
    'Slips', 'KickOffs', 'Blocks', 'NoOfReattaches',
    'AverageBlood', 'MaxBlood',
    'MaxConductivity', 'AvgConductivity', 'RelativeConductivity',
    'ManualMode', 'ManualDetach', 'ForcedRetract',
    'BloodAlarm', 'ConductivityAlarm',
    'TakeOffFlow', 'UpDateTime',
    'source',  # 标记来源: sql / master / y_milk
]

# ============== 1. 读 SQL parquet (基底) ==============
print("\n[1/4] 读 SQL parquet (基底)...")
sql_files = sorted(glob.glob('${DATA_ROOT}/parquet/sql_parquet/*.parquet'))
dfs_sql = []
for f in sql_files:
    df = pd.read_parquet(f)
    name = os.path.basename(f)
    print(f"  {name[:40]}: {len(df):,} 行")
    dfs_sql.append(df)
df_sql = pd.concat(dfs_sql, ignore_index=True)
print(f"  合计: {len(df_sql):,} 行, {df_sql['farm_id'].nunique()} 场")

# 删除 FARM_8 特有的 3 个字段(fixBullNumber/cowshed/twoMinsMilkYeld),
# 因为其他场没有,合并时全 NaN 不利
nx03_extra = ['fixBullNumber', 'cowshed', 'twoMinsMilkYeld']
for c in nx03_extra:
    if c in df_sql.columns:
        df_sql = df_sql.drop(columns=[c])

df_sql['source'] = 'sql'
df_sql['milkDate'] = pd.to_datetime(df_sql['milkDate'], errors='coerce')

# ============== 2. 读 master_session_qc (补早期) ==============
print("\n[2/4] 读 master_session_qc (补 SQL 之前的早期)...")
df_master = pd.read_parquet('${DATA_ROOT}/derived/master_session_qc.parquet')
print(f"  原始: {len(df_master):,} 行")

# 字段名对齐
df_master = df_master.rename(columns={
    'CowID': 'bullNumber',
    'MilkDate': 'milkDate',
    'BeginTime': 'milkingStart',
    'EndTime': 'milkingEnd',
    'TotalYield': 'milkYield',
    'AverageFlow': 'flowVelocity',
    'PeakFlow': 'flowRate',
    'BatchNo': 'batch',
    'SessionNo': 'shift',
    'ParlorDeviceName': 'milkingParlor',
    'Animal0id': 'idenNumber',  # 用 Animal0id 占 idenNumber 字段
})

df_master['milkDate'] = pd.to_datetime(df_master['milkDate'], errors='coerce')

# 只保留 SQL 起点(2024-05-01)之前的数据
cutoff = pd.Timestamp('2024-05-01')
df_master_early = df_master[df_master['milkDate'] < cutoff].copy()
print(f"  保留 < 2024-05-01 的部分: {len(df_master_early):,} 行")
print(f"    场分布: {df_master_early['farm_id'].value_counts().sort_index().to_dict()}")

df_master_early['source'] = 'master'

# ============== 3. 读 y_YieldMilk (补更早 + FARM_X (excluded farm)) ==============
print("\n[3/4] 读 y_YieldMilk (补早期 + FARM_X (excluded farm))...")
df_yy = pd.read_csv(
    '${DATA_ROOT}/raw/csv_raw/y_YieldMilk.csv',
    dtype={'farm_id': str, 'CowID': str, 'BatchNo': str, 'SessionNo': str},
    on_bad_lines='skip',
)
df_yy = df_yy[df_yy['farm_id'] != 'farm_id'].copy()
df_yy['farm_id'] = pd.to_numeric(df_yy['farm_id'], errors='coerce')
df_yy = df_yy.dropna(subset=['farm_id'])
df_yy['farm_id'] = df_yy['farm_id'].astype(int)

# 字段对齐
df_yy = df_yy.rename(columns={
    'CowID': 'bullNumber',
    'MilkDate': 'milkDate',
    'BeginTime': 'milkingStart',
    'EndTime': 'milkingEnd',
    'TotalYield': 'milkYield',
    'AverageFlow': 'flowVelocity',
    'PeakFlow': 'flowRate',
    'BatchNo': 'batch',
    'SessionNo': 'shift',
    'ParlorDeviceName': 'milkingParlor',
    'AnimalOid': 'idenNumber',
})

df_yy['milkDate'] = pd.to_datetime(df_yy['milkDate'], errors='coerce')
df_yy = df_yy.dropna(subset=['milkDate'])

# 过滤异常日期 (1970, 2004 等)
df_yy = df_yy[(df_yy['milkDate'] >= '2018-01-01') & (df_yy['milkDate'] < '2026-12-31')].copy()
print(f"  过滤异常日期后: {len(df_yy):,} 行")

# 只保留 < 2024-05-01 (跟 SQL 不重叠) 或 farm=7 (FARM_X (excluded farm),SQL/master 都没有)
df_yy_keep = df_yy[(df_yy['milkDate'] < cutoff) | (df_yy['farm_id'] == 7)].copy()
print(f"  保留 (< 2024-05-01 或 farm=7): {len(df_yy_keep):,} 行")
print(f"    场分布: {df_yy_keep['farm_id'].value_counts().sort_index().to_dict()}")

df_yy_keep['source'] = 'y_milk'

# ============== 4. 合并 ==============
print("\n[4/4] 合并三批...")

# 各取 UNIFIED_COLS 中存在的字段(缺失填 NaN)
def align(df):
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[UNIFIED_COLS]

df_sql_a = align(df_sql)
df_master_a = align(df_master_early)
df_yy_a = align(df_yy_keep)

df_unified = pd.concat([df_sql_a, df_master_a, df_yy_a], ignore_index=True)
print(f"  合并后: {len(df_unified):,} 行")

# 去重 (保险措施)
print("\n  去重 (farm_id + bullNumber + milkDate + milkingStart)...")
before = len(df_unified)
df_unified = df_unified.drop_duplicates(
    subset=['farm_id', 'bullNumber', 'milkDate', 'milkingStart'],
    keep='first',  # 保留先放入的(SQL 优先)
)
after = len(df_unified)
print(f"    {before:,} → {after:,} (去 {before-after:,} 行)")

# 排序
df_unified = df_unified.sort_values(['farm_id', 'milkDate', 'bullNumber']).reset_index(drop=True)

# ============== 5. 保存 + 报告 ==============
out_path = '${DATA_ROOT}/derived/unified_session.parquet'
print(f"\n保存到 {out_path}...")
df_unified.to_parquet(out_path, index=False)

print("\n" + "=" * 70)
print("FINAL REPORT")
print("=" * 70)
print(f"\n总行数: {len(df_unified):,}")
print(f"总场数: {df_unified['farm_id'].nunique()}")
print(f"时间范围: {df_unified['milkDate'].min()} → {df_unified['milkDate'].max()}")

print(f"\n按场的来源分布:")
g = df_unified.groupby(['farm_id', 'source']).size().unstack(fill_value=0)
print(g.to_string())

print(f"\n按月份的总行数 (头尾 5 个):")
df_unified['ym'] = df_unified['milkDate'].dt.to_period('M')
monthly = df_unified.groupby('ym').size()
print("最早 5 个月:")
print(monthly.head(5))
print("最近 5 个月:")
print(monthly.tail(5))

print(f"\n用时: {time.time()-t_start:.1f} 秒")
