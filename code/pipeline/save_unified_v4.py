"""
v4: 修 v3 的 3 个问题
  1. F06 master 行丢失 → 找出原因
  2. 去重多去 34 万行 → 把 farm_id 在去重前统一为 Int64
  3. 1970 脏日期 → 所有源都 dropna(milkDate)
"""
import pandas as pd
import glob, os, time, gc

t = time.time()

print(f"[{time.time()-t:.0f}s] 读 SQL...", flush=True)
dfs = []
for f in sorted(glob.glob('${DATA_ROOT}/parquet/sql_parquet/*.parquet')):
    df = pd.read_parquet(f)
    for c in ['fixBullNumber', 'cowshed', 'twoMinsMilkYeld']:
        if c in df.columns:
            df = df.drop(columns=[c])
    df['source'] = 'sql'
    dfs.append(df)
df_sql = pd.concat(dfs, ignore_index=True)
del dfs; gc.collect()
df_sql['milkDate'] = pd.to_datetime(df_sql['milkDate'], errors='coerce')
df_sql = df_sql.dropna(subset=['milkDate'])  # ★ 新增:过滤 NaT
df_sql = df_sql[df_sql['milkDate'] >= '2018-01-01'].copy()  # ★ 防 1970
print(f"  SQL: {len(df_sql):,}", flush=True)

print(f"[{time.time()-t:.0f}s] 读 master 早期...", flush=True)
df_m = pd.read_parquet('${DATA_ROOT}/derived/master_session_qc.parquet')
df_m = df_m.rename(columns={
    'CowID': 'bullNumber', 'MilkDate': 'milkDate',
    'BeginTime': 'milkingStart', 'EndTime': 'milkingEnd',
    'TotalYield': 'milkYield',
    'AverageFlow': 'flowVelocity', 'PeakFlow': 'flowRate',
    'BatchNo': 'batch', 'SessionNo': 'shift',
    'ParlorDeviceName': 'milkingParlor', 'Animal0id': 'idenNumber',
})
df_m['milkDate'] = pd.to_datetime(df_m['milkDate'], errors='coerce')
df_m = df_m.dropna(subset=['milkDate'])  # ★ 新增
df_m = df_m[df_m['milkDate'] >= '2018-01-01'].copy()  # ★ 防 1970
# 诊断:看 F06 在 master 里的 milkDate 分布
f6_master = df_m[df_m['farm_id'] == 6]
print(f"  master 里 F06 总行数: {len(f6_master):,}", flush=True)
if len(f6_master) > 0:
    print(f"  master F06 时间窗: {f6_master['milkDate'].min()} → {f6_master['milkDate'].max()}", flush=True)
df_m_early = df_m[df_m['milkDate'] < pd.Timestamp('2024-05-01')].copy()
print(f"  master 早期 (< 2024-05-01): {len(df_m_early):,}", flush=True)
# 诊断:F06 早期
f6_early = df_m_early[df_m_early['farm_id'] == 6]
print(f"  master 早期 F06: {len(f6_early):,}", flush=True)
df_m_early['source'] = 'master'

print(f"[{time.time()-t:.0f}s] 读 y_milk...", flush=True)
df_y = pd.read_csv(
    '${DATA_ROOT}/raw/csv_raw/y_YieldMilk.csv',
    dtype=str, on_bad_lines='skip', low_memory=False,
)
df_y = df_y[df_y['farm_id'] != 'farm_id'].copy()
df_y['farm_id'] = pd.to_numeric(df_y['farm_id'], errors='coerce')
df_y = df_y.dropna(subset=['farm_id'])
df_y['farm_id'] = df_y['farm_id'].astype(int)
df_y = df_y.rename(columns={
    'CowID': 'bullNumber', 'MilkDate': 'milkDate',
    'BeginTime': 'milkingStart', 'EndTime': 'milkingEnd',
    'TotalYield': 'milkYield',
    'AverageFlow': 'flowVelocity', 'PeakFlow': 'flowRate',
    'BatchNo': 'batch', 'SessionNo': 'shift',
    'ParlorDeviceName': 'milkingParlor', 'AnimalOid': 'idenNumber',
})
df_y['milkDate'] = pd.to_datetime(df_y['milkDate'], errors='coerce')
df_y = df_y.dropna(subset=['milkDate'])
df_y = df_y[(df_y['milkDate'] >= '2018-01-01') & (df_y['milkDate'] < '2026-12-31')].copy()
df_y = df_y[(df_y['milkDate'] < pd.Timestamp('2024-05-01')) | (df_y['farm_id'] == 7)].copy()
df_y['source'] = 'y_milk'
print(f"  y_milk: {len(df_y):,}", flush=True)

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
    'source',
]

STR_COLS = ['bullNumber', 'idenNumber', 'milkingParlor',
            'milkYield', 'flowVelocity', 'flowRate',
            'RelativeConductivity', 'TakeOffFlow',
            'ManualMode', 'ManualDetach', 'ForcedRetract',
            'BloodAlarm', 'ConductivityAlarm', 'source']

# ★ 关键修复:去重前要用纯 int (不是 Int64) 保证类型一致
def normalize(df):
    for c in UNIFIED_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[UNIFIED_COLS].copy()
    for c in STR_COLS:
        df[c] = df[c].astype('string')
    # farm_id 转 int (普通 int,不是 Int64)
    df['farm_id'] = pd.to_numeric(df['farm_id'], errors='coerce').fillna(-1).astype('int64')
    # 其他 int 字段转 nullable Int64
    for c in ['shift', 'batch', 'platformNumber', 'Slips', 'KickOffs', 'Blocks', 'NoOfReattaches']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in ['AverageBlood', 'MaxBlood', 'MaxConductivity', 'AvgConductivity']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['milkDate', 'idenDate', 'milkingStart', 'milkingEnd', 'peakTime', 'UpDateTime']:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

print(f"[{time.time()-t:.0f}s] normalize SQL...", flush=True)
df_sql = normalize(df_sql)
print(f"[{time.time()-t:.0f}s] normalize master...", flush=True)
df_m_early = normalize(df_m_early)
print(f"[{time.time()-t:.0f}s] normalize y_milk...", flush=True)
df_y = normalize(df_y)

print(f"[{time.time()-t:.0f}s] concat...", flush=True)
df_all = pd.concat([df_sql, df_m_early, df_y], ignore_index=True)
del df_sql, df_m_early, df_y; gc.collect()
print(f"  合并: {len(df_all):,}", flush=True)

# ★ 诊断:concat 之后 F06 行数
f6 = df_all[df_all['farm_id'] == 6]
print(f"  concat 后 F06 总行: {len(f6):,}", flush=True)
print(f"  F06 按 source: {f6['source'].value_counts().to_dict()}", flush=True)

print(f"[{time.time()-t:.0f}s] 去重...", flush=True)
before = len(df_all)
df_all = df_all.drop_duplicates(
    subset=['farm_id', 'bullNumber', 'milkDate', 'milkingStart'],
    keep='first',
)
print(f"  {before:,} → {len(df_all):,} (去 {before-len(df_all):,})", flush=True)

# ★ 诊断:去重后 F06 行数
f6 = df_all[df_all['farm_id'] == 6]
print(f"  去重后 F06: {len(f6):,}", flush=True)
print(f"  F06 按 source: {f6['source'].value_counts().to_dict()}", flush=True)

print(f"[{time.time()-t:.0f}s] 写 parquet...", flush=True)
out = '${DATA_ROOT}/derived/unified_session.parquet'
df_all.to_parquet(out, index=False, compression='snappy')
print(f"  ✅ {out}", flush=True)

print(f"\n[{time.time()-t:.0f}s] FINAL REPORT", flush=True)
print(f"  shape: {df_all.shape}", flush=True)
print(f"  场: {sorted(df_all['farm_id'].unique())}", flush=True)
print(f"  时间: {df_all['milkDate'].min()} → {df_all['milkDate'].max()}", flush=True)
print(f"\n按场×来源:", flush=True)
print(df_all.groupby(['farm_id', 'source'], observed=True).size().unstack(fill_value=0), flush=True)
