"""parity 推算 + merge_asof 改用一次性 global sort (不用 by 参数)"""
import pandas as pd
import numpy as np

CLEAN = '${DATA_ROOT}/derived/unified_daily_clean.parquet'
ECAL  = '${DATA_ROOT}/derived/e_calving.parquet'

print('读 e_calving...')
ec = pd.read_parquet(ECAL)
ec_sorted = ec[['farm_id','bullNumber','calvingDate']].dropna().copy()
ec_sorted['farm_id'] = ec_sorted['farm_id'].astype('int64')
ec_sorted['bullNumber'] = ec_sorted['bullNumber'].astype(str)
ec_sorted['calvingDate'] = ec_sorted['calvingDate'].astype('datetime64[ns]')
ec_sorted = ec_sorted.sort_values(['farm_id','bullNumber','calvingDate']).reset_index(drop=True)
ec_sorted['inferred_parity'] = ec_sorted.groupby(['farm_id','bullNumber']).cumcount() + 1

# 制作 (farm, cow) 联合 key,避免 by 参数
ec_sorted['_key'] = ec_sorted['farm_id'].astype(str) + '_' + ec_sorted['bullNumber']

print('读 clean parquet...')
df = pd.read_parquet(CLEAN)
df['farm_id'] = df['farm_id'].astype('int64')
df['bullNumber'] = df['bullNumber'].astype(str)
df['milkDate'] = df['milkDate'].astype('datetime64[ns]')
df['_key'] = df['farm_id'].astype(str) + '_' + df['bullNumber']

print('准备 sort...')
# 全局排序 by milkDate (无 NaT)
df_idx = df[['_key','milkDate']].copy()
df_idx['_orig_idx'] = df_idx.index
# 关键: NaT 也要 filter,虽然不应该有
print(f'  milkDate NaT: {df_idx["milkDate"].isna().sum()}')
df_idx = df_idx.dropna(subset=['milkDate']).sort_values('milkDate').reset_index(drop=True)
ec_sorted = ec_sorted.dropna(subset=['calvingDate']).sort_values('calvingDate').reset_index(drop=True)

print('asof merge (global sort)...')
merged = pd.merge_asof(
    df_idx, ec_sorted[['_key','calvingDate','inferred_parity']],
    left_on='milkDate', right_on='calvingDate',
    by='_key',
    direction='backward'
)

print('回填到 df...')
df['inferred_parity'] = pd.NA
df.loc[merged['_orig_idx'].values, 'inferred_parity'] = merged['inferred_parity'].values
df = df.drop(columns=['_key'])

print(f'\nparity merge 结果:')
print(f'  有 parity: {df["inferred_parity"].notna().sum():,} ({df["inferred_parity"].notna().mean()*100:.1f}%)')
print(df['inferred_parity'].value_counts(dropna=False).sort_index().head(10))

print('\n按场 (头胎%, ≥4胎%, 中位):')
for fid in sorted(df['farm_id'].unique()):
    sub = df[(df['farm_id']==fid) & df['inferred_parity'].notna()]
    if len(sub)==0: continue
    p1 = (sub['inferred_parity']==1).mean()*100
    p_high = (sub['inferred_parity']>=4).mean()*100
    print(f'  NX{fid:02d}: 头胎 {p1:>4.0f}%, ≥4胎 {p_high:>4.0f}%, 中位 {sub["inferred_parity"].median():.0f}')

df.to_parquet(CLEAN, index=False)
print(f'\n已更新: {CLEAN}')
