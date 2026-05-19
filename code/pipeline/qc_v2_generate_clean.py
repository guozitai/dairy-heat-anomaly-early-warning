"""
QC v2 - 生成清洁子集 unified_daily_clean.parquet

双触发规则:
  (a) records_per_cow_per_day < 0.5      # 数据采集故障
  (b) yield_ratio < 0.6                   # 产量异常崩塌
       expected_milk = 同场同THI±3 范围内的非故障月份均值

输出:在 v2 基础上加 qc_flag (bool) 和 qc_reason (str) 两列
"""
import pandas as pd

IN = '${DATA_ROOT}/derived/unified_daily_enriched_v2.parquet'
OUT = '${DATA_ROOT}/derived/unified_daily_clean.parquet'

print('读 v2...')
df = pd.read_parquet(IN)
print(f'原始: {len(df):,} 行')

# === 计算每场每月的 QC 指标 ===
df_sql = df[(df['source']=='sql') & (~df['yield_suspicious'])].dropna(subset=['daily_milk_total','thi_max']).copy()
df_sql['ym'] = df_sql['milkDate'].dt.to_period('M')
df_sql['days'] = df_sql['milkDate'].dt.daysinmonth

farm_month_qc = {}
for fid in df_sql['farm_id'].unique():
    sub = df_sql[df_sql['farm_id']==fid]
    tbl = sub.groupby('ym').agg(
        cows=('bullNumber','nunique'),
        n_records=('daily_milk_total','size'),
        days=('days','first'),
        mean_milk=('daily_milk_total','mean'),
        mean_thi=('thi_max','mean'),
    )
    tbl['rec_per_cow_day'] = tbl['n_records'] / (tbl['cows'] * tbl['days'])
    
    # expected_milk 用同 THI±3 范围内、density 正常的月份均值
    expected = []
    for ym, row in tbl.iterrows():
        peers = tbl[(tbl['mean_thi'].between(row['mean_thi']-3, row['mean_thi']+3)) 
                    & (tbl.index != ym) & (tbl['rec_per_cow_day']>=0.5)]
        expected.append(peers['mean_milk'].mean() if len(peers)>0 else None)
    tbl['expected_milk'] = expected
    tbl['yield_ratio'] = tbl['mean_milk'] / tbl['expected_milk']
    
    tbl['flag_density'] = tbl['rec_per_cow_day'] < 0.5
    tbl['flag_yield'] = (tbl['yield_ratio'].notna()) & (tbl['yield_ratio'] < 0.6)
    tbl['qc_flag'] = tbl['flag_density'] | tbl['flag_yield']
    
    for ym, row in tbl.iterrows():
        reasons = []
        if row['flag_density']: reasons.append(f'density={row["rec_per_cow_day"]:.2f}')
        if row['flag_yield']: reasons.append(f'yield_ratio={row["yield_ratio"]:.2f}')
        farm_month_qc[(fid, ym)] = {
            'qc_flag': bool(row['qc_flag']),
            'qc_reason': '; '.join(reasons) if reasons else '',
            'rec_per_cow_day': float(row['rec_per_cow_day']),
            'yield_ratio': float(row['yield_ratio']) if pd.notna(row['yield_ratio']) else None,
        }

# === 把 QC flag 映射回原 df 的每一行 ===
df['ym'] = df['milkDate'].dt.to_period('M')
df['qc_flag'] = False
df['qc_reason'] = ''

# 只对 sql 场月查 qc(jibuqi 等其他 source 不在 qc 范围,默认 False)
for (fid, ym), qc in farm_month_qc.items():
    mask = (df['farm_id']==fid) & (df['ym']==ym)
    df.loc[mask, 'qc_flag'] = qc['qc_flag']
    df.loc[mask, 'qc_reason'] = qc['qc_reason']

df = df.drop(columns=['ym'])

print(f'\nQC 汇总:')
print(f'  总行数: {len(df):,}')
print(f'  qc_flag=True: {df["qc_flag"].sum():,} ({df["qc_flag"].mean()*100:.1f}%)')
print(f'  qc_flag=False (主分析子集): {(~df["qc_flag"]).sum():,}')

print(f'\n按场 qc 排除比例:')
for fid in sorted(df['farm_id'].unique()):
    sub = df[df['farm_id']==fid]
    if len(sub)>0:
        print(f'  NX{fid:02d}: {sub["qc_flag"].sum():>8,} / {len(sub):>8,} ({sub["qc_flag"].mean()*100:>5.1f}%)')

df.to_parquet(OUT, index=False)
print(f'\n保存: {OUT}')
