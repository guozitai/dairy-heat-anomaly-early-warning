"""
Paired Comparison v2 - Figure 2 数据 (修复版)
"""
import pandas as pd
import numpy as np

PATH = '${DATA_ROOT}/derived/unified_daily_clean.parquet'
OUT  = '${DATA_ROOT}/derived/paired_comparison_v2.csv'

print('读 clean parquet...')
df = pd.read_parquet(PATH, columns=['farm_id','bullNumber','daily_milk_total','thi_max','DIM','qc_flag','yield_suspicious','source','milkDate'])

df = df[(df['qc_flag']==False) & (~df['yield_suspicious'])]
df = df.dropna(subset=['daily_milk_total','thi_max','DIM'])
df = df[(df['DIM']>=60) & (df['DIM']<=200)]
df['daily_milk_total'] = df['daily_milk_total'].astype('float64')

print(f'过滤后: {len(df):,} 行\n')

df['zone'] = pd.cut(df['thi_max'], 
                    bins=[-1, 65, 72, 78, 999], 
                    labels=['cold','baseline','transition','hot'])
df = df[df['zone'].isin(['baseline','hot'])]
print(f'保留 baseline + hot: {len(df):,} 行\n')

results = []
for fid in sorted(df['farm_id'].unique()):
    sub = df[df['farm_id']==fid]
    
    cow_means = sub.groupby(['bullNumber','zone'], observed=True).agg(
        milk=('daily_milk_total','mean'),
        n=('daily_milk_total','size')
    ).reset_index()
    cow_means = cow_means[cow_means['n']>=5]
    
    # 防御性检查
    zones_present = cow_means['zone'].unique()
    if 'baseline' not in zones_present or 'hot' not in zones_present:
        print(f'NX{fid:02d}: 缺 zone (有 {list(zones_present)}), 跳过')
        continue
    
    pv = cow_means.pivot(index='bullNumber', columns='zone', values='milk')
    
    if 'baseline' not in pv.columns or 'hot' not in pv.columns:
        print(f'NX{fid:02d}: pivot 缺列, 跳过')
        continue
    
    pv = pv.dropna(subset=['baseline','hot'])
    
    if len(pv) == 0:
        print(f'NX{fid:02d}: 无配对牛, 跳过'); continue
    
    pv['diff'] = pv['hot'] - pv['baseline']
    
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_medians = np.array([
        np.median(rng.choice(pv['diff'].values, size=len(pv), replace=True))
        for _ in range(n_boot)
    ])
    ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])
    
    res = {
        'farm_id': fid,
        'n_cows_paired': len(pv),
        'baseline_median': pv['baseline'].median(),
        'hot_median': pv['hot'].median(),
        'median_diff': pv['diff'].median(),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'pct_decline': (pv['diff']<0).mean()*100,
        'pct_decline_2kg': (pv['diff']<-2).mean()*100,
    }
    results.append(res)
    
    print(f'NX{fid:02d}: n={len(pv):>6,} cows | baseline {res["baseline_median"]:>5.2f} → hot {res["hot_median"]:>5.2f} | Δ={res["median_diff"]:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}] | {res["pct_decline"]:.0f}% decline, {res["pct_decline_2kg"]:.0f}% decline>2kg')

pd.DataFrame(results).to_csv(OUT, index=False)
print(f'\n保存: {OUT}')
