"""
β slope v2 - Figure 1 数据

设计:
  - 数据: clean parquet, qc_flag=False, ~yield_suspicious
  - 范围: THI >= 68, DIM in [60, 300]
  - 公式: daily_milk ~ thi_max + DIM + I(DIM**2) + (1|bullNumber)
  - 容错: mixedlm singular 时降级 OLS with cluster-robust SE
  - Sensitivity: 同时输出 THI >= 68 和 THI >= 72 两套
"""
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

PATH = '${DATA_ROOT}/derived/unified_daily_clean.parquet'
OUT = '${DATA_ROOT}/derived/beta_lmm_v2.csv'

print('读 clean parquet...')
df = pd.read_parquet(PATH, columns=['farm_id','bullNumber','daily_milk_total','thi_max','DIM','qc_flag','yield_suspicious','source'])
df = df[(df['qc_flag']==False) & (~df['yield_suspicious'])]
df = df.dropna(subset=['daily_milk_total','thi_max','DIM'])
df = df[df['DIM'].between(60, 300)]
df = df.astype({'daily_milk_total':'float64','thi_max':'float64','DIM':'float64'})
print(f'过滤 (DIM 60-300, qc, suspicious): {len(df):,} 行\n')

results = []

for thi_cutoff in [68, 72]:
    print(f'\n{"="*60}\n=== THI >= {thi_cutoff} ===\n{"="*60}')
    dft = df[df['thi_max']>=thi_cutoff]
    
    for fid in sorted(dft['farm_id'].unique()):
        sub = dft[dft['farm_id']==fid].copy()
        n_rows = len(sub)
        n_cows = sub['bullNumber'].nunique()
        
        if n_rows < 1000 or n_cows < 50:
            print(f'NX{fid:02d}: n_rows={n_rows:,} n_cows={n_cows:,} — too small, skip')
            continue
        
        print(f'NX{fid:02d}: n_rows={n_rows:,} n_cows={n_cows:,} ', end='', flush=True)
        t0 = time.time()
        
        # 尝试 mixedlm
        try:
            m = smf.mixedlm(
                "daily_milk_total ~ thi_max + DIM + I(DIM**2)",
                sub, groups=sub['bullNumber']
            ).fit(method='lbfgs', disp=False, reml=True)
            
            cow_var = float(m.cov_re.iloc[0,0])
            singular = cow_var < 0.01  # cow_var ≈ 0 表示崩了
            
            if singular:
                raise ValueError(f'cow_var={cow_var:.4f} singular')
            
            res = {
                'thi_cutoff': thi_cutoff, 'farm_id': fid, 'method': 'mixedlm',
                'n_rows': n_rows, 'n_cows': n_cows,
                'beta_thi': m.params['thi_max'], 'se_thi': m.bse['thi_max'], 'p_thi': m.pvalues['thi_max'],
                'beta_DIM': m.params['DIM'], 'beta_DIM2': m.params['I(DIM ** 2)'],
                'intercept': m.params['Intercept'],
                'cow_var': cow_var, 'resid_var': m.scale,
                'minutes': (time.time()-t0)/60,
            }
            print(f'mixedlm β={res["beta_thi"]:+.4f} cow_var={cow_var:.2f} {res["minutes"]:.1f}min')
        
        except Exception as e:
            # 降级 OLS with cluster-robust SE
            print(f'[mixedlm failed: {str(e)[:50]}] → OLS w/ cluster SE ', end='', flush=True)
            m = smf.ols(
                "daily_milk_total ~ thi_max + DIM + I(DIM**2)", sub
            ).fit(cov_type='cluster', cov_kwds={'groups': sub['bullNumber']})
            
            res = {
                'thi_cutoff': thi_cutoff, 'farm_id': fid, 'method': 'ols_cluster',
                'n_rows': n_rows, 'n_cows': n_cows,
                'beta_thi': m.params['thi_max'], 'se_thi': m.bse['thi_max'], 'p_thi': m.pvalues['thi_max'],
                'beta_DIM': m.params['DIM'], 'beta_DIM2': m.params['I(DIM ** 2)'],
                'intercept': m.params['Intercept'],
                'cow_var': np.nan, 'resid_var': float(m.mse_resid),
                'minutes': (time.time()-t0)/60,
            }
            print(f'β={res["beta_thi"]:+.4f} {res["minutes"]:.1f}min')
        
        results.append(res)
        pd.DataFrame(results).to_csv(OUT, index=False)

print(f'\n\n保存: {OUT}')
print('\n=== 最终结果对比 ===')
final = pd.DataFrame(results)
piv = final.pivot(index='farm_id', columns='thi_cutoff', values='beta_thi').round(4)
piv.columns = [f'β@THI≥{c}' for c in piv.columns]
print(piv)
