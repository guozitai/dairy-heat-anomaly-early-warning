"""
Step 2: 5 farms (FARM_7/06/13/14/15) rumination v3.1 vs v4 sensitivity
Train/test 限制在反刍数据窗口内
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import warnings, time, os
warnings.filterwarnings('ignore')

CLEAN = '${DATA_ROOT}/derived/unified_daily_clean.parquet'
OUT_CSV = '${DATA_ROOT}/derived/rumination_sensitivity_5farms.csv'

FARMS = [5, 6, 13, 14, 15]

FARM_SPLIT = {
    5:  {'data_start':'2022-01-01', 'train_end':'2023-03-31', 'test_start':'2023-04-01', 'test_end':'2023-09-30'},
    6:  {'data_start':'2022-01-01', 'train_end':'2023-03-31', 'test_start':'2023-04-01', 'test_end':'2023-09-30'},
    13: {'data_start':'2022-01-01', 'train_end':'2023-03-31', 'test_start':'2023-04-01', 'test_end':'2023-09-30'},
    14: {'data_start':'2022-01-01', 'train_end':'2023-03-31', 'test_start':'2023-04-01', 'test_end':'2023-09-30'},
    15: {'data_start':'2022-01-01', 'train_end':'2023-06-30', 'test_start':'2023-07-01', 'test_end':'2024-05-31'},
}

FEATURES_V31 = ['thi_max','thi_max_lag1','thi_max_lag3_avg','thi_max_lag7_cum',
                'heatwave_3d','hours_thi_gt_72','hours_thi_gt_79',
                'parity_group','DIM','milk_lag1','expected_milk','ratio_lag1']
FEATURES_V4 = FEATURES_V31 + ['daily_rumination','daily_activity','rumination_lag1','activity_lag1',
                              'has_rumination','has_activity','has_rum_lag1','has_act_lag1']


def run_farm(fid):
    cfg = FARM_SPLIT[fid]
    rum_path = f'${DATA_ROOT}/derived/NX{fid:02d}_rumination_daily.parquet'
    if not os.path.exists(rum_path):
        return {'farm_id':fid, 'status':f'no_parquet'}
    
    print(f'\n{"="*65}\n=== NX{fid:02d} ===\n{"="*65}', flush=True)
    
    cols = ['farm_id','bullNumber','milkDate','daily_milk_total','thi_max',
            'thi_max_lag1','thi_max_lag3_avg','thi_max_lag7_cum','heatwave_3d',
            'hours_thi_gt_72','hours_thi_gt_79',
            'DIM','inferred_parity','qc_flag','yield_suspicious']
    df = pd.read_parquet(CLEAN, columns=cols)
    df = df[df['farm_id']==fid]
    df = df[(df['qc_flag']==False) & (~df['yield_suspicious'])]
    df = df.dropna(subset=['daily_milk_total','thi_max','DIM','inferred_parity'])
    df = df[df['DIM'].between(60,300)]
    df = df[(df['milkDate']>=cfg['data_start']) & (df['milkDate']<=cfg['test_end'])]
    df['daily_milk_total'] = df['daily_milk_total'].astype('float64')
    df['inferred_parity'] = df['inferred_parity'].astype('float64')
    df['parity_group'] = df['inferred_parity'].clip(upper=4).astype(int)
    df['DIM_bin'] = pd.cut(df['DIM'], bins=range(60,331,30), labels=False)
    df['bullNumber'] = df['bullNumber'].astype(str)
    df['milkDate'] = df['milkDate'].astype('datetime64[ns]')
    
    if len(df) < 5000:
        return {'farm_id':fid, 'status':f'data_too_small ({len(df)})'}
    
    rum = pd.read_parquet(rum_path, columns=['bullNumber','date','daily_rumination','daily_activity'])
    rum['bullNumber'] = rum['bullNumber'].astype(str)
    rum['date'] = pd.to_datetime(rum['date']).astype('datetime64[ns]')
    rum = rum.rename(columns={'date':'milkDate'})
    df = df.merge(rum, on=['bullNumber','milkDate'], how='left')
    
    df = df.sort_values(['bullNumber','milkDate']).reset_index(drop=True)
    df['milk_lag1'] = df.groupby('bullNumber', observed=True)['daily_milk_total'].shift(1)
    df['rumination_lag1'] = df.groupby('bullNumber', observed=True)['daily_rumination'].shift(1)
    df['activity_lag1'] = df.groupby('bullNumber', observed=True)['daily_activity'].shift(1)
    df['date_lag1'] = df.groupby('bullNumber', observed=True)['milkDate'].shift(1)
    df['lag1_gap'] = (df['milkDate'] - df['date_lag1']).dt.days
    for col in ['milk_lag1','rumination_lag1','activity_lag1']:
        df.loc[df['lag1_gap']>7, col] = np.nan
    df = df.drop(columns=['date_lag1','lag1_gap'])
    
    df['has_rumination'] = df['daily_rumination'].notna().astype(int)
    df['has_activity']   = df['daily_activity'].notna().astype(int)
    df['has_rum_lag1']   = df['rumination_lag1'].notna().astype(int)
    df['has_act_lag1']   = df['activity_lag1'].notna().astype(int)
    for col in ['daily_rumination','daily_activity','rumination_lag1','activity_lag1']:
        df[col] = df[col].fillna(0)
    
    train_mask = df['milkDate'] <= cfg['train_end']
    parity_dim_exp = df[train_mask].groupby(['parity_group','DIM_bin'], observed=True)['daily_milk_total'].agg(
        median='median', n='size').reset_index()
    parity_dim_exp = parity_dim_exp[parity_dim_exp['n']>=200].rename(columns={'median':'expected_milk'})
    df = df.merge(parity_dim_exp[['parity_group','DIM_bin','expected_milk']], on=['parity_group','DIM_bin'], how='left')
    df = df.dropna(subset=['expected_milk'])
    df['ratio'] = df['daily_milk_total'] / df['expected_milk']
    df['log_ratio'] = np.log(df['ratio'].clip(lower=0.01))
    df['ratio_lag1'] = df.groupby('bullNumber', observed=True)['ratio'].shift(1)
    
    thresholds = {}
    for p in sorted(df['parity_group'].unique()):
        sub = df[train_mask & (df['parity_group']==p)]['ratio']
        if len(sub) < 100: continue
        thresholds[p] = sub.mean() - 1.5*sub.std()
    df['threshold'] = df['parity_group'].map(thresholds)
    df['event'] = (df['ratio'] < df['threshold']).astype(int)
    
    train = df[df['milkDate']<=cfg['train_end']].dropna(subset=['milk_lag1','ratio_lag1','threshold']).copy()
    test = df[(df['milkDate']>=cfg['test_start'])&(df['milkDate']<=cfg['test_end'])].dropna(subset=['milk_lag1','ratio_lag1','threshold']).copy()
    
    if len(test)<500 or test['event'].sum()<30:
        return {'farm_id':fid, 'status':f'test_too_small ({len(test)}, ev={test["event"].sum()})'}
    
    print(f'  Train: {len(train):,} (rum {train["has_rumination"].mean()*100:.0f}%)', flush=True)
    print(f'  Test:  {len(test):,} (events {test["event"].sum():,}, rum {test["has_rumination"].mean()*100:.0f}%)', flush=True)
    
    val_idx = train.sample(frac=0.1, random_state=42).index
    
    def fit_eval(features):
        X_tr = train.drop(index=val_idx)[features]; y_tr = train.drop(index=val_idx)['log_ratio']
        X_val = train.loc[val_idx, features]; y_val = train.loc[val_idx, 'log_ratio']
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63,
            min_child_samples=200, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], callbacks=[lgb.early_stopping(200, verbose=False)])
        test_pred = np.exp(m.predict(test[features]))
        score = -test_pred; yt = test['event'].values
        ap = average_precision_score(yt, score); pos_rate = yt.mean()
        lift = ap / max(pos_rate, 0.001)
        thr_score = np.quantile(score, 0.9)
        top10 = ((score>=thr_score) & (yt==1)).sum() / max(1,yt.sum())
        test_local = test.copy(); test_local['pred'] = test_pred
        test_local = test_local.sort_values(['bullNumber','milkDate']).reset_index(drop=True)
        test_local['pred_lag3'] = test_local.groupby('bullNumber', observed=True)['pred'].shift(3)
        events_only = test_local[test_local['event']==1].dropna(subset=['pred_lag3'])
        baseline = test_local['pred'].mean()
        lead3 = float((events_only['pred_lag3'] < baseline*0.95).mean())
        return {'lift':lift, 'top10':top10, 'lead3':lead3, 'best_iter':m.best_iteration_}
    
    t0 = time.time()
    r31 = fit_eval(FEATURES_V31); print(f'  v3.1: Lift={r31["lift"]:.2f}x  Lead-3d={r31["lead3"]*100:.0f}%', flush=True)
    r4  = fit_eval(FEATURES_V4);  print(f'  v4:   Lift={r4["lift"]:.2f}x  Lead-3d={r4["lead3"]*100:.0f}%', flush=True)
    gain_pct = (r4['lift']-r31['lift'])/r31['lift']*100
    print(f'  Gain: Lift {gain_pct:+.1f}%, Lead-3d {(r4["lead3"]-r31["lead3"])*100:+.0f}pp  ({(time.time()-t0)/60:.1f}min)', flush=True)
    
    return {
        'farm_id':fid, 'status':'ok',
        'n_train':len(train), 'n_test':len(test),
        'test_rum_coverage': float(test['has_rumination'].mean()),
        'v31_lift':r31['lift'], 'v31_lead3d':r31['lead3'],
        'v4_lift':r4['lift'],   'v4_lead3d':r4['lead3'],
        'gain_pct':gain_pct,
    }


if __name__ == '__main__':
    print(f'Starting Step 2: 5 farms rumination sensitivity\n', flush=True)
    results = []
    for fid in FARMS:
        try:
            r = run_farm(fid)
            results.append(r)
            pd.DataFrame(results).to_csv(OUT_CSV, index=False)
        except Exception as e:
            print(f'NX{fid:02d} 错误: {str(e)[:150]}', flush=True)
            results.append({'farm_id':fid, 'status':f'error: {str(e)[:150]}'})
    
    print(f'\n\n{"="*70}\n5 场反刍 sensitivity 总结:\n{"="*70}', flush=True)
    ok = pd.DataFrame([r for r in results if r.get('status')=='ok'])
    if len(ok)>0:
        print(ok[['farm_id','n_test','test_rum_coverage','v31_lift','v4_lift','gain_pct','v31_lead3d','v4_lead3d']].round(3).to_string(index=False), flush=True)
        n_strong = (ok['gain_pct']>=10).sum()
        avg_gain = ok['gain_pct'].mean()
        print(f'\n  平均 gain: {avg_gain:.1f}%', flush=True)
        print(f'  ≥10% gain 场数: {n_strong}/{len(ok)}', flush=True)
    
    print(f'\n保存: {OUT_CSV}', flush=True)
