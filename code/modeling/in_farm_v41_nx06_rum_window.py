"""
FARM_1 v4.1: 反刍数据存在窗口内 (2022-2023)
  Train: 2022-03 ~ 2023-03 (反刍覆盖率高)
  Test:  2023-04 ~ 2023-09 (完整 2023 夏季, 反刍 NA 31%)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

CLEAN = '${DATA_ROOT}/derived/unified_daily_clean.parquet'
RUM   = '${DATA_ROOT}/derived/FARM_1_rumination_daily.parquet'

FARM = 6
TRAIN_END = '2023-03-31'
TEST_START = '2023-04-01'
TEST_END = '2023-09-30'

print(f'=== NX{FARM:02d} v4.1: 反刍窗口内 train/test ===\n')

# (与 v4 完全相同的 pipeline, 只改 TRAIN_END / TEST_START / TEST_END)

cols = ['farm_id','bullNumber','milkDate','daily_milk_total','thi_max',
        'thi_max_lag1','thi_max_lag3_avg','thi_max_lag7_cum','heatwave_3d',
        'hours_thi_gt_72','hours_thi_gt_79',
        'DIM','inferred_parity','qc_flag','yield_suspicious']
df = pd.read_parquet(CLEAN, columns=cols)
df = df[df['farm_id']==FARM]
df = df[(df['qc_flag']==False) & (~df['yield_suspicious'])]
df = df.dropna(subset=['daily_milk_total','thi_max','DIM','inferred_parity'])
df = df[df['DIM'].between(60,300)]
df['daily_milk_total'] = df['daily_milk_total'].astype('float64')
df['inferred_parity'] = df['inferred_parity'].astype('float64')
df['parity_group'] = df['inferred_parity'].clip(upper=4).astype(int)
df['DIM_bin'] = pd.cut(df['DIM'], bins=range(60,331,30), labels=False)
df['bullNumber'] = df['bullNumber'].astype(str)
df['milkDate'] = df['milkDate'].astype('datetime64[ns]')

# 限制时段在反刍数据窗口内
df = df[(df['milkDate']>='2022-01-01') & (df['milkDate']<='2024-06-07')]
print(f'FARM_1 (限制反刍窗口内): {len(df):,} 行')

rum = pd.read_parquet(RUM, columns=['bullNumber','date','daily_rumination','daily_activity'])
rum['bullNumber'] = rum['bullNumber'].astype(str)
rum['date'] = pd.to_datetime(rum['date']).astype('datetime64[ns]')
rum = rum.rename(columns={'date':'milkDate'})

df = df.merge(rum, on=['bullNumber','milkDate'], how='left')
print(f'Merge 后: {len(df):,} 行')
print(f'  daily_rumination 覆盖: {df["daily_rumination"].notna().mean()*100:.1f}%')
print(f'  daily_activity 覆盖:   {df["daily_activity"].notna().mean()*100:.1f}%')

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

train_mask = df['milkDate'] <= TRAIN_END
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

train = df[df['milkDate']<=TRAIN_END].dropna(subset=['milk_lag1','ratio_lag1','threshold']).copy()
test  = df[(df['milkDate']>=TEST_START)&(df['milkDate']<=TEST_END)].dropna(subset=['milk_lag1','ratio_lag1','threshold']).copy()

print(f'\nTrain: {len(train):,}  (rum 覆盖 {train["has_rumination"].mean()*100:.0f}%, act 覆盖 {train["has_activity"].mean()*100:.0f}%)')
print(f'Test:  {len(test):,}  (events {test["event"].sum():,}, {test["event"].mean()*100:.1f}%)')
print(f'        rum 覆盖 {test["has_rumination"].mean()*100:.0f}%, act 覆盖 {test["has_activity"].mean()*100:.0f}%')

if len(test)<1000 or test['event'].sum()<50:
    print('test 不足'); exit()

FEATURES_V31 = ['thi_max','thi_max_lag1','thi_max_lag3_avg','thi_max_lag7_cum',
                'heatwave_3d','hours_thi_gt_72','hours_thi_gt_79',
                'parity_group','DIM','milk_lag1','expected_milk','ratio_lag1']
FEATURES_V4 = FEATURES_V31 + ['daily_rumination','daily_activity','rumination_lag1','activity_lag1',
                              'has_rumination','has_activity','has_rum_lag1','has_act_lag1']

val_idx = train.sample(frac=0.1, random_state=42).index

def train_eval(features, label):
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
    print(f'\n=== {label} ===')
    print(f'  Lift={lift:.2f}x  Top10Rec={top10:.3f}  Lead-3d={lead3*100:.0f}%  best_iter={m.best_iteration_}')
    imp = pd.DataFrame({'f':features,'g':m.booster_.feature_importance('gain')}).sort_values('g',ascending=False)
    return ap, lift, lead3, imp

print('\n--- v3.1 (no rumination) ---')
ap_v31, lift_v31, lead_v31, imp_v31 = train_eval(FEATURES_V31, 'v3.1')

print('\n--- v4 (with rumination) ---')
ap_v4, lift_v4, lead_v4, imp_v4 = train_eval(FEATURES_V4, 'v4')
print('\nTop 10 features (v4):')
print(imp_v4.head(10).to_string(index=False))

gain_pct = (lift_v4 - lift_v31) / lift_v31 * 100
lead_gain = (lead_v4 - lead_v31) * 100
print(f'\n=== 总结 (反刍窗口内 train/test) ===')
print(f'  v3.1: Lift {lift_v31:.2f}x  Lead-3d {lead_v31*100:.0f}%')
print(f'  v4:   Lift {lift_v4:.2f}x  Lead-3d {lead_v4*100:.0f}%')
print(f'  Gain: Lift {gain_pct:+.1f}%  Lead-3d {lead_gain:+.0f}pp')
if gain_pct >= 10:
    print(f'\n  ★ 反刍真有效 (gain ≥ 10%), 论文升级 IF14 路径可考虑')
elif gain_pct >= 5:
    print(f'\n  反刍中等有效 (gain 5-10%), 反刍作 sensitivity in CEA')
else:
    print(f'\n  反刍无效 (gain < 5%), future work 一句话')
