#!/usr/bin/env python3
"""B9. The model submitted in the earlier version, evaluated on the cow-days for which its predictions exist (its own test folds),
against the same outcomes and with all scores ranked within the same set of cows."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","cow","date","dnum","D","dev_raw","y_sum","DIM","pg","elig","risk_dis","y_dis","dn_dis","risk_onset","y_onset"]); F=F[F.elig]; F["year"]=F.date.dt.year
O=pd.read_parquet(D0+"E1_time_anchored_loso_regression_predictions.parquet",columns=["farm_id","cow_id","day","risk_score","event"]).rename(columns={"cow_id":"cow"}); O["cow"]=O.cow.astype(str); O["date"]=pd.to_datetime(O.day)-pd.Timedelta(days=1)
F=F.merge(O[["farm_id","cow","date","risk_score","event"]],on=["farm_id","cow","date"],how="inner")
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}; rng=np.random.default_rng(1)
def lists(day,score,k):
    o=np.lexsort((-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); n=len(day); cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); s=np.zeros(n,bool); s[o]=rank<np.repeat(np.ceil(cnt*k).clip(min=1),cnt); return s
rows=[]
for f,g in F.groupby("farm_id"):
    te=g[g.risk_dis].reset_index(drop=True)
    if te.y_dis.sum()<30: continue
    y=te.y_dis.to_numpy(); day=te.dnum.to_numpy(); pos=np.flatnonzero(y==1); case=te.cid.to_numpy().astype(np.int64)[pos]*100000+(te.dnum.to_numpy()[pos]+te.dn_dis.to_numpy()[pos]).astype(np.int64); cu,ci=np.unique(case,return_inverse=True)
    def cs(S):
        h=np.zeros(len(cu)); np.maximum.at(h,ci,S[pos].astype(float)); return h.mean()
    sc={"random":rng.random(len(te)),"submitted model":te.risk_score.to_numpy(),"session-aware deviation":-te.D.fillna(te.D.median()).to_numpy(),"daily-sum deviation":-te.dev_raw.fillna(te.dev_raw.median()).to_numpy()}
    r=dict(farm=LAB[f],period=f"{te.date.min().date()}..{te.date.max().date()}",cowdays=len(te),cases=len(cu))
    for nm,s in sc.items():
        for k in (0.05,0.10): r[f"{nm}@{int(k*100)}"]=cs(lists(day,s,k))
    t1=g[g.risk_onset]; y1=t1.y_onset.to_numpy(); d1=t1.dnum.to_numpy()
    r["onset lift: submitted"]=lists(d1,t1.risk_score.to_numpy(),0.10)[y1==1].mean()*10; r["onset lift: deviation rule"]=lists(d1,-t1.D.to_numpy(),0.10)[y1==1].mean()*10
    # how the submitted model ranked its own label on these rows (sanity: reproduces the earlier claim)
    ev=g.dropna(subset=["event"]); r["own label lift"]=lists(ev.dnum.to_numpy(),ev.risk_score.to_numpy(),0.10)[ev.event.to_numpy()==1].mean()*10
    rows.append(r)
R=pd.DataFrame(rows); pd.set_option("display.width",250); print(R.round(3).to_string()); print("\nmeans:"); print(R.drop(columns=["farm","period"]).mean().round(3).to_string()); R.to_csv(V+"submitted_model_eval.csv",index=False)
