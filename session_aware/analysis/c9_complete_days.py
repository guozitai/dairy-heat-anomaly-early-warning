#!/usr/bin/env python3
"""C9. Task 1 restricted to cow-days with three usable milkings on day t and on day t+1 (deviation of the same precision on both days),
and the contribution of the features that describe the completeness of a cow's records. Same folds, lists and settings as C1.
Configurations: deviation rule; reference model refitted on the restricted data; the same without the completeness features (recorded and
usable milkings, recorded milkings relative to herd). Also, on the full risk set: reference model without the completeness features."""
import pandas as pd, numpy as np, time, warnings, sys, lightgbm as lgb
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; NT=6
PART=int(sys.argv[1]); NPARTS=int(sys.argv[2])
def P(*a): print(*a,flush=True)
L=np.log(0.8); L9=np.log(0.9); FARMS=[3,5,6,13,14,15,16,17]; LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
FOLDS=[(Y,f) for Y in (2024,2025) for f in FARMS]; MY=[x for i,x in enumerate(FOLDS) if i%NPARTS==PART]
YH=["D","D_l1","D_l2","D_l3","D_l4","D_l5","D_l6","ma3","min3","ma7","min7","sd7","slope","drops7","drops30","Dmin","loss_kg","lbase","my24","DIM","pg","n_sess","n_dev","sess_gap"]
REF=YH+["cond_dev","cond_dev_l1","condmax_dev","flow_dev","H","hdrop"]; NOC=[c for c in REF if c not in ("n_sess","n_dev","sess_gap")]
cols=list(dict.fromkeys(REF+["cid","farm_id","dnum","date","elig","D_next","D_next2","risk_onset","y_onset"]))
F=pd.read_parquet(V+"feat_v2.parquet",columns=cols); F=F[F.elig].sort_values(["cid","dnum"]).reset_index(drop=True); nxt=(F.cid.shift(-1)==F.cid)&(F.dnum.shift(-1)==F.dnum+1); F["nd_next"]=F.n_dev.shift(-1).where(nxt); F["year"]=F.date.dt.year
PAR=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=NT,seed=20261111)
def order(day,score,tb):
    o=np.lexsort((tb,-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); return o,st
def sel(o,st,n,k):
    cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); s=np.zeros(n,bool); s[o]=rank<np.repeat(np.ceil(cnt*k).clip(min=1),cnt); return s
def fit(tr,va,feats,rng):
    keep=(tr.y_onset.to_numpy()==1)|(rng.random(len(tr))<0.25); a=tr[keep]; kv=(va.y_onset.to_numpy()==1)|(rng.random(len(va))<0.25); v=va[kv]; ds=lgb.Dataset(a[feats],label=a.y_onset.astype(int),categorical_feature=["pg"]); dv=lgb.Dataset(v[feats],label=v.y_onset.astype(int),reference=ds,categorical_feature=["pg"])
    return lgb.train(PAR,ds,num_boost_round=1500,valid_sets=[dv],callbacks=[lgb.early_stopping(80,verbose=False)])
rows=[]
for (Y,f) in MY:
    teA=F[(F.farm_id==f)&(F.year==Y)]
    if len(teA)<30000: continue
    cut=pd.Timestamp(f"{Y-1}-12-31"); pool=F[(F.farm_id!=f)&(F.date<=cut)]; cows=np.sort(pool.cid.unique()); vc=np.random.default_rng([20261111,Y,f,0]).choice(cows,size=int(round(0.2*len(cows))),replace=False); isv=pool.cid.isin(vc).to_numpy(); trA=pool[~isv]; vaA=pool[isv]; tag=f"{LAB[f]} {Y}"; rng=np.random.default_rng([20261111,Y,f,9])
    # (a) full risk set, reference model without completeness features
    te=teA[teA.D.notna()&(teA.D>=L)].reset_index(drop=True); obs=te.D_next.notna().to_numpy(); y=(te.D_next.to_numpy()<L)&obs; ys=y&(te.D_next2.to_numpy()<L9); day=te.dnum.to_numpy(); tb=rng.random(len(te)); r=dict(fold=tag)
    if y.sum()>=200:
        m=fit(trA[trA.risk_onset],vaA[vaA.risk_onset],NOC,rng); S=sel(*order(day,m.predict(te[NOC],num_iteration=m.best_iteration),tb),len(te),0.10); Sr=sel(*order(day,-te.D.to_numpy(),tb),len(te),0.10)
        r.update(full_n=len(te),full_onsets=int(y.sum()),full_rule=Sr[y].mean(),full_model_nocompl=S[y].mean(),full_rule_conf=Sr[ys].mean(),full_model_nocompl_conf=S[ys].mean())
    # (b) three usable milkings on day t and t+1
    mk=lambda d: d.risk_onset&(d.n_dev==3)&(d.nd_next==3); t3=teA[teA.D.notna()&(teA.D>=L)&(teA.n_dev==3)].reset_index(drop=True); o3=t3.D_next.notna().to_numpy()&(t3.nd_next.to_numpy()==3); y3=(t3.D_next.to_numpy()<L)&o3; ys3=y3&(t3.D_next2.to_numpy()<L9); d3=t3.dnum.to_numpy(); tb3=rng.random(len(t3))
    if y3.sum()>=150:
        tr=trA[mk(trA)]; va=vaA[mk(vaA)]; Sr=sel(*order(d3,-t3.D.to_numpy(),tb3),len(t3),0.10); r.update(c3_n=len(t3),c3_obs=int(o3.sum()),c3_onsets=int(y3.sum()),c3_rate=y3.sum()/o3.sum(),c3_conf=int(ys3.sum()),c3_rule=Sr[y3].mean(),c3_rule_conf=Sr[ys3].mean() if ys3.sum() else np.nan,c3_train=len(tr))
        for nm,feats in (("ref",REF),("nocompl",NOC)):
            m=fit(tr,va,feats,rng); S=sel(*order(d3,m.predict(t3[feats],num_iteration=m.best_iteration),tb3),len(t3),0.10); r[f"c3_model_{nm}"]=S[y3].mean(); r[f"c3_model_{nm}_conf"]=S[ys3].mean() if ys3.sum() else np.nan
    rows.append(r); P({k:(round(v,4) if isinstance(v,float) else v) for k,v in r.items()},round(time.time()-t0)); pd.DataFrame(rows).to_csv(V+f"c1/complete_part{PART}.csv",index=False)
P("DONE",round(time.time()-t0))
