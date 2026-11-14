#!/usr/bin/env python3
"""E1. Zero-shot time-series foundation model (Chronos-Bolt) as the yield expectation, on the 16 test farm-years of C1.
For each test cow-day t, the model forecasts the yield of day t from days t-28..t-1 and the yield of day t+1 from days t-27..t,
once with the daily sum of recorded milkings (y_sum) and once with the 24-h equivalent yield computed from intervals (my24).
Scores (all pre-specified, no fitting):
  anom_<s> = ln(s_t / q50_t|t-1)                  deviation of day t from the forecast (FM analogue of the deviation rule; lowest first)
  fc_<s>   = ln(q10_t+1|t / median(s, t-27..t))    forecast lower bound for day t+1 relative to the cow's recent level (lowest first)
Outputs per fold: flag composition on all test cow-days with a value of D, Task 1 sensitivity at 10% lists, Task 2 case sensitivity at 5% lists,
with the rule and gradient-boosting scores of C1 on the same rows. Usage: e1_fm.py <model> <part> <nparts>"""
import pandas as pd, numpy as np, time, sys, os, torch
from chronos import BaseChronosPipeline
t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; C1=V+"c1/"; OUT="/home/guozitai/20261112_aiia_dl/e1/"; os.makedirs(OUT,exist_ok=True)
MODEL=sys.argv[1]; PART=int(sys.argv[2]); NPARTS=int(sys.argv[3]); TAGM=MODEL.split("/")[-1]
def P(*a): print(*a,flush=True)
L=np.log(0.8); CTX=28; BS=8192
FARMS=[3,5,6,13,14,15,16,17]; LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
FOLDS=[(Y,f) for Y in (2024,2025) for f in FARMS]; MY=[x for i,x in enumerate(FOLDS) if i%NPARTS==PART]
pipe=BaseChronosPipeline.from_pretrained(MODEL,device_map="cuda",torch_dtype=torch.float32)
def order(day,score,tb):
    o=np.lexsort((tb,-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); return o,st
def sel_fixed(o,st,n,k):
    cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); kk=np.ceil(cnt*k).clip(min=1); s=np.zeros(n,bool); s[o]=rank<np.repeat(kk,cnt); return s
def fill(v):
    v=np.asarray(v,float); ok=np.isfinite(v); return np.where(ok,v,np.nanmedian(v[ok]) if ok.any() else 0.0)
def forecast(X):
    q=np.full((len(X),2),np.nan,np.float32); has=np.isfinite(X).sum(1)>=7
    idx=np.flatnonzero(has)
    for i in range(0,len(idx),BS):
        b=idx[i:i+BS]; qq,_=pipe.predict_quantiles(torch.tensor(X[b]),prediction_length=1,quantile_levels=[0.1,0.5]); q[b]=qq[:,0,:].numpy()
    return q
cols=["cid","dnum","farm_id","date","y_sum","my24","D","dev_raw","sess_gap","elig"]
rows1=[];rows2=[];comp=[]
for (Y,f) in MY:
    p1=C1+f"pred1_{f}_{Y}.parquet"; p2=C1+f"pred2_{f}_{Y}.parquet"
    if not os.path.exists(p1): continue
    tag=f"{LAB[f]} {Y}"; A=pd.read_parquet(V+"feat_v2.parquet",columns=cols,filters=[("farm_id","==",f)]).sort_values(["cid","dnum"]).reset_index(drop=True)
    key=A.cid.to_numpy().astype(np.int64)*100000+A.dnum.to_numpy().astype(np.int64)
    T=A[(A.date.dt.year==Y)&A.elig&A.D.notna()].reset_index(drop=True); tk=T.cid.to_numpy().astype(np.int64)*100000+T.dnum.to_numpy().astype(np.int64)
    def lagmat(series,lags):
        v=A[series].to_numpy(np.float32); M=np.full((len(tk),len(lags)),np.nan,np.float32)
        for j,k in enumerate(lags):
            q=tk-k; pos=np.searchsorted(key,q).clip(max=len(key)-1); ok=key[pos]==q; M[ok,j]=v[pos[ok]]
        return M
    S={}
    for s in ("y_sum","my24"):
        Xa=lagmat(s,range(CTX,0,-1)); Xb=lagmat(s,range(CTX-1,-1,-1)); qa=forecast(Xa); qb=forecast(Xb); cur=T[s].to_numpy(np.float32)
        with np.errstate(all="ignore"):
            S["anom_"+s]=np.log(cur/qa[:,1]); S["fc_"+s]=np.log(qb[:,0]/np.nanmedian(Xb,axis=1)); S["fc50_"+s]=np.log(qb[:,1]/np.nanmedian(Xb,axis=1))
        P(f"[{tag}] {s} forecasts done n={len(T):,} [{round(time.time()-t0)}s]")
    Sdf=pd.DataFrame({"cid":T.cid,"dnum":T.dnum,**S}); Sdf.to_parquet(OUT+f"scores_{TAGM}_{f}_{Y}.parquet",index=False)
    # flag composition on all test cow-days with D
    inc=T.sess_gap.to_numpy()>0; dD=T.D.to_numpy()<L
    for nm,v in (("D",T.D.to_numpy()),("dailysum_deviation",T.dev_raw.to_numpy()),("anom_y_sum",S["anom_y_sum"]),("anom_my24",S["anom_my24"])):
        ok=np.isfinite(v); fl=ok&(v<L); comp.append(dict(fold=tag,score=nm,n=int(ok.sum()),flag_rate=fl.sum()/ok.sum(),flags_on_incomplete=inc[fl].mean(),incomplete_share=inc[ok].mean(),flags_with_D_drop=dD[fl].mean()))
    # Task 1
    R=pd.read_parquet(p1).merge(Sdf,on=["cid","dnum"],how="left"); n=len(R); day=R.dnum.to_numpy(); tb=np.random.default_rng([20261112,Y,f,1]).random(n); y=R.y.to_numpy(); obs=R.obs.to_numpy()
    sc={"R_deviation":R.rule.to_numpy(),"M_ref":R.model.to_numpy(),"M_ref+weather":R.model_weather.to_numpy()}
    for c in S: sc["FM_"+c]=-fill(R[c])
    for nm,s in sc.items():
        S10=sel_fixed(*order(day,s,tb),n,0.10); rows1.append(dict(fold=tag,farm=LAB[f],year=Y,config=nm,events=int(y.sum()),sens10=S10[y].mean(),missing=float(np.isnan(R[nm[3:]]).mean()) if nm.startswith("FM_") else 0.0))
    # Task 2
    R=pd.read_parquet(p2).merge(Sdf,on=["cid","dnum"],how="left"); n=len(R); day=R.dnum.to_numpy(); tb=np.random.default_rng([20261112,Y,f,2]).random(n); y=R.y.to_numpy()==1; dnd=R.dn_dis.to_numpy()
    pos=np.flatnonzero(y); case=R.cid.to_numpy().astype(np.int64)[pos]*100000+(day[pos]+dnd[pos]).astype(np.int64); cu,ci=np.unique(case,return_inverse=True)
    sc={"R_deviation":R.rule.to_numpy(),"M_ref":R.model.to_numpy(),"M_ref+weather":R.model_weather.to_numpy()}
    for c in S: sc["FM_"+c]=-fill(R[c])
    for nm,s in sc.items():
        o,st=order(day,s,tb); r=dict(fold=tag,farm=LAB[f],year=Y,config=nm,cases=len(cu))
        for k in (0.05,0.10):
            Sk=sel_fixed(o,st,n,k); h=np.zeros(len(cu)); np.maximum.at(h,ci,Sk[pos].astype(float)); r[f"case_sens{int(k*100)}"]=h.mean()
        rows2.append(r)
    g=lambda rows,nm,c: [x for x in rows if x["fold"]==tag and x["config"]==nm][0][c]
    P(f"[{tag}] T1 sens10: rule {g(rows1,'R_deviation','sens10'):.3f} model {g(rows1,'M_ref','sens10'):.3f} | FM anom sum {g(rows1,'FM_anom_y_sum','sens10'):.3f} my24 {g(rows1,'FM_anom_my24','sens10'):.3f} | FM fc sum {g(rows1,'FM_fc_y_sum','sens10'):.3f} my24 {g(rows1,'FM_fc_my24','sens10'):.3f}")
    P(f"[{tag}] T2 case5: rule {g(rows2,'R_deviation','case_sens5'):.3f} model {g(rows2,'M_ref','case_sens5'):.3f} | FM anom sum {g(rows2,'FM_anom_y_sum','case_sens5'):.3f} my24 {g(rows2,'FM_anom_my24','case_sens5'):.3f} | FM fc sum {g(rows2,'FM_fc_y_sum','case_sens5'):.3f} my24 {g(rows2,'FM_fc_my24','case_sens5'):.3f}")
    for c in comp[-4:]: P(f"[{tag}] flags {c['score']}: rate {c['flag_rate']:.3f} on-incomplete {c['flags_on_incomplete']:.3f} (base {c['incomplete_share']:.3f}) with-D-drop {c['flags_with_D_drop']:.3f}")
    for nm,x in (("t1",rows1),("t2",rows2),("comp",comp)): pd.DataFrame(x).to_csv(OUT+f"{nm}_{TAGM}_part{PART}.csv",index=False)
P("DONE",round(time.time()-t0))
