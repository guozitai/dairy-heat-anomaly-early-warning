#!/usr/bin/env python3
"""B3z. Checks of the herd-level response to heat load: joint model with the THI of the following and preceding days,
standard errors clustered by farm-month and by date, and estimates by period."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"
fd=pd.read_parquet(V+"farmday_v2.parquet").merge(pd.read_parquet(V+"farmday_lagref.parquet"),on=["farm_id","loc","date"],how="left"); fd["HL"]=fd.HL.where(fd.nL>=200)
T=pd.read_csv(V+"era5_farm_daily.csv",parse_dates=["date"]).drop(columns=["n"]).sort_values(["loc","date"]); g=T.groupby("loc"); X=T[["loc","date"]].copy()
for k in range(3,11): X[f"L{k}"]=g.tmax.shift(k)
for k in range(1,5): X[f"F{k}"]=g.tmax.shift(-k)
fd=fd.merge(X,on=["loc","date"],how="left"); fd["L0"]=fd.tmax; fd["L1"]=fd.tmax_l1; fd["L2"]=fd.tmax_l2; fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d")
warm=fd[fd.m.isin([5,6,7,8,9])].copy()
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]
    def vc(G): sc=pd.DataFrame(S).groupby(G).sum().values; return XtXi@(sc.T@sc)@XtXi
    Vf=vc(z["fym"].values); Vd=vc(z["dkey"].values); V2=Vf+Vd-XtXi@(S.T@S)@XtXi; return b,np.sqrt(np.diag(Vf)),np.sqrt(np.clip(np.diag(V2),0,None)),len(z),Vf
bins=[0,68,72,75,77,79,81,90]
print("== completeness (share of cow-days with three recorded milkings, pp) and cows with D (relative), by class of THI ==")
warm["nrel"]=warm.n/warm.groupby("farm_id").n.transform("mean")
for y in ("c3","nrel"):
    for lagc in ("tmax","tmax_l1"):
        z=warm.dropna(subset=[y,lagc]).copy(); z["b"]=pd.cut(z[lagc],bins).astype(str); names=[str(i) for i in pd.cut(pd.Series([1.0]),bins).cat.categories]; xs=[]
        for j,nm in enumerate(names[1:]): z[f"I{j}"]=(z.b==nm).astype(float); xs.append(f"I{j}")
        b,se,se2,n,_=fe(z,y,xs); print(f"  {y} vs {lagc}: "+" | ".join(f"{nm}: {bb*100:+.2f} ({s*100:.2f})" for nm,bb,s in zip(names[1:],b,se)))
print("warm-season mean share of cow-days with three recorded milkings: %.3f ; SD within farm-month %.3f"%(warm.c3.mean(),(warm.c3-warm.groupby("fym").c3.transform("mean")).std()))
print("== joint lead-lag, alternative specifications ==")
for tau in (72,76):
    d=warm.copy(); cols=[]
    for k in (4,3,2,1): d[f"x_F{k}"]=np.maximum(d[f"F{k}"]-tau,0); cols.append(f"x_F{k}")
    for k in range(0,11): d[f"x_L{k}"]=np.maximum(d[f"L{k}"]-tau,0); cols.append(f"x_L{k}")
    b,se,se2,n,Vf=fe(d,"H",cols); w=np.zeros(len(cols)); w[:4]=1; sl=w@b; ssl=np.sqrt(w@Vf@w); w2=np.zeros(len(cols)); w2[5:10]=1
    print(f"  tau={tau}: "+" ".join(f"{cc[2:]}:{bb*100:+.2f}({s1*100:.2f})" for cc,bb,s1 in zip(cols,b,se))); print(f"     sum of leads {sl*100:+.3f} (SE {ssl*100:.3f}); sum of lags 1-5 {(w2@b)*100:+.3f} (SE {np.sqrt(w2@Vf@w2)*100:.3f})")
d=warm.copy(); cols=[]
for k in (4,3,2,1):
    d[f"a_F{k}"]=np.clip(d[f"F{k}"]-72,0,7); d[f"b_F{k}"]=np.maximum(d[f"F{k}"]-79,0); cols+= [f"a_F{k}",f"b_F{k}"]
for k in range(0,11):
    d[f"a_L{k}"]=np.clip(d[f"L{k}"]-72,0,7); d[f"b_L{k}"]=np.maximum(d[f"L{k}"]-79,0); cols+=[f"a_L{k}",f"b_L{k}"]
b,se,se2,n,Vf=fe(d,"H",cols)
print("  two segments (72-79 and above 79): "+" ".join(f"{cc}:{bb*100:+.2f}({s1*100:.2f})" for cc,bb,s1 in zip(cols,b,se)))
w=np.zeros(len(cols)); w[[1,3,5,7]]=1; print("     sum of leads, segment above 79: %+.3f (SE %.3f)"%(w@b*100,np.sqrt(w@Vf@w)*100)); w=np.zeros(len(cols)); w[[0,2,4,6]]=1; print("     sum of leads, segment 72-79: %+.3f (SE %.3f)"%(w@b*100,np.sqrt(w@Vf@w)*100))
print("== base rates, warm season ==")
print("mean share of cows in acute drop %.4f ; mean onset rate %.4f ; mean H %.4f ; farm-days %d"%(warm["drop"].mean(),warm.onset.mean(),warm.H.mean(),len(warm)))
print("farm-days by class of tmax_l1:",pd.cut(warm.tmax_l1,bins).value_counts().sort_index().to_dict())
print("7-day mean exposure: farm-days with tmax7p>76: %d ; >79: %d"%((warm.tmax.rolling(1).mean().notna()&(g.tmax.transform(lambda s:s.shift(1).rolling(7,min_periods=7).mean()).reindex(T.index).pipe(lambda s: fd.merge(T.assign(t7=s)[["loc","date","t7"]],on=["loc","date"],how="left").loc[warm.index,"t7"])>76)).sum(),(fd.merge(T.assign(t7=g.tmax.transform(lambda s:s.shift(1).rolling(7,min_periods=7).mean()))[["loc","date","t7"]],on=["loc","date"],how="left").loc[warm.index,"t7"]>79).sum()))
