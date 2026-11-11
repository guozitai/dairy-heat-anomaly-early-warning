#!/usr/bin/env python3
"""B3x. Robustness of the herd-level response to heat load (May-September).
Standard errors under alternative clustering (farm-month; calendar date; two-way; calendar week), placebo with the THI of future days,
distributed lags and leads, estimates by year, alternative exposure definitions, lagged reference window, complete cow-days only,
and completeness of recording as an outcome. Output: tab3_robust.csv and tab3_dlag.csv."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250)
fd=pd.read_parquet(V+"farmday_v2.parquet"); lr=pd.read_parquet(V+"farmday_lagref.parquet"); fd=fd.merge(lr,on=["farm_id","loc","date"],how="left")
fd["HL"]=fd.HL.where(fd.nL>=200); fd["H3"]=fd.H3.where(fd.n3>=100)
T=pd.read_csv(V+"era5_farm_daily.csv",parse_dates=["date"]).drop(columns=["n"]).sort_values(["loc","date"]); g=T.groupby("loc"); X=T[["loc","date"]].copy()
for k in range(3,11): X[f"L{k}"]=g.tmax.shift(k)
for k in range(1,5): X[f"F{k}"]=g.tmax.shift(-k)
X["tmax7p"]=g.tmax.transform(lambda s:s.shift(1).rolling(7,min_periods=7).mean()); X["tmean2p"]=g.tmean.transform(lambda s:s.shift(1).rolling(2,min_periods=2).mean()); X["h72_2p"]=g.h72.transform(lambda s:s.shift(1).rolling(2,min_periods=2).mean()); X["tairmax2p"]=g.tair_max.transform(lambda s:s.shift(1).rolling(2,min_periods=2).mean())
X["lead23"]=(X.F2+X.F3)/2
fd=fd.merge(X,on=["loc","date"],how="left"); fd["L0"]=fd.tmax; fd["L1"]=fd.tmax_l1; fd["L2"]=fd.tmax_l2
fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d"); fd["wkey"]=fd.date.dt.strftime("%G-%V"); fd["yr"]=fd.date.dt.year
warm=fd[fd.m.isin([5,6,7,8,9])].copy(); P("warm-season farm-days",len(warm))
def absorb(v,keys,it=12):
    v=np.asarray(v,float).copy()
    for _ in range(it):
        for k in keys: v=v-pd.Series(v).groupby(k).transform("mean").values
    return v
def fe(df,y,xs,cls=("fym","dkey","wkey")):
    z=df.dropna(subset=[y]+xs); K=[z["fym"].values]; yd=absorb(z[y].values,K); Xm=np.column_stack([absorb(z[x].values,K) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm)
    def vc(G):
        meat=np.zeros((Xm.shape[1],)*2); sc=pd.DataFrame(Xm*e[:,None]).groupby(G).sum().values
        return XtXi@(sc.T@sc)@XtXi
    Vs={c:vc(z[c].values) for c in cls}; hc=XtXi@((Xm*e[:,None]).T@(Xm*e[:,None]))@XtXi; Vs["twoway"]=Vs["fym"]+Vs["dkey"]-hc
    se={k:np.sqrt(np.clip(np.diag(v),0,None)) for k,v in Vs.items()}; return b,se,len(z),{k:z[k].nunique() for k in cls}
R=[]
def add(label,df,y,x,tau,scale=100,extra=None):
    d=df.copy(); d["h"]=np.maximum(d[x]-tau,0) if tau is not None else d[x]; xs=["h"]+(extra or []); b,se,n,nc=fe(d,y,xs)
    R.append(dict(analysis=label,outcome=y,exposure=x,tau=tau,n=n,slope=b[0]*scale,se_fm=se["fym"][0]*scale,se_date=se["dkey"][0]*scale,se_week=se["wkey"][0]*scale,se_twoway=se["twoway"][0]*scale,clusters_fm=nc["fym"],clusters_date=nc["dkey"],clusters_week=nc["wkey"]))
    P(f"{label:58s} {y:5s} {x:9s} tau={tau}: {b[0]*scale:+.3f}  SE farm-month {se['fym'][0]*scale:.3f} | date {se['dkey'][0]*scale:.3f} | week {se['wkey'][0]*scale:.3f} | two-way {se['twoway'][0]*scale:.3f}  (n={n}; clusters {nc})"); return b,se
P("\n== main estimates under alternative clustering ==")
for tau in (72,76,79): add("main",warm,"H","tmax2p",tau)
for tau in (76,): add("share in acute drop",warm,"drop","tmax2p",tau); add("onset rate",warm,"onset","tmax2p",tau); add("daily-sum measure",warm,"Hraw","tmax2p",tau)
P("\n== class means (THI of previous day) with two-way clustered SE ==")
bins=[0,68,72,75,77,79,81,90]; z=warm.dropna(subset=["H","tmax_l1"]).copy(); z["b"]=pd.cut(z.tmax_l1,bins).astype(str)
for lagc in ("tmax","tmax_l1","tmax_l2"):
    z=warm.dropna(subset=["H",lagc]).copy(); z["b"]=pd.cut(z[lagc],bins).astype(str); names=sorted(z.b.unique(),key=lambda s:float(s.split(",")[0][1:])); ref=names[1]
    for nm in names: z["I_"+nm]=(z.b==nm).astype(float)
    xs=["I_"+nm for nm in names if nm!=names[0]]; b,se,n,nc=fe(z,"H",xs)
    P(f"  {lagc}: relative to class {names[0]}: "+" | ".join(f"{nm}: {bb*100:+.2f} (fm {s1*100:.2f}; 2way {s2*100:.2f})" for nm,bb,s1,s2 in zip(names[1:],b,se['fym'],se['twoway'])))
    for nm,bb,s1,s2,s3 in zip(names[1:],b,se["fym"],se["twoway"],se["dkey"]): R.append(dict(analysis="class contrast vs <=68",outcome="H",exposure=lagc,tau=nm,n=n,slope=bb*100,se_fm=s1*100,se_twoway=s2*100,se_date=s3*100))
P("\n== placebo: THI of future days (mean of days d+2 and d+3), jointly with the exposure of the two preceding days ==")
for tau in (76,79):
    d=warm.copy(); d["h"]=np.maximum(d.tmax2p-tau,0); d["hf"]=np.maximum(d.lead23-tau,0); b,se,n,nc=fe(d,"H",["h","hf"])
    P(f"   tau={tau}: past {b[0]*100:+.3f} (fm {se['fym'][0]*100:.3f}; 2way {se['twoway'][0]*100:.3f}) | future {b[1]*100:+.3f} (fm {se['fym'][1]*100:.3f}; 2way {se['twoway'][1]*100:.3f}) n={n}")
    R.append(dict(analysis="placebo: future THI (d+2, d+3), joint",outcome="H",exposure="lead23",tau=tau,n=n,slope=b[1]*100,se_fm=se["fym"][1]*100,se_twoway=se["twoway"][1]*100,se_date=se["dkey"][1]*100)); R.append(dict(analysis="past exposure in joint model with future THI",outcome="H",exposure="tmax2p",tau=tau,n=n,slope=b[0]*100,se_fm=se["fym"][0]*100,se_twoway=se["twoway"][0]*100,se_date=se["dkey"][0]*100))
P("\n== distributed lags and leads (hinge at 76), joint model ==")
DL=[]
for y in ("H","HL"):
    d=warm.copy(); cols=[]
    for k in (4,3,2,1): d[f"x_F{k}"]=np.maximum(d[f"F{k}"]-76,0); cols.append(f"x_F{k}")
    for k in range(0,11): d[f"x_L{k}"]=np.maximum(d[f"L{k}"]-76,0); cols.append(f"x_L{k}")
    b,se,n,nc=fe(d,y,cols)
    for cc,bb,s1,s2 in zip(cols,b,se["fym"],se["twoway"]): DL.append(dict(outcome=y,term=cc,day=(-int(cc[3:]) if cc[2]=="F" else int(cc[3:])),coef=bb*100,se_fm=s1*100,se_twoway=s2*100,n=n))
    P(f"  {y}: "+" ".join(f"{cc[2:]}:{bb*100:+.2f}({s2*100:.2f})" for cc,bb,s2 in zip(cols,b,se['twoway']))); P(f"     sum of lags 0-3: {b[4:8].sum()*100:+.3f} ; lags 0-10: {b[4:].sum()*100:+.3f} ; leads: {b[:4].sum()*100:+.3f}")
pd.DataFrame(DL).to_csv(V+"tab3_dlag.csv",index=False)
P("\n== by year ==")
for yv,s in warm.groupby("yr"): add(f"year {yv}",s,"H","tmax2p",76)
P("\n== alternative exposure definitions (two preceding days) ==")
for tau in (66,68,70): add("daily mean THI",warm,"H","tmean2p",tau)
add("hours with THI above 72 (per hour)",warm,"H","h72_2p",None)
for tau in (28,30,32): add("daily maximum air temperature (per deg C)",warm,"H","tairmax2p",tau)
P("\n== lagged reference window (days t-21..t-9) ==")
for tau in (72,76,79): add("lagged reference, 2 preceding days",warm,"HL","tmax2p",tau)
for tau in (72,76,79): add("lagged reference, 7 preceding days",warm,"HL","tmax7p",tau)
for tau in (72,76,79): add("standard reference, 7 preceding days",warm,"H","tmax7p",tau)
for y in ("H","HL"):
    for lagc in ("tmax_l1","tmax_l2"):
        z=warm.dropna(subset=[y,lagc]).copy(); z["yd"]=z[y]-z.groupby("fym")[y].transform("mean"); z["b"]=pd.cut(z[lagc],bins); r=z.groupby("b",observed=True).yd.agg(["size","mean"]); P(f"  {y} by class of {lagc}: "+" | ".join(f"{str(i)}: {m*100:+.2f} (n={n})" for i,n,m in zip(r.index,r['size'],r['mean'])))
P("   SD of farm-month-demeaned HL: %.4f ; H: %.4f"%((warm.HL-warm.groupby("fym").HL.transform("mean")).std(),(warm.H-warm.groupby("fym").H.transform("mean")).std()))
P("\n== cow-days with three usable milkings only; completeness of recording as outcome ==")
for tau in (76,79): add("three usable milkings only",warm,"H3","tmax2p",tau)
add("share of cow-days with three recorded milkings (pp)",warm,"c3","tmax2p",76); add("number of cows with a day deviation (per cent of mean)",warm.assign(nrel=warm.n/warm.groupby("farm_id").n.transform("mean")),"nrel","tmax2p",76)
pd.DataFrame(R).to_csv(V+"tab3_robust.csv",index=False); P("DONE",round(time.time()-t0))
