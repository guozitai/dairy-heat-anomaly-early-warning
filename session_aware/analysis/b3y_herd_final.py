#!/usr/bin/env python3
"""B3y. Herd-level response to heat load, final estimates (May-September).
All class effects are within-estimates: regression of the herd deviation on indicators of THI class with farm x year-month fixed effects
(reference class: daily maximum THI <= 68), standard errors clustered by farm-month (and two-way by farm-month and date).
Also: joint model with leads and lags, forecast-announced hot days as a within-estimate, estimates by year.
Outputs: fig3_dose_response.csv (replaces the descriptive version), tab3_forecast_announced.csv, tab3_dlag.csv, tab3_byyear.csv."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250)
fd=pd.read_parquet(V+"farmday_v2.parquet").merge(pd.read_parquet(V+"farmday_lagref.parquet"),on=["farm_id","loc","date"],how="left")
fd["HL"]=fd.HL.where(fd.nL>=200); fd["H3"]=fd.H3.where(fd.n3>=100)
T=pd.read_csv(V+"era5_farm_daily.csv",parse_dates=["date"]).drop(columns=["n"]).sort_values(["loc","date"]); g=T.groupby("loc"); X=T[["loc","date"]].copy()
for k in range(3,11): X[f"L{k}"]=g.tmax.shift(k)
for k in range(1,5): X[f"F{k}"]=g.tmax.shift(-k)
fd=fd.merge(X,on=["loc","date"],how="left"); fd["L0"]=fd.tmax; fd["L1"]=fd.tmax_l1; fd["L2"]=fd.tmax_l2; fd["tmax_l3"]=fd.L3
fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d"); fd["yr"]=fd.date.dt.year
warm=fd[fd.m.isin([5,6,7,8,9])].copy(); P("warm-season farm-days",len(warm))
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]
    def vc(G): sc=pd.DataFrame(S).groupby(G).sum().values; return XtXi@(sc.T@sc)@XtXi
    Vf=vc(z["fym"].values); Vd=vc(z["dkey"].values); V2=Vf+Vd-XtXi@(S.T@S)@XtXi
    return b,np.sqrt(np.diag(Vf)),np.sqrt(np.clip(np.diag(V2),0,None)),len(z),z
bins=[0,68,72,75,77,79,81,90]; rows=[]
for y in ("H","Hraw","drop","onset","HL","H3"):
    for lagc in ("tmax","tmax_l1","tmax_l2","tmax_l3"):
        z=warm.dropna(subset=[y,lagc]).copy(); z["b"]=pd.cut(z[lagc],bins).astype(str); names=[str(i) for i in pd.cut(pd.Series([1.0]),bins).cat.categories]; xs=[]
        for j,nm in enumerate(names[1:]): z[f"I{j}"]=(z.b==nm).astype(float); xs.append(f"I{j}")
        b,se,se2,n,zz=fe(z,y,xs); cnt=z.b.value_counts()
        rows.append(dict(outcome=y,lag=lagc,b=names[0],size=int(cnt.get(names[0],0)),mean=0.0,sem=0.0,sem_twoway=0.0))
        for nm,bb,s1,s2 in zip(names[1:],b,se,se2): rows.append(dict(outcome=y,lag=lagc,b=nm,size=int(cnt.get(nm,0)),mean=bb,sem=s1,sem_twoway=s2))
B=pd.DataFrame(rows); B.to_csv(V+"fig3_dose_response.csv",index=False)
for y in ("H","Hraw","drop","onset","HL","H3"):
    for x in ("tmax","tmax_l1","tmax_l2"):
        r=B[(B.outcome==y)&(B.lag==x)]; P(f"{y:5s} vs {x:8s}: "+" | ".join(f"{b}: {m*100:+.2f} ({s*100:.2f}/{s2*100:.2f}; n={n})" for b,n,m,s,s2 in zip(r.b,r["size"],r["mean"],r["sem"],r["sem_twoway"])))
P("\n== joint model with leads (d+1..d+4) and lags (d..d-10) ==")
DL=[]
for tau in (76,79):
    for y in ("H","HL"):
        d=warm.copy(); cols=[]
        for k in (4,3,2,1): d[f"x_F{k}"]=np.maximum(d[f"F{k}"]-tau,0); cols.append(f"x_F{k}")
        for k in range(0,11): d[f"x_L{k}"]=np.maximum(d[f"L{k}"]-tau,0); cols.append(f"x_L{k}")
        b,se,se2,n,zz=fe(d,y,cols)
        for cc,bb,s1,s2 in zip(cols,b,se,se2): DL.append(dict(tau=tau,outcome=y,term=cc,day=(-int(cc[3:]) if cc[2]=="F" else int(cc[3:])),coef=bb*100,se_fm=s1*100,se_twoway=s2*100,n=n))
        P(f"  tau={tau} {y}: "+" ".join(f"{cc[2:]}:{bb*100:+.2f}({s1*100:.2f})" for cc,bb,s1 in zip(cols,b,se))); P(f"     sum lags 0-7: {b[4:12].sum()*100:+.3f} ; lags 1-5: {b[5:10].sum()*100:+.3f} ; leads: {b[:4].sum()*100:+.3f}")
pd.DataFrame(DL).to_csv(V+"tab3_dlag.csv",index=False)
P("\n== by year (slope per unit of mean daily maximum THI of the two preceding days above 76) ==")
BY=[]
for yv,s in warm.groupby("yr"):
    s=s.copy(); s["h"]=np.maximum(s.tmax2p-76,0); b,se,se2,n,zz=fe(s,"H",["h"]); BY.append(dict(year=yv,farms=s.farm_id.nunique(),farm_months=s.fym.nunique(),farm_days=n,days_above76=int((s.tmax2p>76).sum()),days_above79=int((s.tmax2p>79).sum()),slope=b[0]*100,se=se[0]*100,se_twoway=se2[0]*100))
BY=pd.DataFrame(BY); BY.to_csv(V+"tab3_byyear.csv",index=False); P(BY.round(3).to_string())
P(warm.groupby(["yr","farm"]).size().unstack().to_string())
P("\n== forecast-announced hot days: within-estimate of the herd deviation over the two following days ==")
FC=pd.read_csv(V+"fcst_farm_daily.csv",parse_dates=["date"]); FS=pd.read_csv(V+"fig3_forecast_skill.csv").set_index("lead")
h=fd[["farm_id","date","H"]].copy(); h1=h.assign(date=h.date-pd.Timedelta(days=1)).rename(columns={"H":"Hp1"}); h2=h.assign(date=h.date-pd.Timedelta(days=2)).rename(columns={"H":"Hp2"})
f2=fd.merge(h1,on=["farm_id","date"],how="left").merge(h2,on=["farm_id","date"],how="left").merge(FC,on=["loc","date"],how="inner"); f2=f2[f2.m.isin([5,6,7,8,9])].copy(); f2["Hn"]=(f2.Hp1+f2.Hp2)/2
AN=[]
f2["x"]=(f2.tmax>=79).astype(float); b,se,se2,n,zz=fe(f2.dropna(subset=["fc_tmax_l1"]),"Hn",["x"]); AN.append(dict(lead=0,n=n,n_hot=int(zz.x.sum()),diff=b[0],se=se[0],se_twoway=se2[0]))
for k in range(1,8):
    z=f2.dropna(subset=[f"fc_tmax_l{k}"]).copy(); z["x"]=(z[f"fc_tmax_l{k}"]-FS.loc[k,"bias"]>=79).astype(float); b,se,se2,n,zz=fe(z,"Hn",["x"]); AN.append(dict(lead=k,n=n,n_hot=int(zz.x.sum()),diff=b[0],se=se[0],se_twoway=se2[0]))
AN=pd.DataFrame(AN); AN.to_csv(V+"tab3_forecast_announced.csv",index=False); P(AN.assign(diff=AN["diff"]*100,se=AN.se*100,se_twoway=AN.se_twoway*100).round(3).to_string())
P("farm-days with forecasts (warm seasons): %d ; farms %d ; years %s"%(len(f2),f2.farm_id.nunique(),sorted(f2.date.dt.year.unique())))
P("DONE",round(time.time()-t0))
