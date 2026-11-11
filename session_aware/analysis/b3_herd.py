#!/usr/bin/env python3
"""B3. Herd-level response to heat load. Herd index H = mean day deviation D over cows (>=200 cows), ERA5 THI at the farm grid cell.
Dose-response by lag, hinge slopes with farm x year-month fixed effects (cluster farm-month), per-farm slopes, subgroup responses,
incidence of acute drops vs THI, comparison with raw daily sums, archived-forecast skill by lead day. Outputs CSVs for Table 3 / Fig 3."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250); L=np.log(0.8)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","loc","D","dev_raw","elig","DIM","par","my24","base_my24","risk_onset","y_onset"]); F=F[F.elig&F["loc"].notna()].copy()
F["drop"]=np.where(F.D.notna(),(F.D<L).astype(float),np.nan)
F["par2"]=np.where(F.par>=2,"multiparous","primiparous"); F["stage"]=pd.cut(F.DIM,[9,100,200,400],labels=["DIM 10-100","DIM 101-200","DIM 201-400"])
med=F.groupby(["farm_id","dnum"]).base_my24.transform("median"); F["ylev"]=np.where(F.base_my24>=med,"higher-yielding half","lower-yielding half")
def agg(g): return g.agg(n=("D","count"),H=("D","mean"),Hmed=("D","median"),Hraw=("dev_raw","mean"),drop=("drop","mean"),n_risk=("risk_onset","sum"),n_on=("y_onset","sum"))
fd=agg(F.groupby(["farm_id","loc","date"])).reset_index(); fd=fd[fd.n>=200].copy(); fd["onset"]=fd.n_on/fd.n_risk.where(fd.n_risk>=100)
T=pd.read_csv(V+"era5_farm_daily.csv",parse_dates=["date"]).drop(columns=["n"]).sort_values(["loc","date"]); g=T.groupby("loc")
for k in (1,2,3): T[f"tmax_l{k}"]=g.tmax.shift(k)
T["tmax3"]=g.tmax.transform(lambda s:s.rolling(3,min_periods=3).mean()); T["tmax2p"]=g.tmax.transform(lambda s:s.shift(1).rolling(2,min_periods=2).mean())
fd=fd.merge(T,on=["loc","date"],how="left"); fd["m"]=fd.date.dt.month; fd["fym"]=fd.farm_id.astype(str)+"_"+fd.date.dt.to_period("M").astype(str); fd["farm"]=fd.farm_id.map(LAB)
fd.to_parquet(V+"farmday_v2.parquet",index=False); warm=fd[fd.m.isin([5,6,7,8,9])].copy()
P("farm-days",len(fd),"warm-season",len(warm),"farms",fd.farm_id.nunique()); P(warm.groupby("farm").agg(days=("date","size"),cows=("n","median"),ge79=("tmax",lambda s:(s>=79).sum()),ge81=("tmax",lambda s:(s>=81).sum())).to_string())
dm=lambda s,k: s-s.groupby(k).transform("mean")
P("\nSD of farm-month-demeaned herd index (warm season): session-aware %.4f | raw daily sum %.4f"%(dm(warm.H,warm.fym).std(),dm(warm.Hraw,warm.fym).std()))
bins=[0,68,72,75,77,79,81,90]; rows=[]
for y in ("H","Hraw","drop","onset"):
    for x in ("tmax","tmax_l1","tmax_l2","tmax_l3"):
        z=warm.dropna(subset=[y,x]).copy(); z["yd"]=dm(z[y],z.fym); z["b"]=pd.cut(z[x],bins)
        r=z.groupby("b",observed=True).yd.agg(["size","mean"]).reset_index()
        # standard error of the class mean, clustered by farm-month
        z["res"]=z.yd-z.groupby("b",observed=True).yd.transform("mean"); cs=z.groupby(["b","fym"],observed=True).res.sum().reset_index(); se=np.sqrt(cs.assign(r2=cs.res**2).groupby("b",observed=True).r2.sum())/z.groupby("b",observed=True).size()
        r["sem"]=r.b.map(se).astype(float).values; r["outcome"]=y; r["lag"]=x; rows.append(r)
B=pd.concat(rows); B.to_csv(V+"fig3_dose_response.csv",index=False)
for y in ("H","Hraw","drop","onset"):
    for x in ("tmax","tmax_l1","tmax_l2"):
        r=B[(B.outcome==y)&(B.lag==x)]; P(f"{y:5s} vs {x:8s}: "+" | ".join(f"{str(b)}: {m*100:+.2f} ({s*100:.2f}; n={n})" for b,n,m,s in zip(r.b,r["size"],r["mean"],r["sem"])))
def absorb(v,keys,it=12):
    v=np.asarray(v,float).copy()
    for _ in range(it):
        for k in keys: v=v-pd.Series(v).groupby(k).transform("mean").values
    return v
def fe(df,y,xs,keys,cl):
    z=df.dropna(subset=[y]+xs); K=[z[k].values for k in keys]; yd=absorb(z[y].values,K); X=np.column_stack([absorb(z[x].values,K) for x in xs])
    b=np.linalg.lstsq(X,yd,rcond=None)[0]; e=yd-X@b; XtXi=np.linalg.pinv(X.T@X); meat=np.zeros((X.shape[1],)*2); G=z[cl].values
    for gg in np.unique(G):
        i=G==gg; s=X[i].T@e[i]; meat+=np.outer(s,s)
    return b,np.sqrt(np.diag(XtXi@meat@XtXi)),len(z)
P("\nhinge slopes per THI unit above tau (farm x year-month FE; SE clustered by farm-month):"); S=[]
for y in ("H","Hraw","drop","onset"):
    for x in ("tmax","tmax_l1","tmax_l2","tmax2p","tmax3"):
        for tau in (72,76,79):
            warm["h"]=np.maximum(warm[x]-tau,0); b,se,n=fe(warm,y,["h"],["fym"],"fym"); S.append(dict(outcome=y,exposure=x,tau=tau,slope=b[0],se=se[0],t=b[0]/se[0],n=n))
S=pd.DataFrame(S); S.to_csv(V+"tab3_hinge.csv",index=False); P(S[S.exposure.isin(["tmax_l1","tmax2p","tmax3"])].assign(slope=lambda d:d.slope*100,se=lambda d:d.se*100).round(3).to_string())
P("\nper farm (exposure = mean of previous two days' maximum THI, hinge at 76):"); PF=[]
for f,s in warm.groupby("farm"):
    s=s.copy(); s["h"]=np.maximum(s.tmax2p-76,0); r=dict(farm=f,days=len(s))
    for y in ("H","drop","onset","Hraw"):
        b,se,n=fe(s,y,["h"],["fym"],"fym"); r[y]=b[0]; r[y+"_se"]=se[0]
    PF.append(r)
PF=pd.DataFrame(PF); PF.to_csv(V+"fig3_per_farm.csv",index=False); P(PF.assign(**{c:PF[c]*100 for c in PF.columns if c not in ("farm","days")}).round(3).to_string())
P("\nsubgroups (hinge 76 on previous-two-day mean THI):"); SG=[]
for col in ("par2","stage","ylev"):
    for lev,x in F.groupby(col,observed=True):
        a=x.groupby(["farm_id","loc","date"]).agg(n=("D","count"),H=("D","mean")).reset_index(); a=a[a.n>=80].merge(T[["loc","date","tmax2p"]],on=["loc","date"]); a=a[a.date.dt.month.isin([5,6,7,8,9])].copy(); a["fym"]=a.farm_id.astype(str)+"_"+a.date.dt.to_period("M").astype(str); a["h"]=np.maximum(a.tmax2p-76,0)
        b,se,n=fe(a,"H",["h"],["fym"],"fym"); SG.append(dict(group=str(lev),slope=b[0]*100,se=se[0]*100,n=n)); P(f"   {str(lev):22s} {b[0]*100:+.3f} % per unit (SE {se[0]*100:.3f}; n={n})")
pd.DataFrame(SG).to_csv(V+"tab3_subgroups.csv",index=False)
# forecast skill at farm grid cells
FC=pd.read_csv(V+"fcst_farm_daily.csv",parse_dates=["date"]).merge(T[["loc","date","tmax"]],on=["loc","date"]); FC=FC[FC.date.dt.month.isin([5,6,7,8,9])]
P("\narchived ECMWF forecasts vs ERA5 daily maximum THI (warm season, 4 grid cells): n=%d; ERA5 days >=79: %d"%(len(FC),(FC.tmax>=79).sum())); FS=[]
for k in range(1,8):
    x=FC.dropna(subset=[f"fc_tmax_l{k}"]); e=x[f"fc_tmax_l{k}"]-x.tmax; bias=e.mean(); fcb=x[f"fc_tmax_l{k}"]-bias; ob=x.tmax>=79; f1=x[f"fc_tmax_l{k}"]>=79; f2=fcb>=79
    FS.append(dict(lead=k,n=len(x),mae=e.abs().mean(),bias=bias,r=np.corrcoef(x[f"fc_tmax_l{k}"],x.tmax)[0,1],pod=(ob&f1).sum()/ob.sum(),far=((~ob)&f1).sum()/max(f1.sum(),1),pod_bc=(ob&f2).sum()/ob.sum(),far_bc=((~ob)&f2).sum()/max(f2.sum(),1)))
FS=pd.DataFrame(FS); FS.to_csv(V+"fig3_forecast_skill.csv",index=False); P(FS.round(3).to_string())
# herd response on days announced as hot by the forecast issued k days ahead (response taken at lag 1-2)
fd2=fd.merge(pd.read_csv(V+"fcst_farm_daily.csv",parse_dates=["date"]),on=["loc","date"],how="inner"); fd2=fd2[fd2.m.isin([5,6,7,8,9])].copy(); fd2=fd2.sort_values(["farm_id","date"])
fd2["Hd"]=dm(fd2.H,fd2.fym); g2=fd2.groupby("farm_id"); fd2["Hd_next"]=(g2.Hd.shift(-1)+g2.Hd.shift(-2))/2; AN=[]
for k in (1,3,5,7):
    hotf=fd2[f"fc_tmax_l{k}"]>=79.0+FS.bias[FS.lead==k].iloc[0]; zz=fd2[hotf].dropna(subset=["Hd_next"]); a=zz.Hd_next; b=fd2[~hotf].Hd_next.dropna()
    cl=(zz.assign(res=a-a.mean()).groupby("fym").res.sum()**2).sum(); AN.append(dict(lead=k,n_hot=len(a),resp_hot=a.mean(),se=np.sqrt(cl)/len(a),resp_other=b.mean()))
AN=pd.DataFrame(AN); AN.to_csv(V+"tab3_forecast_announced.csv",index=False); P("\nherd index on the two days after forecast-announced hot days (bias-corrected forecast >=79), farm-month demeaned:"); P(AN.assign(resp_hot=AN.resp_hot*100,se=AN.se*100,resp_other=AN.resp_other*100).round(3).to_string())
P("DONE",round(time.time()-t0))
