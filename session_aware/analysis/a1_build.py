#!/usr/bin/env python3
"""A1. Session-aware measurement layer, version 2.
Per recorded milking: secretion rate = yield / interval since the previous recorded milking (valid if 5-11.5 h).
Reference = median of the same cow's valid rates in the same shift on days t-8..t-2 (>=3 values).
Session deviation = log(rate/reference). Day deviation D = mean of the day's session deviations.
Also: MY24 = 24*sum(yield)/sum(interval) over valid milkings; own-baseline deviations of MY24 and of the raw daily sum for comparison."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time()
W="/home/guozitai/20261111_sat_rescue/"; V=W+"v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
import os; os.makedirs(V,exist_ok=True)
def P(*a): print(*a,flush=True)
s=pd.read_parquet(W+"sessions_clean.parquet",columns=["farm_id","cow","date","shift","t","y","cond","flow","nrec","dt"])
n0=len(s); s=s[(s.nrec<=3)&(s.y<=45)&(s.y>0.5)].copy(); P("sessions",n0,"->",len(s),round(time.time()-t0))
key=s.farm_id.astype(str)+"_"+s.cow.astype(str); cid,uniq=pd.factorize(key); s["cid"]=cid.astype("int32"); del key
cowtab=pd.DataFrame({"cid":np.arange(len(uniq),dtype="int32"),"key":uniq}); cowtab["farm_id"]=cowtab.key.str.split("_",n=1).str[0].astype(int); cowtab["cow"]=cowtab.key.str.split("_",n=1).str[1]
# drop pseudo-cows with implausible session counts per day (unidentified animals recorded under a dummy id)
s["dnum"]=((s.date-pd.Timestamp("2022-01-01")).dt.days).astype("int32")
perday=s.groupby(["cid","dnum"]).size(); bad=perday[perday>5].index.get_level_values(0).unique()
P("pseudo-cow ids with >5 sessions on some day:",len(bad)); s=s[~s.cid.isin(bad)].copy()
# recompute interval after filtering
s=s.sort_values(["cid","t"]).reset_index(drop=True)
c=s.cid.to_numpy(); tt=s.t.to_numpy().astype("datetime64[s]").astype("int64")
dt=np.full(len(s),np.nan,dtype="float32"); same=c[1:]==c[:-1]; dt[1:][same]=((tt[1:]-tt[:-1])[same]/3600.0)
s["dt"]=dt; s["valid"]=(s.dt>=5)&(s.dt<=11.5); s["rate"]=(s.y/s.dt).where(s.valid).astype("float32")
P("valid share",round(s.valid.mean(),4),"sessions",len(s),round(time.time()-t0))
# ---- shift-matched trailing reference ----
s=s.sort_values(["cid","shift","t"]).reset_index(drop=True)
c=s.cid.to_numpy(); sh=s["shift"].to_numpy(); dn=s.dnum.to_numpy(); r=s.rate.to_numpy()
N=len(s); M=np.full((8,N),np.nan,dtype="float32")
for k in range(1,9):
    ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8)
    v=np.where(ok,r[:-k],np.nan); M[k-1,k:]=v
cnt=np.sum(~np.isnan(M),axis=0); ref=np.nanmedian(M,axis=0); ref[cnt<3]=np.nan; del M
s["ref"]=ref.astype("float32"); s["sdev"]=np.log(s.rate/s.ref).astype("float32")
P("sessions with deviation",int(s.sdev.notna().sum()),"robust SD of session deviation",round(float((s.sdev-s.sdev.median()).abs().median()*1.4826),4),round(time.time()-t0))
# ICAR-type flag: long interval and yield/interval below 80% of the cow's usual rate -> preceding milking unrecorded
longgap=(s.dt>11.5)&(s.dt<=34); s["unrec"]=longgap&((s.y/s.dt)<0.8*s.ref)
P("long gaps (11.5-34 h):",int(longgap.sum()),"share of sessions",round(longgap.mean(),4),"; of those with reference, share meeting the ICAR unrecorded-milking criterion:",round(float(s.unrec[longgap&s.ref.notna()].mean()),4))
s[["cid","dnum","shift","t","y","dt","valid","rate","ref","sdev","cond","flow","unrec"]].to_parquet(V+"sess_v2.parquet",index=False)
# ---- cow-day ----
s["yv"]=s.y.where(s.valid); s["dtv"]=s.dt.where(s.valid)
g=s.groupby(["cid","dnum"],sort=True)
d=g.agg(n_sess=("y","size"),n_valid=("valid","sum"),n_dev=("sdev","count"),y_sum=("y","sum"),yv=("yv","sum"),dtv=("dtv","sum"),D=("sdev","mean"),Dmin=("sdev","min"),cond=("cond","mean"),cond_max=("cond","max"),flow=("flow","mean")).reset_index()
d["my24"]=(24*d.yv/d.dtv).where(d.n_valid>=1).astype("float32"); d["D"]=d.D.astype("float32")
P("cow-days",len(d),"with D",int(d.D.notna().sum()),round(time.time()-t0))
# own-baseline deviations for MY24 and raw sum (days t-8..t-2, >=4 values)
c=d.cid.to_numpy(); dn=d.dnum.to_numpy()
def trailing(col,minn=4):
    x=d[col].to_numpy(dtype="float32"); N=len(x); M=np.full((8,N),np.nan,dtype="float32")
    for k in range(1,9):
        ok=(c[k:]==c[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,x[:-k],np.nan)
    cnt=np.sum(~np.isnan(M),axis=0); med=np.nanmedian(M,axis=0); med[cnt<minn]=np.nan; return med
d["base_my24"]=trailing("my24"); d["dev_my24"]=np.log(d.my24/d.base_my24).astype("float32")
d["base_raw"]=trailing("y_sum"); d["dev_raw"]=np.log(d.y_sum/d.base_raw).astype("float32")
d=d.merge(cowtab[["cid","farm_id","cow"]],on="cid",how="left"); d["date"]=pd.Timestamp("2022-01-01")+pd.to_timedelta(d.dnum,unit="D")
u=pd.read_parquet(D0+"unified_daily_clean.parquet",columns=["farm_id","bullNumber","milkDate","DIM","inferred_parity","parity","calvingDate","qc_flag","yield_suspicious","daily_milk_total","daily_n_sessions"]).rename(columns={"bullNumber":"cow","milkDate":"date"})
u["cow"]=u.cow.astype(str); d=d.merge(u,on=["farm_id","cow","date"],how="left")
d["par"]=pd.to_numeric(d.inferred_parity,errors="coerce").fillna(pd.to_numeric(d.parity,errors="coerce")).astype("float32")
P("matched to herd records (DIM available):",round(d.DIM.notna().mean(),4))
x=d[d.D.notna()]
P("D: robust SD",round(float((x.D-x.D.median()).abs().median()*1.4826),4)," share D<log(0.8):",round(float((x.D<np.log(0.8)).mean()),4))
for k in (1,2,3):
    z=x[x.n_dev==k].D; P(f"  n_dev={k}: n={len(z)} robust SD {float((z-z.median()).abs().median()*1.4826):.4f} share<0.8: {float((z<np.log(0.8)).mean()):.4f}")
z=d[d.dev_my24.notna()]; P("MY24 own-baseline deviation: robust SD",round(float((z.dev_my24-z.dev_my24.median()).abs().median()*1.4826),4)," share<0.8",round(float((z.dev_my24<np.log(0.8)).mean()),4))
z=d[d.dev_raw.notna()]; P("raw-sum own-baseline deviation: robust SD",round(float((z.dev_raw-z.dev_raw.median()).abs().median()*1.4826),4)," share<0.8",round(float((z.dev_raw<np.log(0.8)).mean()),4))
cc=d[(d.n_sess==3)&(d.n_valid==3)]; P("complete days:",len(cc)," MY24/sum median",round(float((cc.my24/cc.y_sum).median()),4))
d.drop(columns=["inferred_parity","parity","yv","dtv"]).to_parquet(V+"cowday_v2.parquet",index=False); cowtab.to_parquet(V+"cowtab.parquet",index=False)
P("DONE",round(time.time()-t0))
