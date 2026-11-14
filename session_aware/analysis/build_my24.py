#!/usr/bin/env python3
"""SAT rescue step 1: session-aware daily production index.
Rebuilds a 24-h equivalent yield (MY24) per cow-day from session records using inter-milking intervals,
so that unrecorded sessions do not appear as yield drops. Writes sessions_clean / cowday_my24 parquet + diagnostics."""
import pandas as pd, numpy as np, time
t0=time.time(); W="/home/guozitai/20261111_sat_rescue/"; D="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
cols=["farm_id","bullNumber","milkDate","shift","milkingStart","milkYield","AvgConductivity","flowRate","source"]
s=pd.read_parquet(D+"unified_session.parquet",columns=cols).rename(columns={"bullNumber":"cow","milkDate":"date","milkingStart":"t"})
s["y"]=pd.to_numeric(s.milkYield,errors="coerce").astype("float32"); s["cond"]=s.AvgConductivity.astype("float32").where(s.AvgConductivity>0)
s["flow"]=pd.to_numeric(s.flowRate,errors="coerce").astype("float32")
s=s.drop(columns=["milkYield","AvgConductivity","flowRate"])
n0=len(s); s=s[(s.y>0.5)&(s.y<45)&s.t.notna()]
P("sessions",n0,"-> y in (0.5,45):",len(s),round(time.time()-t0))
# exact duplicates (same cow, same start time): keep the larger yield
s=s.sort_values(["farm_id","cow","t","y"]).drop_duplicates(["farm_id","cow","t"],keep="last")
P("after exact-dup removal",len(s))
# records of the same cow within 60 min = reattachment: sum yields, keep first start
g=s.groupby(["farm_id","cow"],sort=False)
gapmin=(s.t-g.t.shift(1)).dt.total_seconds()/60
newblk=(gapmin.isna())|(gapmin>60)
s["blk"]=newblk.cumsum()
agg=s.groupby("blk",sort=False).agg(farm_id=("farm_id","first"),cow=("cow","first"),date=("date","first"),shift=("shift","first"),t=("t","first"),y=("y","sum"),cond=("cond","mean"),flow=("flow","mean"),source=("source","first"),nrec=("y","size")).reset_index(drop=True)
P("after merging reattachments",len(agg)," merged share",round((agg.nrec>1).mean(),4),round(time.time()-t0))
s=agg; del agg
g=s.groupby(["farm_id","cow"],sort=False)
s["dt"]=((s.t-g.t.shift(1)).dt.total_seconds()/3600).astype("float32")
P("\ninterval (h) quantiles by farm:"); P(s.groupby("farm_id").dt.quantile([.01,.05,.25,.5,.75,.95,.99]).unstack().round(1).to_string())
bins=[0,3,5,6,7,8,9,10,11,12,14,17,20,26,1e9]
P("\ninterval histogram share by farm:"); P(pd.crosstab(s.farm_id,pd.cut(s.dt,bins),normalize="index").round(3).to_string())
s["valid"]=(s.dt>=5)&(s.dt<=11.5)
s["rate"]=(s.y/s.dt).where(s.valid).astype("float32")
P("\nvalid-interval share by farm:"); P(s.groupby("farm_id").valid.mean().round(3).to_string())
P("rate kg/h quantiles (valid):",s.rate.quantile([.01,.1,.5,.9,.99]).round(2).to_dict())
s.to_parquet(W+"sessions_clean.parquet",index=False)
# ---- cow-day ----
s["yv"]=s.y.where(s.valid); s["dtv"]=s.dt.where(s.valid)
d=s.groupby(["farm_id","cow","date"],sort=False).agg(n_sess=("y","size"),n_valid=("valid","sum"),y_sum=("y","sum"),yv=("yv","sum"),dtv=("dtv","sum"),cond=("cond","mean"),cond_max=("cond","max"),flow=("flow","mean"),t_first=("t","min")).reset_index()
d["my24"]=(24*d.yv/d.dtv).where(d.n_valid>=1).astype("float32")
P("\ncow-days",len(d)," with my24",int(d.my24.notna().sum()),round(time.time()-t0))
P("n_sess share:",d.n_sess.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())
P("n_valid share:",d.n_valid.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())
P("my24 quantiles:",d.my24.quantile([.01,.1,.5,.9,.99]).round(1).to_dict(),"| raw daily sum quantiles:",d.y_sum.quantile([.01,.1,.5,.9,.99]).round(1).to_dict())
# attributes from unified_daily_clean
u=pd.read_parquet(D+"unified_daily_clean.parquet",columns=["farm_id","bullNumber","milkDate","DIM","inferred_parity","parity","calvingDate","qc_flag","yield_suspicious","county_adcode","thi_max","thi_mean","thi_min","hours_thi_gt_72","hours_thi_gt_79","thi_max_lag1","thi_max_lag3_avg","daily_milk_total"]).rename(columns={"bullNumber":"cow","milkDate":"date"})
d=d.merge(u,on=["farm_id","cow","date"],how="left")
P("merge with unified_daily_clean: matched share",round(d.DIM.notna().mean(),4)," corr(y_sum,daily_milk_total)",round(d[["y_sum","daily_milk_total"]].corr().iloc[0,1],4))
d["par"]=pd.to_numeric(d.inferred_parity,errors="coerce").fillna(pd.to_numeric(d.parity,errors="coerce"))
d=d.sort_values(["farm_id","cow","date"]).reset_index(drop=True)
# ---- cow-specific trailing baseline: median of MY24 on days t-8..t-2 (>=4 values) ----
def trailing_median(d,col,lo=2,hi=8,minn=4,maxlag=8):
    g=d.groupby(["farm_id","cow"],sort=False)
    vals=[]
    for k in range(1,maxlag+1):
        v=g[col].shift(k); dd=(d.date-g.date.shift(k)).dt.days
        vals.append(v.where((dd>=lo)&(dd<=hi)).to_numpy(dtype="float32"))
    M=np.vstack(vals)
    cnt=np.sum(~np.isnan(M),axis=0)
    med=np.nanmedian(M,axis=0)
    med[cnt<minn]=np.nan
    return med
import warnings; warnings.filterwarnings("ignore")
d["base"]=trailing_median(d,"my24"); d["dev"]=np.log(d.my24/d.base).astype("float32")
d["base_raw"]=trailing_median(d,"y_sum"); d["dev_raw"]=np.log(d.y_sum/d.base_raw).astype("float32")
P("\nvalid dev share",round(d.dev.notna().mean(),3)," dev SD (my24)",round(d.dev.std(),4)," robust SD (MAD*1.4826)",round((d.dev-d.dev.median()).abs().median()*1.4826,4))
P("dev_raw SD (raw daily sum)",round(d.dev_raw.std(),4)," robust",round((d.dev_raw-d.dev_raw.median()).abs().median()*1.4826,4))
for thr in (0.85,0.8,0.75,0.7,0.6):
    P(f"  share of cow-days below {thr:.2f} of own baseline: my24 {np.mean(d.dev<np.log(thr)):.4f} | raw sum {np.mean(d.dev_raw<np.log(thr)):.4f}")
# how much of the raw-sum drops are session artifacts
dd=d[d.dev_raw.notna()&d.dev.notna()]
raw_drop=dd.dev_raw<np.log(0.8); my_drop=dd.dev<np.log(0.8)
P("\namong raw-sum drops(<0.8): share that are also my24 drops:",round(my_drop[raw_drop].mean(),3)," | among my24 drops: share also raw drops:",round(raw_drop[my_drop].mean(),3))
md=dd.groupby(["farm_id","date"]).n_sess.transform(lambda x:x.mode().iloc[0]); fewer=dd.n_sess<md
P("raw drops on fewer-session days:",round(fewer[raw_drop].mean(),3)," my24 drops on fewer-session days:",round(fewer[my_drop].mean(),3)," base rate fewer:",round(fewer.mean(),3))
d.to_parquet(W+"cowday_my24.parquet",index=False)
P("DONE",round(time.time()-t0))
