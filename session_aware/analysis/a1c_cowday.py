#!/usr/bin/env python3
"""A1c. Final day deviation.
For a milking after a regular interval (5-11.5 h): expected yield = shift-matched reference secretion rate x interval.
For a milking after a long gap (11.5-34 h; the preceding milking was not recorded): expected yield = the cow's usual yield in that shift
(median of the same-shift yields on days t-8..t-2). Day deviation D = log(sum observed / sum expected) over the day's usable milkings."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time()
W="/home/guozitai/20261111_sat_rescue/"; V=W+"v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
s=pd.read_parquet(V+"sess_v2.parquet")
s=s.sort_values(["cid","shift","t"]).reset_index(drop=True)
c=s.cid.to_numpy(); sh=s["shift"].to_numpy(); dn=s.dnum.to_numpy(); N=len(s)
yy=s.y.where(s.dt<=34).to_numpy(dtype="float32"); M=np.full((8,N),np.nan,dtype="float32")
for k in range(1,9):
    ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,yy[:-k],np.nan)
cnt=np.sum(~np.isnan(M),axis=0); refy=np.nanmedian(M,axis=0); refy[cnt<3]=np.nan; del M; s["ref_y"]=refy
reg=s.valid&s.ref.notna(); lg=(s.dt>11.5)&(s.dt<=34)&s.ref_y.notna()
s["yo"]=np.where(reg|lg,s.y,np.nan); s["ye"]=np.where(reg,s.ref*s.dt,np.where(lg,s.ref_y,np.nan)); s["use"]=(reg|lg); s["use_rate"]=reg
P("usable milkings: regular-interval %.3f, long-gap (yield-based) %.3f, unusable %.3f of %d"%(reg.mean(),lg.mean(),1-(reg|lg).mean(),N))
sd=np.log(s.yo/s.ye); rs=lambda z: float((z-z.median()).abs().median()*1.4826)
P("session deviation robust SD: regular %.4f | long-gap %.4f ; median: regular %.4f | long-gap %.4f"%(rs(sd[reg]),rs(sd[lg]),sd[reg].median(),sd[lg].median()))
s["yv"]=s.y.where(s.valid); s["dtv"]=s.dt.where(s.valid); s["sdev_all"]=sd.astype("float32")
d=s.groupby(["cid","dnum"],sort=True).agg(n_sess=("y","size"),n_valid=("valid","sum"),n_dev=("use","sum"),n_rate=("use_rate","sum"),y_sum=("y","sum"),yv=("yv","sum"),dtv=("dtv","sum"),yo=("yo","sum"),ye=("ye","sum"),Dmin=("sdev_all","min"),cond=("cond","mean"),cond_max=("cond","max"),flow=("flow","mean")).reset_index()
d["my24"]=(24*d.yv/d.dtv).where(d.n_valid>=1).astype("float32"); d["D"]=np.log(d.yo/d.ye).where(d.n_dev>=1).astype("float32")
c=d.cid.to_numpy(); dn=d.dnum.to_numpy()
def trailing(col,minn=4):
    x=d[col].to_numpy(dtype="float32"); n=len(x); M=np.full((8,n),np.nan,dtype="float32")
    for k in range(1,9):
        ok=(c[k:]==c[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,x[:-k],np.nan)
    cnt=np.sum(~np.isnan(M),axis=0); med=np.nanmedian(M,axis=0); med[cnt<minn]=np.nan; return med
d["base_my24"]=trailing("my24"); d["dev_my24"]=np.log(d.my24/d.base_my24).astype("float32"); d["base_raw"]=trailing("y_sum"); d["dev_raw"]=np.log(d.y_sum/d.base_raw).astype("float32")
# expected daily yield (kg) for loss in kg: own trailing median of MY24
ct=pd.read_parquet(V+"cowtab.parquet"); d=d.merge(ct[["cid","farm_id","cow"]],on="cid",how="left"); d["date"]=pd.Timestamp("2022-01-01")+pd.to_timedelta(d.dnum,unit="D")
u=pd.read_parquet(D0+"unified_daily_clean.parquet",columns=["farm_id","bullNumber","milkDate","DIM","inferred_parity","parity","calvingDate","qc_flag","daily_milk_total"]).rename(columns={"bullNumber":"cow","milkDate":"date"}); u["cow"]=u.cow.astype(str)
d=d.merge(u,on=["farm_id","cow","date"],how="left"); d["par"]=pd.to_numeric(d.inferred_parity,errors="coerce").fillna(pd.to_numeric(d.parity,errors="coerce")).astype("float32")
L=np.log(0.8); x=d[d.D.notna()]
P("cow-days",len(d),"with D",len(x),"(%.3f)"%(len(x)/len(d))," D robust SD %.4f share<0.8 %.4f"%(rs(x.D),(x.D<L).mean()))
for k in (1,2,3):
    z=x[x.n_dev==k]; P(f"  usable milkings={k}: n={len(z)} ({len(z)/len(x):.3f}) robust SD {rs(z.D):.4f} share<0.8 {(z.D<L).mean():.4f}")
z=d[d.dev_my24.notna()]; P("MY24 own-baseline: n=%d robust SD %.4f share<0.8 %.4f"%(len(z),rs(z.dev_my24),(z.dev_my24<L).mean())); z=d[d.dev_raw.notna()]; P("raw-sum own-baseline: n=%d robust SD %.4f share<0.8 %.4f"%(len(z),rs(z.dev_raw),(z.dev_raw<L).mean()))
cc=d[(d.n_sess==3)&(d.n_valid==3)]; P("complete days: %d ; MY24/sum median %.4f IQR %.4f-%.4f"%(len(cc),(cc.my24/cc.y_sum).median(),(cc.my24/cc.y_sum).quantile(.25),(cc.my24/cc.y_sum).quantile(.75)))
d.drop(columns=["inferred_parity","parity","yv","dtv","yo","ye"]).to_parquet(V+"cowday_v2.parquet",index=False)
s[["cid","dnum","shift","t","y","dt","valid","rate","ref","ref_y","sdev_all","use","use_rate","cond","flow"]].to_parquet(V+"sess_v3.parquet",index=False)
P("DONE",round(time.time()-t0))
