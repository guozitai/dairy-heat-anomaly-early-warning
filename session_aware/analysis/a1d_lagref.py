#!/usr/bin/env python3
"""A1d. Sensitivity versions of the day deviation for the herd-level analysis.
(1) Lagged reference window: same construction as D, but the same-shift reference rate / usual yield is the median over days t-21..t-9
    (>=5 values), so that the reference is not affected by heat load in the week before day t.
(2) Herd means restricted to cow-days with three usable milkings; share of cow-days with three recorded milkings (completeness)."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt","valid","rate"])
s["sh"]=pd.factorize(s["shift"])[0].astype("int8"); s=s.sort_values(["cid","sh","t"]).reset_index(drop=True)
c=s.cid.to_numpy(); sh=s.sh.to_numpy(); dn=s.dnum.to_numpy(); N=len(s); P("sessions",N,round(time.time()-t0))
def lagmed(x,lo,hi,minn,K=24):
    M=np.full((K,N),np.nan,dtype="float32")
    for k in range(1,K+1):
        ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=lo)&(dd<=hi); M[k-1,k:]=np.where(ok,x[:-k],np.nan)
    cnt=np.sum(~np.isnan(M),axis=0); med=np.nanmedian(M,axis=0); med[cnt<minn]=np.nan; return med
r=s.rate.to_numpy(dtype="float32"); yy=s.y.where(s.dt<=34).to_numpy(dtype="float32")
s["refL"]=lagmed(r,9,21,5); P("lagged reference rate done",round(time.time()-t0)); s["refyL"]=lagmed(yy,9,21,5); P("lagged usual yield done",round(time.time()-t0))
reg=s.valid&s.refL.notna(); lg=(s.dt>11.5)&(s.dt<=34)&s.refyL.notna()
s["yoL"]=np.where(reg|lg,s.y,np.nan); s["yeL"]=np.where(reg,s.refL*s.dt,np.where(lg,s.refyL,np.nan)); s["useL"]=(reg|lg)
d=s.groupby(["cid","dnum"],sort=True).agg(yoL=("yoL","sum"),yeL=("yeL","sum"),nL=("useL","sum")).reset_index()
d["DL"]=np.log(d.yoL/d.yeL).where(d.nL>=1).astype("float32"); d[["cid","dnum","DL","nL"]].to_parquet(V+"cowday_lagref.parquet",index=False)
P("cow-days with lagged-reference deviation",int(d.DL.notna().sum()),round(time.time()-t0))
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","loc","D","elig","n_dev","n_sess"]); F=F[F.elig&F["loc"].notna()].merge(d[["cid","dnum","DL"]],on=["cid","dnum"],how="left")
x=F.dropna(subset=["D","DL"]); P("eligible cow-days with both: %d ; corr(D,DL)=%.3f ; mean D %.4f mean DL %.4f ; share DL<ln0.8 %.4f"%(len(x),np.corrcoef(x.D,x.DL)[0,1],x.D.mean(),x.DL.mean(),(x.DL<np.log(0.8)).mean()))
F["D3"]=F.D.where(F.n_dev==3); F["c3"]=(F.n_sess>=3).astype("float32"); F["DLc"]=F.DL.where(F.D.notna())
fd=F.groupby(["farm_id","loc","date"]).agg(nD=("D","count"),nL=("DL","count"),HL=("DL","mean"),n3=("D3","count"),H3=("D3","mean"),c3=("c3","mean"),ncd=("c3","size")).reset_index()
fd.to_parquet(V+"farmday_lagref.parquet",index=False); P("farm-days",len(fd),"DONE",round(time.time()-t0))
