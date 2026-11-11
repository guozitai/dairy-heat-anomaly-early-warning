#!/usr/bin/env python3
"""C10. Records removed at cleaning: split by criterion, and what follows a record with a yield of 0.5 kg or less.
If such a record were a milking at which the cow gave almost no milk, the next milking would contain the milk of two intervals;
if it is a recording artefact, the next milking yields the usual amount."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); D="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
s=pd.read_parquet(D+"unified_session.parquet",columns=["farm_id","bullNumber","milkingStart","milkYield"]).rename(columns={"bullNumber":"cow","milkingStart":"t"}); s["y"]=pd.to_numeric(s.milkYield,errors="coerce").astype("float32")
n=len(s); P("raw records",n); nt=s.t.isna(); lo=(s.y<=0.5)|s.y.isna(); hi=s.y>=45
P("no start time: %d ; yield <=0.5 kg or missing: %d (of which exactly 0 or missing: %d) ; yield >=45 kg: %d ; any: %d"%(nt.sum(),lo.sum(),((s.y<=0)|s.y.isna()).sum(),hi.sum(),(nt|lo|hi).sum()))
P("yield of low records: quantiles",s.y[lo].quantile([.05,.25,.5,.75,.95]).round(2).to_dict())
s=s[~nt].sort_values(["farm_id","cow","t"]).reset_index(drop=True); same=(s.farm_id.shift(-1)==s.farm_id)&(s.cow.shift(-1)==s.cow); s["y_next"]=s.y.shift(-1).where(same); s["dt_next"]=((s.t.shift(-1)-s.t).dt.total_seconds()/3600).where(same)
samep=(s.farm_id.shift(1)==s.farm_id)&(s.cow.shift(1)==s.cow); s["dt_prev"]=((s.t-s.t.shift(1)).dt.total_seconds()/3600).where(samep); s["y_prev"]=s.y.shift(1).where(samep)
ok=(s.y>0.5)&(s.y<45); g=s[ok].groupby(["farm_id","cow"]).y.median().rename("ymed"); s=s.join(g,on=["farm_id","cow"])
L=s[(s.y<=0.5)&s.y_next.notna()&(s.y_next>0.5)&(s.y_next<45)&s.ymed.notna()]
P("low records followed by a normal record of the same cow: %d"%len(L))
for a,b in ((0,1),(1,5),(5,11.5),(11.5,34)):
    z=L[(L.dt_next>a)&(L.dt_next<=b)]; r=z.y_next/z.ymed
    if len(z): P("  next record %g-%g h later: n=%d ; next yield / cow median: median %.3f (IQR %.3f-%.3f)"%(a,b,len(z),r.median(),r.quantile(.25),r.quantile(.75)))
z=L[(L.dt_next>5)&(L.dt_next<=11.5)&(L.dt_prev>5)&(L.dt_prev<=11.5)]; P("low record with regular intervals on both sides: n=%d ; next/median %.3f ; previous/median %.3f"%(len(z),(z.y_next/z.ymed).median(),(z.y_prev/z.ymed).median()))
P("share of low records with another record of the same cow within 60 min (before or after): %.3f"%(((s.dt_next<=1)|(s.dt_prev<=1))[s.y<=0.5].mean()))
P("low records by farm:",s[s.y<=0.5].groupby("farm_id").size().to_dict())
P("DONE",round(time.time()-t0))
