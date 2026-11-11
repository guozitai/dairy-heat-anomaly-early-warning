#!/usr/bin/env python3
"""B8b. New disease records of the study cows and the availability of milking data on the three preceding days."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","D","elig","risk_dis"])
dis=pd.read_parquet(V+"disease_cid.parquet"); cov=F.groupby("farm_id").dnum.agg(["min","max"]); cows=F[F.elig][["cid"]].drop_duplicates()
d8=dis.merge(cov,left_on="farm_id",right_index=True); d8=d8[(d8.dnum>=d8["min"]+3)&(d8.dnum<=d8["max"])].merge(cows,on="cid").sort_values(["cid","dnum"])
gap=d8.groupby("cid").dnum.diff(); new=d8[gap.isna()|(gap>14)][["cid","dnum","grp"]].drop_duplicates(["cid","dnum"]).reset_index(drop=True); new["case"]=np.arange(len(new))
parts=[]
for k in (1,2,3):
    t=new[["case","cid","dnum"]].copy(); t["dnum"]=t.dnum-k; parts.append(t)
L=pd.concat(parts).merge(F,on=["cid","dnum"],how="left")
g=L.groupby("case").agg(any_row=("farm_id",lambda x:x.notna().any()),any_elig=("elig",lambda x:(x==True).any()),any_D=("D",lambda x:x.notna().any()),any_risk=("risk_dis",lambda x:(x==True).any()),nD=("D","count"))
new=new.join(g,on="case")
print("new disease records of study cows (>=3 d after start of milking records):",len(new))
print("  with a milking record on days -3..-1: %d (%.1f%%)"%(new.any_row.sum(),100*new.any_row.mean()))
print("  with an eligible cow-day (DIM 10-400): %d (%.1f%%)"%(new.any_elig.sum(),100*new.any_elig.mean()))
print("  with a day deviation: %d (%.1f%%)"%(new.any_D.sum(),100*new.any_D.mean()))
print("  in the Task 2 risk set on at least one day: %d (%.1f%%)"%(new.any_risk.sum(),100*new.any_risk.mean()))
print("  with >=2 day deviations: %d"%((new.nD>=2).sum()))
print(new.groupby("grp").agg(n=("case","size"),milked=("any_row","mean"),elig=("any_elig","mean"),withD=("any_D","mean")).round(3).to_string())
