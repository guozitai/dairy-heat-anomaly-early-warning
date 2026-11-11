#!/usr/bin/env python3
"""B8. Counts quoted in the text: availability and dispersion of the day deviation D, overlap of the flags of the two measures,
usable milkings per cow-day, and disease records within the periods covered by milking records."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"; L=np.log(0.8)
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","D","dev_raw","dev_my24","elig","n_dev","n_sess","DIM","par"]); F=F[F.elig]
rs=lambda z: float((z-z.median()).abs().median()*1.4826)
b=F[F.D.notna()&F.dev_raw.notna()]
print("eligible cow-days",len(F),"with D",int(F.D.notna().sum()),"share",round(F.D.notna().mean(),4))
print("robust SD: D %.4f | daily-sum deviation %.4f"%(rs(F.D.dropna()),rs(F.dev_raw.dropna())))
print("flag daily sum %.4f | flag D %.4f | of daily-sum flags also D %.4f | of D flags also daily-sum %.4f"%((F.dev_raw.dropna()<L).mean(),(F.D.dropna()<L).mean(),(b.D<L)[b.dev_raw<L].mean(),(b.dev_raw<L)[b.D<L].mean()))
print("usable milkings per cow-day with D: 1: %.3f 2: %.3f 3+: %.3f"%((F.n_dev[F.D.notna()]==1).mean(),(F.n_dev[F.D.notna()]==2).mean(),(F.n_dev[F.D.notna()]>=3).mean()))
print("parity: primiparous share of cow-days %.3f ; median DIM %.0f"%((F.par==1).mean(),F.DIM.median()))
dis=pd.read_parquet(V+"disease_cid.parquet"); cov=F.groupby("farm_id").dnum.agg(["min","max"]); cows=F[["cid"]].drop_duplicates()
d8=dis.merge(cov,left_on="farm_id",right_index=True); d8=d8[(d8.dnum>=d8["min"])&(d8.dnum<=d8["max"])].merge(cows,on="cid")
print("disease records of study cows within the milking-record period:",len(d8)); print(d8.grp.value_counts().to_string())
d8=d8.sort_values(["cid","dnum"]); gap=d8.groupby("cid").dnum.diff(); print("new records (none in previous 14 d):",int((gap.isna()|(gap>14)).sum()))
