#!/usr/bin/env python3
"""C5b. Cohort-referenced daily-sum label: share of cows flagged on more than 20% of their days (cows with at least 60 days) and their share of the flags."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","y_sum","par","DIM","elig"]); F=F[F.elig].copy()
F["pc"]=F.par.clip(1,3).astype("int8"); F["dc"]=(F.DIM//30).astype("int16"); g=F.groupby(["farm_id","dnum","pc","dc"]).y_sum; F["med"]=g.transform("median"); F["gn"]=g.transform("size")
F["z"]=np.log(F.y_sum/F.med).where(F.gn>=5); gz=F.groupby(["farm_id","pc"]).z; F["flag"]=(F.z<gz.transform("mean")-1.5*gz.transform("std")).where(F.z.notna()); x=F[F.flag.notna()].copy(); x["flag"]=x.flag.astype(bool)
cs=x.groupby("cid").flag.agg(["mean","sum","size"]); c60=cs[cs["size"]>=60]; ch=c60[c60["mean"]>0.2]
print("cows with >=60 days: %d ; flagged on >20%% of days: %d (%.4f) ; their share of the flags of cows with >=60 days: %.4f ; cows never flagged: %.4f ; share of all flags from cows with >=60 days: %.4f"%(len(c60),len(ch),len(ch)/len(c60),ch["sum"].sum()/c60["sum"].sum(),(c60["sum"]==0).mean(),c60["sum"].sum()/cs["sum"].sum()))
