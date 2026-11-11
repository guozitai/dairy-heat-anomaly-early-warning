#!/usr/bin/env python3
"""C0b. Pattern of missing milking records by farm-day and shift, restricted to the three regular shifts.
Roster of a farm-day = cows with at least one recorded milking on that day; coverage of a shift = cows recorded in the shift / roster.
Days without any record are counted separately."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"; pd.set_option("display.width",260)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
ct=pd.read_parquet(V+"cowtab.parquet"); s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift"]).merge(ct[["cid","farm_id"]],on="cid"); s=s[s.farm_id.isin(LAB)]; s["farm"]=s.farm_id.map(LAB)
print("shift labels:",s["shift"].value_counts().to_dict()); s=s[s["shift"].isin([1,2,3,"1","2","3"])]
ro=s.groupby(["farm","dnum"]).cid.nunique().rename("roster").reset_index(); fs=s.groupby(["farm","dnum","shift"]).cid.nunique().rename("rec").reset_index()
sh=pd.DataFrame({"shift":sorted(s["shift"].unique())}); full=ro.merge(sh,how="cross").merge(fs,on=["farm","dnum","shift"],how="left").fillna({"rec":0}); full["cov"]=full.rec/full.roster; full["miss"]=full.roster-full.rec
def pat(x): tm=x.miss.sum(); return pd.Series(dict(farm_days=x.dnum.nunique() if "dnum" in x else np.nan,farm_day_shifts=len(x),shifts_lt20=(x["cov"]<0.2).mean(),shifts_20_80=((x["cov"]>=0.2)&(x["cov"]<0.8)).mean(),shifts_ge80=(x["cov"]>=0.8).mean(),missing_records=tm,missing_share=tm/x.roster.sum(),miss_in_lt20=x.miss[x["cov"]<0.2].sum()/tm,miss_in_20_80=x.miss[(x["cov"]>=0.2)&(x["cov"]<0.8)].sum()/tm,miss_in_ge80=x.miss[x["cov"]>=0.8].sum()/tm,median_cov_ge80=x["cov"][x["cov"]>=0.8].median()))
PT=pd.concat([full.groupby("farm").apply(pat),pat(full.assign(dnum=full.farm+full.dnum.astype(str))).to_frame("All").T]); print(PT.round(3).to_string()); PT.to_csv(V+"tab_missing_pattern.csv")
print("mean coverage by farm and shift:"); print(full.groupby(["farm","shift"])["cov"].mean().unstack().round(3).to_string())
# calendar days without any record inside each farm's period
g=ro.groupby("farm").dnum.agg(["min","max","nunique"]); g["calendar"]=g["max"]-g["min"]+1; g["days_without_records"]=g.calendar-g["nunique"]; print(g.to_string())
