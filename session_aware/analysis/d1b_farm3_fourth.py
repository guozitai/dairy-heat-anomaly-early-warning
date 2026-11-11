#!/usr/bin/env python3
"""D1b. Farm 3, June to September 2025: the fourth milking session. Start hours by shift, cows concerned, intervals before and after."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt"]); ct=pd.read_parquet(V+"cowtab.parquet"); s=s.merge(ct[["cid","farm_id"]],on="cid"); s=s[s.farm_id==16].copy(); s["t"]=pd.to_datetime(s.t); s["ym"]=s.t.dt.strftime("%Y-%m"); s["h"]=s.t.dt.hour
print("Farm 3 milkings",len(s),"| shift counts:",s["shift"].value_counts().sort_index().to_dict())
z=s[s.ym.isin(["2025-06","2025-07","2025-08","2025-09"])]; pd.set_option("display.width",250)
print("\nstart hour by shift, 2025-06..09 (share within shift):"); print(pd.crosstab(z["shift"],z.h,normalize="index").round(2).to_string())
print("\nstart hour by shift, 2025-03..05:"); y=s[s.ym.isin(["2025-03","2025-04","2025-05"])]; print(pd.crosstab(y["shift"],y.h,normalize="index").round(2).to_string())
z=z.assign(is4=(z["shift"]==4)); cd=z.groupby(["cid","dnum"]).agg(n=("y","size"),has4=("is4","max")); cd["has4"]=cd.has4.astype(bool); print("\ncow-days 2025-06..09:",len(cd),"with a shift-4 record:",int(cd.has4.sum()),"(%.3f)"%cd.has4.mean(),"| records per cow-day with/without shift 4:",cd[cd.has4].n.value_counts(normalize=True).sort_index().round(3).to_dict(),cd[~cd.has4].n.value_counts(normalize=True).sort_index().round(3).to_dict())
cows4=z[z["shift"]==4].cid.nunique(); print("cows with a shift-4 record:",cows4,"of",z.cid.nunique()); first=z[z["shift"]==4].t.min(); print("first shift-4 record:",first)
c4=set(z[z["shift"]==4].cid.unique()); zz=z[z.cid.isin(c4)&(z.t>=first)]
print("\ncows with shift-4 records, after its start: interval distribution by shift (0.5-h mode and shares)")
for sh,g in zz.groupby("shift"):
    b=(g.dt*2).round()/2; print("  shift",sh,"n",len(g),"mode",b.value_counts().idxmax(),"| 5-7 h %.3f, 7-9 h %.3f, 11.5-14 h %.3f, 14-20 h %.3f, 20-34 h %.3f"%(((g.dt>=5)&(g.dt<7)).mean(),((g.dt>=7)&(g.dt<9)).mean(),((g.dt>11.5)&(g.dt<=14)).mean(),((g.dt>14)&(g.dt<=20)).mean(),((g.dt>20)&(g.dt<=34)).mean()), "| median yield %.1f"%g.y.median())
print("\nother cows in the same months: interval shares by shift")
for sh,g in z[~z.cid.isin(c4)].groupby("shift"):
    print("  shift",sh,"n",len(g),"| 5-7 h %.3f, 7-9 h %.3f, 11.5-14 h %.3f"%(((g.dt>=5)&(g.dt<7)).mean(),((g.dt>=7)&(g.dt<9)).mean(),((g.dt>11.5)&(g.dt<=14)).mean()))
print("\nyield after 11.5-14 h relative to yield after 5-7 h in the same shift (cows with shift 4), by shift:")
for sh,g in zz.groupby("shift"):
    a=g[(g.dt>11.5)&(g.dt<=14)].y.median(); b=g[(g.dt>=5)&(g.dt<7)].y.median(); c=g[(g.dt>=7)&(g.dt<9)].y.median(); print("  shift",sh,"median yield after 11.5-14 h: %.2f ; after 5-7 h: %.2f ; after 7-9 h: %.2f"%(a,b,c))
