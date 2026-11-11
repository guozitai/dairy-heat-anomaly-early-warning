#!/usr/bin/env python3
"""D1. Scheduled milking intervals by farm and period: modal interval of each shift by farm-month, and the share of milkings in
farm-month-shifts whose modal interval lies outside the regular range (5-11.5 h)."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt"]); ct=pd.read_parquet(V+"cowtab.parquet"); print("cowtab cols",list(ct.columns)[:8])
s=s.merge(ct[["cid","farm_id"]],on="cid"); s["farm"]=s.farm_id.map(LAB); s["ym"]=pd.to_datetime(s.t).dt.strftime("%Y-%m"); s=s[s.dt.notna()&(s.dt<=34)]
s["b"]=(s.dt*2).round()/2   # 0.5-h bins
g=s.groupby(["farm","ym","shift"]); mode=g.b.agg(lambda x:x.value_counts().idxmax()).rename("mode"); n=g.size().rename("n"); med=g.dt.median().rename("median")
sh=lambda lo,hi: g.dt.agg(lambda x:((x>=lo)&(x<hi)).mean())
T=pd.concat([n,mode,med,sh(5,7).rename("s5_7"),sh(7,9).rename("s7_9"),sh(9,11.5).rename("s9_115"),sh(11.5,14).rename("s115_14"),sh(14,20).rename("s14_20"),sh(20,34).rename("s20_34")],axis=1).reset_index()
pd.set_option("display.width",250); pd.set_option("display.max_rows",400)
out=T[(T["mode"]>11.5)|(T["mode"]<5)|(T.s115_14>0.10)]; print("farm-month-shifts with modal interval outside 5-11.5 h or >10% of intervals in 11.5-14 h:"); print(out.round(3).to_string())
tot=T.groupby("farm").n.sum(); bad=T[(T["mode"]>11.5)&(T["mode"]<=14)].groupby("farm").n.sum().reindex(tot.index).fillna(0); print("\nshare of milkings in farm-month-shifts with modal interval 11.5-14 h:"); print((bad/tot).round(4).to_string())
print("\nFarm 3 by month: modal interval per shift"); print(T[T.farm=="Farm 3"].pivot(index="ym",columns="shift",values="mode").to_string())
print("\nFarm 3 by month: n per shift"); print(T[T.farm=="Farm 3"].pivot(index="ym",columns="shift",values="n").to_string())
print("\nall farms: overall share of intervals 11.5-12.5 h:"); print(s.assign(x=(s.dt>=11.5)&(s.dt<12.5)).groupby("farm").x.mean().round(4).to_string())
T.to_csv(V+"tab_schedule_by_farm_month_shift.csv",index=False)
