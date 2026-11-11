#!/usr/bin/env python3
"""D4. Farm 3 from June 2025: is the fourth shift code a fourth daily milking?
For the cows with records under shift code 4: start times by shift, intervals and yields on days with four recorded milkings,
yield by length of the preceding interval, secretion rate before and after the change, and the same for the other cows of the farm."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"; pd.set_option("display.width",250)
def P(*a): print(*a,flush=True)
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt"]); ct=pd.read_parquet(V+"cowtab.parquet"); s=s.merge(ct[["cid","farm_id"]],on="cid"); s=s[s.farm_id==16].copy(); s["t"]=pd.to_datetime(s.t); s["hr"]=s.t.dt.hour+s.t.dt.minute/60
D0=(pd.Timestamp("2025-06-07")-pd.Timestamp("2022-01-01")).days; first4=s[s["shift"]==4].groupby("cid").dnum.min().rename("d4"); s=s.join(first4,on="cid"); c4=s.d4.notna()
P("Farm 3 cows with records under shift code 4:",int(first4.size),"| first day of such a record by cow: quantiles",(first4-D0).quantile([0,.1,.5,.9,1]).to_dict(),"(days after 2025-06-07)")
a=s[c4&(s.dnum>=s.d4)].copy()   # these cows from their first shift-4 record
b=s[c4&(s.dnum<s.d4)&(s.dnum>=D0-60)].copy()   # the same cows in the 60 days before the change
o=s[~c4&(s.dnum>=D0)].copy()    # other cows from June 2025
P("\n(1) start time (hour of day) by shift code, cows concerned after the change: quartiles"); P(a.groupby("shift").hr.describe()[["count","25%","50%","75%"]].round(2).to_string())
P("    the same cows in the 60 days before:"); P(b.groupby("shift").hr.describe()[["count","25%","50%","75%"]].round(2).to_string())
P("    other cows from June 2025:"); P(o.groupby("shift").hr.describe()[["count","25%","50%","75%"]].round(2).to_string())
# clock-time clusters irrespective of the shift code
a["slot"]=pd.cut(a.hr,[0,5,12,19.5,24],labels=["night 00-05","morning 05-12","afternoon 12-19.5","evening 19.5-24"],right=False)
P("\n(2) cows concerned after the change: recorded milkings by clock-time slot and shift code (row shares)"); P(pd.crosstab(a.slot,a["shift"],normalize="index").round(3).to_string()); P(a.slot.value_counts(normalize=True).round(3).to_dict())
cd=a.groupby(["cid","dnum"]).agg(n=("y","size"),nslot=("slot","nunique"),ysum=("y","sum")); P("\n(3) cow-days of the cows concerned after the change:",len(cd),"| recorded milkings per cow-day:",cd.n.value_counts(normalize=True).sort_index().round(3).to_dict(),"| distinct clock-time slots per cow-day:",cd.nslot.value_counts(normalize=True).sort_index().round(3).to_dict())
f4=cd[(cd.n==4)&(cd.nslot==4)].index; x=a.set_index(["cid","dnum"]).loc[f4].reset_index()
P("    cow-days with four milkings in four slots: %d ; interval before each milking (median h) and yield (median kg) by slot:"%len(f4)); P(x.groupby("slot").agg(dt=("dt","median"),y=("y","median"),hr=("hr","median")).round(2).to_string()); P("    daily sum on these cow-days: median %.1f kg (IQR %.1f-%.1f)"%(cd.loc[f4].ysum.median(),cd.loc[f4].ysum.quantile(.25),cd.loc[f4].ysum.quantile(.75)))
P("\n(4) yield (median kg) by length of the preceding interval, cows concerned after the change, by clock-time slot:")
a["ic"]=pd.cut(a.dt,[0,5,7,9,11.5,14,20,26,34],labels=["<5","5-7","7-9","9-11.5","11.5-14","14-20","20-26","26-34"]); t=a.groupby(["slot","ic"]).y.agg(["size","median"]).unstack("ic"); P(t["median"].round(2).to_string()); P("    n:"); P(t["size"].to_string())
r=a[(a.dt>=5)&(a.dt<=7)]; P("    secretion rate at 5-7 h intervals after the change: median %.3f kg/h (yield %.2f kg, n=%d)"%((r.y/r.dt).median(),r.y.median(),len(r)))
rb=b[(b.dt>=7)&(b.dt<=9)]; P("    the same cows before the change, 7-9 h intervals: median %.3f kg/h (yield %.2f kg, n=%d) ; intervals before the change: 5-7 h %.3f, 7-9 h %.3f"%((rb.y/rb.dt).median(),rb.y.median(),len(rb),((b.dt>=5)&(b.dt<7)).mean(),((b.dt>=7)&(b.dt<9)).mean()))
cb=b.groupby(["cid","dnum"]).agg(n=("y","size"),ysum=("y","sum")); P("    daily sum before the change on days with three recorded milkings: median %.1f kg (n=%d) ; after the change on days with four in four slots: %.1f kg"%(cb[cb.n==3].ysum.median(),(cb.n==3).sum(),cd.loc[f4].ysum.median()))
ro=o[(o.dt>=7)&(o.dt<=9)]; P("    other cows from June 2025, 7-9 h intervals: median %.3f kg/h (yield %.2f kg) ; intervals 5-7 h %.3f, 7-9 h %.3f ; recorded per cow-day %s"%((ro.y/ro.dt).median(),ro.y.median(),((o.dt>=5)&(o.dt<7)).mean(),((o.dt>=7)&(o.dt<9)).mean(),o.groupby(["cid","dnum"]).size().value_counts(normalize=True).sort_index().round(3).to_dict()))
P("\n(5) pattern of the recorded slots per cow-day, cows concerned after the change (N=night, M=morning, A=afternoon, E=evening):")
code=a.assign(c=a.slot.map({"night 00-05":"N","morning 05-12":"M","afternoon 12-19.5":"A","evening 19.5-24":"E"}).astype(str)).groupby(["cid","dnum"]).c.agg(lambda z:"".join(sorted(set(z),key="NMAE".index))); P(code.value_counts(normalize=True).head(12).round(3).to_dict())
day=a.groupby("dnum").agg(cows=("cid","nunique")); slots=a.groupby(["dnum","slot"]).cid.nunique().unstack("slot"); cov=slots.div(day.cows,axis=0); P("    share of the cows of a day recorded in each slot: mean over days",cov.mean().round(3).to_dict()); P("    days on which fewer than 20%% of the cows were recorded in the slot: %s of %d days"%((cov<0.2).sum().to_dict(),len(cov)))
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","dnum","DIM","par","my24","elig","farm_id"]); F=F[(F.farm_id==16)&(F.dnum>=D0)&F.elig].join(first4,on="cid"); F["g"]=np.where(F.d4.notna()&(F.dnum>=F.d4),"fourth shift","other")
P("\n(6) cows concerned against the other cows from June 2025 (eligible cow-days):"); P(F.groupby("g").agg(cowdays=("cid","size"),cows=("cid","nunique"),DIM=("DIM","median"),primiparous=("par",lambda z:(z==1).mean()),my24=("my24","median")).round(2).to_string())
P("DONE")
