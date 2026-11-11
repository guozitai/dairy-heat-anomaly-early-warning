#!/usr/bin/env python3
"""B1. Measurement audit: intervals between recorded milkings, yield after long gaps, recorded milkings per cow-day,
flagged cow-days under raw and session-aware measurement, composition of the originally submitted label. Outputs CSVs for Table 1 / Fig 1."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time()
V="/home/guozitai/20261111_sat_rescue/v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
ct=pd.read_parquet(V+"cowtab.parquet"); s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt","valid","rate","ref"])
s=s.merge(ct[["cid","farm_id"]],on="cid"); s=s[s.farm_id.isin(LAB)].copy(); s["farm"]=s.farm_id.map(LAB)
P("milkings (8 farms):",len(s),"cows:",s.cid.nunique())
# interval histogram
h=s.dropna(subset=["dt"]); h=h[h.dt<=30]; bins=np.arange(0,30.25,0.25)
H=pd.DataFrame({"h":bins[:-1]+0.125}); H["all"]=np.histogram(h.dt,bins)[0]
for f,x in h.groupby("farm"): H[f]=np.histogram(x.dt,bins)[0]
H.to_csv(V+"fig1_interval_hist.csv",index=False)
tot=s.dt.notna().sum(); P("interval shares: regular 5-11.5 h %.4f | 11.5-20 h %.4f | 20-34 h %.4f | >34 h %.4f | <5 h %.4f"%(((s.dt>=5)&(s.dt<=11.5)).sum()/tot,((s.dt>11.5)&(s.dt<=20)).sum()/tot,((s.dt>20)&(s.dt<=34)).sum()/tot,(s.dt>34).sum()/tot,(s.dt<5).sum()/tot))
# gap test: yield vs usual same-shift yield from regular-interval milkings (days t-8..t-2)
s=s.sort_values(["cid","shift","t"]).reset_index(drop=True); c=s.cid.to_numpy(); sh=s["shift"].to_numpy(); dn=s.dnum.to_numpy(); N=len(s)
yr=s.y.where(s.valid).to_numpy(dtype="float32"); M=np.full((8,N),np.nan,dtype="float32")
for k in range(1,9):
    ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,yr[:-k],np.nan)
cnt=np.sum(~np.isnan(M),axis=0); ry=np.nanmedian(M,axis=0); ry[cnt<3]=np.nan; del M; s["ratio"]=s.y/ry
cat=pd.cut(s.dt,[5,11.5,14,18,22,26],labels=["regular (5-11.5 h)","11.5-14 h","one milking missing (14-18 h)","18-22 h","two milkings missing (22-26 h)"])
g=s.dropna(subset=["ratio"]).groupby(cat,observed=True).ratio; G=g.agg(n="size",median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index(); P("\nyield relative to usual same-shift yield:"); P(G.round(3).to_string())
rb=np.arange(0,3.02,0.02); RH=pd.DataFrame({"ratio":rb[:-1]+0.01})
for nm in ["regular (5-11.5 h)","one milking missing (14-18 h)","two milkings missing (22-26 h)"]:
    x=s.ratio[(cat==nm)&s.ratio.notna()]; RH[nm]=np.histogram(x,rb)[0]/len(x)
RH.to_csv(V+"fig1_gap_ratio_hist.csv",index=False)
byf=s.dropna(subset=["ratio"]).assign(cat=cat).groupby(["farm","cat"],observed=True).ratio.median().unstack(); P("\nmedian ratio by farm:"); P(byf.round(3).to_string()); byf.to_csv(V+"tab_gap_by_farm.csv")
icar=s[(s.dt>11.5)&(s.dt<=34)&s.ref.notna()]; P("ICAR criterion among long gaps with reference: %.4f (n=%d)"%(((icar.y/icar.dt)<0.8*icar.ref).mean(),len(icar)))
del s
# cow-day level
d=pd.read_parquet(V+"cowday_v2.parquet",columns=["cid","farm_id","cow","date","n_sess","n_valid","n_dev","y_sum","my24","D","dev_raw","dev_my24","DIM"]); d=d[d.farm_id.isin(LAB)].copy(); d["farm"]=d.farm_id.map(LAB); L=np.log(0.8)
t1=d.groupby("farm").agg(cow_days=("D","size"),cows=("cid","nunique"),s3=("n_sess",lambda x:(x>=3).mean()),s2=("n_sess",lambda x:(x==2).mean()),s1=("n_sess",lambda x:(x==1).mean()),withD=("D",lambda x:x.notna().mean()),
    flag_raw=("dev_raw",lambda x:(x.dropna()<L).mean()),flag_D=("D",lambda x:(x.dropna()<L).mean()),sd_raw=("dev_raw",lambda x:(x.dropna()-x.dropna().median()).abs().median()*1.4826),sd_D=("D",lambda x:(x.dropna()-x.dropna().median()).abs().median()*1.4826),my24=("my24","median")).reset_index()
P("\nper farm:"); P(t1.round(3).to_string()); t1.to_csv(V+"tab1_measurement_by_farm.csv",index=False)
P("all 8 farms: recorded milkings per cow-day 3+: %.3f | 2: %.3f | 1: %.3f ; flagged raw %.4f vs session-aware %.4f"%((d.n_sess>=3).mean(),(d.n_sess==2).mean(),(d.n_sess==1).mean(),(d.dev_raw.dropna()<L).mean(),(d.D.dropna()<L).mean()))
b=d[d.dev_raw.notna()&d.D.notna()]; P("of raw-sum flags, share also flagged session-aware: %.3f ; of session-aware flags, share also raw: %.3f"%((b.D<L)[b.dev_raw<L].mean(),(b.dev_raw<L)[b.D<L].mean()))
cc=d[(d.n_sess==3)&(d.n_valid==3)]; P("complete days %d: MY24/sum median %.4f (IQR %.4f-%.4f)"%(len(cc),(cc.my24/cc.y_sum).median(),(cc.my24/cc.y_sum).quantile(.25),(cc.my24/cc.y_sum).quantile(.75)))
# originally submitted label (test folds of the submitted pipeline)
p=pd.read_parquet(D0+"v2_predictions.parquet",columns=["farm_id","bullNumber","milkDate","event","farm_label"]).rename(columns={"bullNumber":"cow","milkDate":"date"}); p["cow"]=p.cow.astype(str)
p=p.merge(d[["farm_id","cow","date","n_sess","D","dev_raw"]],on=["farm_id","cow","date"],how="left")
md=p.groupby(["farm_id","date"]).n_sess.transform(lambda x:x.mode().iloc[0] if x.notna().any() else np.nan); p["fewer"]=p.n_sess<md
cs=p.groupby(["farm_id","cow"]).event.agg(["mean","sum","size"]); ch=cs[(cs["size"]>=60)&(cs["mean"]>0.2)]["sum"].sum()/cs[cs["size"]>=60]["sum"].sum()
P("\nsubmitted label, test folds: n=%d prevalence %.4f; share of labelled events on cow-days with fewer recorded milkings than the herd mode: %.4f; event rate on such days %.4f vs %.4f; share of events from cows flagged on >20%% of days: %.3f; share of events that are session-aware drops: %.3f; share of session-aware drops labelled as events: %.3f"%(len(p),p.event.mean(),p[p.event==1].fewer.mean(),p[p.fewer].event.mean(),p[~p.fewer].event.mean(),ch,(p.D<L)[(p.event==1)&p.D.notna()].mean(),p.event[(p.D<L)].mean()))
o=p.groupby("farm_label").apply(lambda g:pd.Series(dict(n=len(g),prev=g.event.mean(),events_on_incomplete=g[g.event==1].fewer.mean(),rate_incomplete=g[g.fewer].event.mean(),rate_complete=g[~g.fewer].event.mean()))).reset_index(); P(o.round(3).to_string()); o.to_csv(V+"tab1_submitted_label_by_farm.csv",index=False)
P("DONE",round(time.time()-t0))
