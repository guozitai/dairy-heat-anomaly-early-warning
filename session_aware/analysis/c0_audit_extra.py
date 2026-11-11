#!/usr/bin/env python3
"""C0. Measurement audit, additional analyses requested in internal review.
(1) cleaning counts per step; (2) interval classes by farm, histogram to 36 h; (3) yield after long gaps: quantiles by interval class, coverage of the test,
test against the cow's usual yield in any shift, within-day comparison on two-milking days; (4) pattern of missing records by farm-day-shift;
(5) deviation noise and drop rate by number of usable milkings; rate- versus yield-based session deviation by farm; (6) coverage by farm-year;
(7) regular-interval-only deviation D_rate (written to cowday_extra.parquet together with the lagged-reference deviation)."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); W="/home/guozitai/20261111_sat_rescue/"; V=W+"v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",260); pd.set_option("display.max_columns",40)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}; L=np.log(0.8)
rs=lambda z: float((z-z.median()).abs().median()*1.4826)
# ---------- (1) cleaning counts
sc=pd.read_parquet(W+"sessions_clean.parquet",columns=["farm_id","cow","date","nrec","y"])
P("== cleaning == blocks after 60-min merge:",len(sc),"records in them:",int(sc.nrec.sum()),"| blocks with >1 record:",int((sc.nrec>1).sum()),"(records %d)"%int(sc.nrec[sc.nrec>1].sum()),"| blocks with >3 records:",int((sc.nrec>3).sum()),"(records %d)"%int(sc.nrec[sc.nrec>3].sum()),"| merged yield >45 kg:",int(((sc.nrec<=3)&(sc.y>45)).sum()))
P("   blocks with 2-3 records: %d (%.4f of blocks)"%(((sc.nrec>1)&(sc.nrec<=3)).sum(),((sc.nrec>1)&(sc.nrec<=3)).mean()))
del sc
ct=pd.read_parquet(V+"cowtab.parquet"); s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift","t","y","dt","valid","rate","ref","ref_y","use","use_rate"])
s=s.merge(ct[["cid","farm_id"]],on="cid"); P("milkings after cleaning (all farms):",len(s)); s=s[s.farm_id.isin(LAB)].copy(); s["farm"]=s.farm_id.map(LAB); P("milkings, eight farms:",len(s),"cows",s.cid.nunique(),round(time.time()-t0))
# ---------- (2) intervals
h=s.dropna(subset=["dt"]); bins=np.arange(0,36.25,0.25); H=pd.DataFrame({"h":bins[:-1]+0.125}); H["all"]=np.histogram(h.dt,bins)[0]
for f,x in h.groupby("farm"): H[f]=np.histogram(x.dt,bins)[0]
H["n_all_intervals"]=len(h); H.to_csv(V+"fig1_interval_hist36.csv",index=False)
def cls(x): tot=x.dt.notna().sum(); return pd.Series(dict(intervals=tot,lt5=(x.dt<5).sum()/tot,regular=((x.dt>=5)&(x.dt<=11.5)).sum()/tot,g11_20=((x.dt>11.5)&(x.dt<=20)).sum()/tot,g20_34=((x.dt>20)&(x.dt<=34)).sum()/tot,gt34=(x.dt>34).sum()/tot,g11_14_of_long=((x.dt>11.5)&(x.dt<=14)).sum()/max(((x.dt>11.5)&(x.dt<=34)).sum(),1),long_gaps=((x.dt>11.5)&(x.dt<=34)).sum()))
IC=pd.concat([s.groupby("farm").apply(cls),cls(s).to_frame("All").T]); P("\n== interval classes (share of all intervals) =="); P(IC.round(4).to_string()); IC.to_csv(V+"tab_interval_classes.csv")
P("modes of the interval distribution by farm (local maxima of the 0.25-h histogram with share >0.5%):")
for f in sorted(LAB.values()):
    y=H[f].to_numpy()/H[f].sum(); idx=[i for i in range(2,len(y)-2) if y[i]>0.005 and y[i]==y[i-2:i+3].max()]; P("  ",f,[(round(float(H.h[i]),2),round(float(100*y[i]),1)) for i in idx])
# ---------- (3) yield after long gaps
s=s.sort_values(["cid","shift","t"]).reset_index(drop=True); c=s.cid.to_numpy(); sh=pd.factorize(s["shift"])[0]; dn=s.dnum.to_numpy(); N=len(s)
yr=s.y.where(s.valid).to_numpy(dtype="float32"); M=np.full((8,N),np.nan,dtype="float32")
for k in range(1,9):
    ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,yr[:-k],np.nan)
cnt=np.sum(~np.isnan(M),axis=0); ry=np.nanmedian(M,axis=0); ry[cnt<3]=np.nan; del M; s["ry"]=ry; s["ratio"]=s.y/s.ry
# usual yield at regular-interval milkings in any shift, days t-8..t-2 (>=3 values)
s=s.sort_values(["cid","t"]).reset_index(drop=True); c=s.cid.to_numpy(); dn=s.dnum.to_numpy(); yr=s.y.where(s.valid).to_numpy(dtype="float32"); K=30; M=np.full((K,N),np.nan,dtype="float32")
for k in range(1,K+1):
    ok=(c[k:]==c[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); M[k-1,k:]=np.where(ok,yr[:-k],np.nan)
cnt=np.sum(~np.isnan(M),axis=0); ra=np.nanmedian(M,axis=0); ra[cnt<3]=np.nan; del M; s["ry_any"]=ra; s["ratio_any"]=s.y/s.ry_any
cat=pd.cut(s.dt,[5,11.5,14,18,22,26,34],labels=["5-11.5 h","11.5-14 h","14-18 h","18-22 h","22-26 h","26-34 h"]); s["cat"]=cat
q=lambda x: pd.Series(dict(n=len(x),q05=x.quantile(.05),q25=x.quantile(.25),median=x.median(),q75=x.quantile(.75),q95=x.quantile(.95)))
G=s.dropna(subset=["ratio","cat"]).groupby("cat",observed=True).ratio.apply(q).unstack(); P("\n== yield relative to usual same-shift yield (regular-interval reference) =="); P(G.round(3).to_string()); G.to_csv(V+"fig1_gap_ratio_quantiles.csv")
lg=s[(s.dt>11.5)&(s.dt<=34)]; P("long gaps (11.5-34 h): %d ; with same-shift regular-interval reference: %d (%.3f) ; with any-shift reference: %d (%.3f) ; with neither: %d (%.3f)"%(len(lg),lg.ry.notna().sum(),lg.ry.notna().mean(),lg.ry_any.notna().sum(),lg.ry_any.notna().mean(),(lg.ry.isna()&lg.ry_any.isna()).sum(),(lg.ry.isna()&lg.ry_any.isna()).mean()))
P("coverage of the same-shift test by farm:"); P(lg.groupby("farm").agg(long_gaps=("y","size"),same_shift_ref=("ry",lambda x:x.notna().mean()),any_shift_ref=("ry_any",lambda x:x.notna().mean())).round(3).to_string())
lg=lg.assign(grp=np.where(lg.ry.notna(),"same-shift reference",np.where(lg.ry_any.notna(),"any-shift reference only","no reference")))
GA=lg.dropna(subset=["ratio_any","cat"]).groupby(["grp","cat"],observed=True).ratio_any.apply(q).unstack(); P("\n== long gaps: yield relative to usual yield at regular-interval milkings in any shift =="); P(GA.round(3).to_string()); GA.to_csv(V+"tab_gap_anyshift.csv")
P("by farm, gaps without same-shift reference, ratio to any-shift usual yield (median):"); P(lg[(lg.grp=="any-shift reference only")].dropna(subset=["ratio_any","cat"]).groupby(["farm","cat"],observed=True).ratio_any.median().unstack().round(3).to_string())
regy=s[s.valid].groupby("farm").y.median(); P("\nmedian yield (kg) of regular-interval milkings by farm:",regy.round(2).to_dict())
P("median yield (kg) of long-gap milkings by reference group and class:"); P(lg.dropna(subset=["cat"]).groupby(["grp","cat"],observed=True).y.agg(["size","median"]).round(2).to_string())
P("median yield (kg) of long-gap milkings without any reference, by farm:"); P(lg[lg.grp=="no reference"].groupby("farm").y.agg(["size","median"]).round(2).to_string())
# 11.5-14 h by farm
P("\n11.5-14 h gaps by farm (same-shift ratio):"); P(s[(s.cat=="11.5-14 h")&s.ratio.notna()].groupby("farm").ratio.agg(["size","median"]).round(3).to_string())
# within-day comparison on cow-days with exactly two recorded milkings: one after a regular interval, one after a long gap
nd=s.groupby(["cid","dnum"]).y.transform("size"); two=s[nd==2].copy(); two["long"]=(two.dt>11.5)&(two.dt<=26); two["reg"]=two.valid
a=two[two.reg][["cid","dnum","y","dt","farm"]].rename(columns={"y":"y_reg","dt":"dt_reg"}); b=two[two.long][["cid","dnum","y","dt","ry"]].rename(columns={"y":"y_long","dt":"dt_long"})
pr=a.merge(b,on=["cid","dnum"]); pr=pr[pr.groupby(["cid","dnum"]).y_reg.transform("size")==1]; pr["r"]=pr.y_long/pr.y_reg; pr["exp_if_omitted"]=pr.dt_long/pr.dt_reg; pr["hasref"]=pr.ry.notna()
P("\n== two-milking cow-days with one regular-interval and one long-gap milking: n=%d (of %d two-milking cow-days) =="%(len(pr),int((nd==2).sum()/2)))
P("   yield after long gap / yield after regular interval: median %.3f (IQR %.3f-%.3f); ratio of the two intervals: median %.2f"%(pr.r.median(),pr.r.quantile(.25),pr.r.quantile(.75),pr.exp_if_omitted.median()))
P(pr.groupby("hasref").agg(n=("r","size"),median=("r","median"),q25=("r",lambda x:x.quantile(.25)),q75=("r",lambda x:x.quantile(.75)),interval_ratio=("exp_if_omitted","median")).round(3).to_string())
TW=pr.groupby(["farm","hasref"]).agg(n=("r","size"),median=("r","median"),interval_ratio=("exp_if_omitted","median")).round(3); P(TW.to_string()); TW.to_csv(V+"tab_two_milking_days.csv")
pr.groupby("hasref").r.apply(q).unstack().to_csv(V+"tab_two_milking_quant.csv")
# ---------- (4) pattern of missing records by farm-day-shift
ro=s.groupby(["farm","dnum"]).cid.nunique().rename("roster"); fs=s.groupby(["farm","dnum","shift"]).cid.nunique().rename("rec").reset_index().merge(ro.reset_index(),on=["farm","dnum"])
allsh=s[["farm","shift"]].drop_duplicates(); full=ro.reset_index().merge(allsh,on="farm").merge(fs[["farm","dnum","shift","rec"]],on=["farm","dnum","shift"],how="left").fillna({"rec":0}); full["cov"]=full.rec/full.roster; full["miss"]=full.roster-full.rec
P("\n== missing records by farm-day-shift == shifts per farm:",allsh.groupby("farm").size().to_dict())
def pat(x): tm=x.miss.sum(); return pd.Series(dict(farm_day_shifts=len(x),share_shifts_cov_lt20=(x["cov"]<0.2).mean(),share_shifts_cov_20_80=((x["cov"]>=0.2)&(x["cov"]<0.8)).mean(),share_shifts_cov_ge80=(x["cov"]>=0.8).mean(),missing_records=tm,miss_in_lt20=x.miss[x["cov"]<0.2].sum()/tm,miss_in_20_80=x.miss[(x["cov"]>=0.2)&(x["cov"]<0.8)].sum()/tm,miss_in_ge80=x.miss[x["cov"]>=0.8].sum()/tm,median_cov=x["cov"].median()))
PT=pd.concat([full.groupby("farm").apply(pat),pat(full).to_frame("All").T]); P(PT.round(3).to_string()); PT.to_csv(V+"tab_missing_pattern.csv")
P("by shift label (all farms): mean coverage"); P(full.groupby(["farm","shift"])["cov"].mean().unstack().round(3).to_string())
# timestamp granularity: distinct start times per farm-day-shift relative to cows recorded
tg=s.groupby(["farm","dnum","shift"]).agg(n=("cid","size"),ut=("t","nunique")).reset_index(); P("distinct start times / milkings per farm-day-shift (median by farm):",(tg.ut/tg.n).groupby(tg.farm).median().round(3).to_dict())
# rate- vs yield-based session deviation for regular-interval milkings
reg=s[s.valid&s.ref.notna()&s.ref_y.notna()]; P("\n== robust SD of the milking-level deviation, regular-interval milkings: rate-based | yield-based | SD of interval (h) within cow-shift =="); 
P(reg.assign(dr=np.log(reg.rate/reg.ref),dy=np.log(reg.y/reg.ref_y)).groupby("farm").agg(n=("y","size"),rate_based=("dr",rs),yield_based=("dy",rs),sd_dt=("dt","std")).round(4).to_string())
# D from regular-interval milkings only
s["yoR"]=np.where(s.use_rate,s.y,np.nan); s["yeR"]=np.where(s.use_rate,s.ref*s.dt,np.nan); dR=s.groupby(["cid","dnum"],sort=True).agg(yoR=("yoR","sum"),yeR=("yeR","sum"),nR=("use_rate","sum")).reset_index(); dR["D_rate"]=np.log(dR.yoR/dR.yeR).where(dR.nR>=1).astype("float32")
lr=pd.read_parquet(V+"cowday_lagref.parquet"); ex=dR[["cid","dnum","D_rate","nR"]].merge(lr[["cid","dnum","DL"]],on=["cid","dnum"],how="outer"); ex.to_parquet(V+"cowday_extra.parquet",index=False); P("cowday_extra written",len(ex),round(time.time()-t0)); del s
# ---------- (5) cow-day level, eligible cow-days
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","D","dev_raw","elig","n_sess","n_valid","n_dev","my24","y_sum","risk_dis","y_dis","risk_onset","y_onset","y_onset_sus","D_next","D_next2","DIM","par"]); F=F[F.elig].merge(ex,on=["cid","dnum"],how="left"); F["farm"]=F.farm_id.map(LAB)
P("\n== eligible cow-days: %d ; cows %d ; recorded milkings on them: %d ; share with >3 recorded milkings %.4f =="%(len(F),F.cid.nunique(),int(F.n_sess.sum()),(F.n_sess>3).mean()))
cc=F[(F.n_sess==3)&(F.n_valid==3)]; r=cc.my24/cc.y_sum; P("three recorded milkings and three regular intervals: n=%d ; 24-h equivalent / daily sum median %.4f (IQR %.4f-%.4f)"%(len(cc),r.median(),r.quantile(.25),r.quantile(.75)))
P("robust SD: D %.5f | daily-sum deviation %.5f | D_rate %.5f | DL %.5f"%(rs(F.D.dropna()),rs(F.dev_raw.dropna()),rs(F.D_rate.dropna()),rs(F.DL.dropna())))
x=F[F.D.notna()]; nd=x.n_dev.clip(upper=3)
BY=x.groupby(nd).agg(cow_days=("D","size"),robust_sd=("D",rs),drop_share=("D",lambda z:(z<L).mean()),median_D=("D","median")).reset_index(); BY["share"]=BY.cow_days/len(x)
rk=F[F.risk_dis]; rk=rk.assign(nd=rk.n_dev.clip(upper=3),dr=rk.D<L); T2=rk.groupby(["nd","dr"]).y_dis.agg(["size","mean"]).unstack(); P("\n== by number of usable milkings =="); P(BY.round(4).to_string()); P("P(new disease record within 3 d) by usable milkings and drop status (risk set):"); P(T2.round(5).to_string())
BY.to_csv(V+"tab_by_usable.csv",index=False); T2.to_csv(V+"tab_by_usable_disease.csv")
P("by farm: robust SD and drop share by usable milkings:"); P(x.assign(nd=nd).groupby(["farm","nd"]).agg(n=("D","size"),sd=("D",rs),drop=("D",lambda z:(z<L).mean())).unstack().round(3).to_string())
on=F[F.risk_onset&(F.y_onset==1)]; P("\nonsets %d: followed by a day with a deviation %.3f ; of those, second day at least 10%% below expected %.3f ; second day still in acute drop %.3f"%(len(on),on.D_next2.notna().mean(),(on.D_next2[on.D_next2.notna()]<np.log(0.9)).mean(),(on.D_next2[on.D_next2.notna()]<L).mean()))
no=F[F.risk_onset&(F.y_onset==0)]; P("   among cow-days at risk without onset: day t+2 at least 10%% below expected %.3f ; in acute drop %.3f"%((no.D_next2[no.D_next2.notna()]<np.log(0.9)).mean(),(no.D_next2[no.D_next2.notna()]<L).mean()))
P("drop share with regular-interval milkings only (D_rate): %.4f (n=%d) ; with lagged reference (DL): %.4f (n=%d)"%((F.D_rate.dropna()<L).mean(),F.D_rate.notna().sum(),(F.DL.dropna()<L).mean(),F.DL.notna().sum()))
m=F.assign(warm=F.date.dt.month.isin([5,6,7,8,9])); P("onset rate May-Sep %.4f | Oct-Apr %.4f ; drop share May-Sep %.4f | Oct-Apr %.4f"%(m[m.risk_onset&m.warm].y_onset.mean(),m[m.risk_onset&~m.warm].y_onset.mean(),(m[m.warm].D.dropna()<L).mean(),(m[~m.warm].D.dropna()<L).mean()))
# ---------- (6) coverage by farm-year
F["year"]=F.date.dt.year; cd=F.groupby(["farm","year","date"]).agg(cows=("cid","size"),withD=("D","count")).reset_index()
CV=cd.groupby(["farm","year"]).agg(first=("date","min"),last=("date","max"),days_with_records=("date","size"),days_ge200=("withD",lambda z:(z>=200).sum()),median_cows=("cows","median"),median_withD=("withD","median")).reset_index(); CV["calendar_days"]=(CV["last"]-CV["first"]).dt.days+1
tot=F.groupby(["farm","year"]).agg(cow_days=("D","size"),with_D=("D","count"),risk_onset=("risk_onset","sum"),onsets=("y_onset","sum"),risk_dis=("risk_dis","sum")).reset_index(); CV=CV.merge(tot,on=["farm","year"]); P("\n== coverage by farm-year =="); P(CV.to_string()); CV.to_csv(V+"tab_coverage_farm_year.csv",index=False)
P("Task 2 risk set per farm-day: median by farm:",F[F.risk_dis].groupby(["farm","dnum"]).size().groupby("farm").median().to_dict())
P("Task 1 deployable risk set (D valid, not in drop) per farm-day: median by farm:",F[F.D.notna()&(F.D>=L)].groupby(["farm","dnum"]).size().groupby("farm").median().to_dict())
P("DONE",round(time.time()-t0))
