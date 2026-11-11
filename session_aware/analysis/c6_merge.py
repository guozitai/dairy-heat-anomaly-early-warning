#!/usr/bin/env python3
"""C6. Merge the parts of the early-warning run and print the summaries used in the text."""
import pandas as pd, numpy as np, glob, warnings
from scipy import stats
warnings.filterwarnings("ignore"); V="/home/guozitai/20261111_sat_rescue/v2/"; C=V+"c1/"
pd.set_option("display.width",280); pd.set_option("display.max_columns",80); pd.set_option("display.max_rows",400)
def P(*a): print(*a,flush=True)
cat=lambda nm: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(C+f"{nm}_part*.csv"))],ignore_index=True)
key=lambda s:(int(s.split()[1]),int(s.split()[2]))
R1=cat("ew1"); R2=cat("ew2"); S=cat("sens"); LT=cat("lead"); IM=cat("imp"); SD=cat("seeds"); FO=cat("folds"); CA=cat("cal")
for nm,x in (("ew_onset",R1),("ew_disease",R2),("sens_analyses",S),("ew_leadtime",LT),("ew_importance",IM),("ew_seeds",SD),("ew_folds",FO),("ew_calibration",CA)): x.to_csv(V+nm+".csv",index=False)
cases=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(C+"cases_*.parquet"))],ignore_index=True); cases.to_parquet(V+"ew_cases.parquet",index=False)
A="R_deviation"; B="M_ref"; folds=sorted(R1.fold.unique(),key=key); P("farm-years Task 1:",len(folds),"| Task 2:",R2.fold.nunique())
def sign_p(k,n): return min(1.0,2*stats.binom.cdf(min(k,n-k),n,0.5))
# ---------------- Task 1
P("\n===== TASK 1 ====="); n1=R1[R1.config==A]; P("risk-set cow-days %d ; with observed outcome %d (%.4f) ; onsets %d (rate %.4f pooled; mean of farm-years %.4f) ; confirmed onsets %d (%.3f of onsets)"%(n1.n.sum(),n1.n_obs.sum(),n1.n_obs.sum()/n1.n.sum(),n1.events.sum(),n1.events.sum()/n1.n_obs.sum(),n1.rate.mean(),n1.n_conf.sum(),n1.n_conf.sum()/n1.events.sum()))
g=R1.groupby("config"); T=pd.DataFrame({"sens10":g.sens10.mean(),"min":g.sens10.min(),"max":g.sens10.max(),"prec10":g.prec10.mean(),"entries":g.entries_per_detection.mean(),"conf":g.sens10_conf.mean(),"alt":g.sens10_alt.mean(),"pr_lift":g.pr_lift.mean(),"auc":g.auc.mean(),"unobs_listed":g.unobs_listed.mean(),"unobs_unlisted":g.unobs_unlisted.mean()}).sort_values("sens10"); P((T*100).round(2).assign(entries=T.entries.round(2),pr_lift=T.pr_lift.round(3),auc=T.auc.round(3)).to_string()); P("chance (mean): %.4f"%n1.chance10.mean())
pv=R1.pivot(index="fold",columns="config",values="sens10").loc[folds]; pc=R1.pivot(index="fold",columns="config",values="sens10_conf").loc[folds]; rules=[c for c in pv.columns if c.startswith("R_")]; best=pv[rules].max(axis=1)
for c in pv.columns:
    if c!=A: P(f"   {c:34s} above deviation rule in {(pv[c]>pv[A]).sum()}/{len(pv)} farm-years (confirmed onsets {(pc[c]>pc[A]).sum()}/{len(pc)}); mean diff {100*(pv[c]-pv[A]).mean():+.2f} pp")
P("model above every rule in %d/%d farm-years; sign test model vs rule P=%.5f"%((pv[B]>best).sum(),len(pv),sign_p((pv[B]>pv[A]).sum(),len(pv))))
d=pv["M_ref+weather"]-pv[B]; P("weather added: mean %.2f pp (range %.2f..%.2f); lower in %d/%d farm-years"%(100*d.mean(),100*d.min(),100*d.max(),(d<0).sum(),len(d)))
x=R1[R1.config.isin([A,B,"M_logistic"])]; P("by year:"); P((x.pivot_table(index="year",columns="config",values=["sens10","sens10_conf"],aggfunc="mean")*100).round(2).to_string())
fm=x.pivot_table(index="farm",columns="config",values="sens10",aggfunc="mean"); df_=fm[B]-fm[A]; tci=stats.t.interval(0.95,len(df_)-1,loc=df_.mean(),scale=df_.std(ddof=1)/np.sqrt(len(df_))); P("by farm (mean of two years): model above rule in %d/8 (sign test P=%.4f); mean difference %.2f pp (95%% CI %.2f to %.2f)"%((df_>0).sum(),sign_p((df_>0).sum(),8),100*df_.mean(),100*tci[0],100*tci[1])); P((fm*100).round(2).to_string())
fmc=x.pivot_table(index="farm",columns="config",values="sens10_conf",aggfunc="mean"); dc=fmc[B]-fmc[A]; tcc=stats.t.interval(0.95,7,loc=dc.mean(),scale=dc.std(ddof=1)/np.sqrt(8)); P("confirmed onsets by farm: model above rule in %d/8; mean difference %.2f pp (95%% CI %.2f to %.2f)"%((dc>0).sum(),100*dc.mean(),100*tcc[0],100*tcc[1]))
P("per farm-year (rule, CI, model, CI, diff CI, +weather, weather only, logistic, confirmed rule/model):")
for f in folds:
    a=R1[(R1.fold==f)&(R1.config==A)].iloc[0]; b=R1[(R1.fold==f)&(R1.config==B)].iloc[0]; P(f"   {f:12s} {100*a.sens10:5.1f} ({100*a.lo:.1f}-{100*a.hi:.1f}) | {100*b.sens10:5.1f} ({100*b.lo:.1f}-{100*b.hi:.1f}) | diff {100*b.diff_lo:+.1f}..{100*b.diff_hi:+.1f} | {100*pv.loc[f,'M_ref+weather']:.1f} | {100*pv.loc[f,'M_weather+DIM+parity']:.1f} | {100*pv.loc[f,'M_logistic']:.1f} | conf {100*a.sens10_conf:.1f} / {100*b.sens10_conf:.1f} (diff {100*b.conf_diff_lo:+.1f}..{100*b.conf_diff_hi:+.1f})")
P("bootstrap: model-rule interval excludes zero (model higher / rule higher): %d / %d ; confirmed: %d / %d"%((R1[R1.config==B].diff_lo>0).sum(),(R1[R1.config==B].diff_hi<0).sum(),(R1[R1.config==B].conf_diff_lo>0).sum(),(R1[R1.config==B].conf_diff_hi<0).sum()))
s1=SD[SD.task=="onset"].groupby("seed")[["sens","sens_conf"]].mean(); P("seeds (mean over farm-years):",(s1.sens*100).round(2).to_dict(),"SD %.3f pp ; iterations range %d-%d"%(100*s1.sens.std(ddof=1),SD[SD.task=="onset"].iterations.min(),SD[SD.task=="onset"].iterations.max()))
u=R1[R1.config.isin([A,B])].groupby("config")[["sens10_u1","sens10_u2","sens10_u3"]].mean()*100; P("sensitivity for onsets of cows with 1/2/3 usable milkings on day t:"); P(u.round(2).to_string())
# ---------------- Task 2
P("\n===== TASK 2 ====="); n2=R2[R2.config==A]; P("risk-set cow-days %d ; cases %d ; positive cow-days %d ; chance 2/5/10%%: %.4f %.4f %.4f ; chance entries per case 5%%: %.1f ; cases with 1/2/3 risk days: %s"%(n2.n.sum(),n2.cases.sum(),n2.pos.sum(),n2.chance2.mean(),n2.chance5.mean(),n2.chance10.mean(),n2.chance_entries5.mean(),FO[["t2_cases_1day","t2_cases_2days","t2_cases_3days"]].sum().to_dict()))
g=R2.groupby("config"); cols=["case_sens2","case_sens5","case_sens10","cowday_sens5","cowday_sens10","prec5","prec10","entries_per_case5","entries_per_case10","case_sens5_early","case_sens5_d1","auc","pr_lift"]; T2=g[cols].mean().assign(min5=g.case_sens5.min(),max5=g.case_sens5.max()).sort_values("case_sens5"); P(T2.round(4).to_string())
p5=R2.pivot(index="fold",columns="config",values="case_sens5").loc[folds]; p10=R2.pivot(index="fold",columns="config",values="case_sens10").loc[folds]
for c in p5.columns:
    if c!=A: P(f"   {c:34s} above deviation rule in {(p5[c]>p5[A]).sum()}/{len(p5)} at 5% ({(p10[c]>p10[A]).sum()}/{len(p10)} at 10%); ties {(p5[c]==p5[A]).sum()}; mean diff {100*(p5[c]-p5[A]).mean():+.2f} pp")
P("rule above cohort ratio in %d/%d; above daily-sum deviation in %d/%d; above 24-h equivalent deviation in %d/%d"%((p5[A]>p5["R_dailysum_cohort_ratio"]).sum(),len(p5),(p5[A]>p5["R_dailysum_deviation"]).sum(),len(p5),(p5[A]>p5["R_my24_deviation"]).sum(),len(p5)))
nf7=[f for f in folds if not f.startswith("Farm 7")]; P("without Farm 7: rule %.2f model %.2f (diff %.2f pp)"%(100*p5.loc[nf7,A].mean(),100*p5.loc[nf7,B].mean(),100*(p5.loc[nf7,B]-p5.loc[nf7,A]).mean()))
x=R2[R2.config.isin([A,B,"M_logistic"])]; P("by year:"); P((x.pivot_table(index="year",columns="config",values=["case_sens5","case_sens10"],aggfunc="mean")*100).round(2).to_string())
fm=x.pivot_table(index="farm",columns="config",values="case_sens5",aggfunc="mean"); df_=fm[B]-fm[A]; tci=stats.t.interval(0.95,7,loc=df_.mean(),scale=df_.std(ddof=1)/np.sqrt(8)); P("by farm (mean of two years): model above rule in %d/8; mean difference %.2f pp (95%% CI %.2f to %.2f)"%((df_>0).sum(),100*df_.mean(),100*tci[0],100*tci[1])); P((fm*100).round(2).to_string())
P("bootstrap 5%%: model-rule interval excludes zero (model higher / rule higher): %d / %d ; 10%%: %d / %d"%((R2[R2.config==B].d5_lo>0).sum(),(R2[R2.config==B].d5_hi<0).sum(),(R2[R2.config==B].d10_lo>0).sum(),(R2[R2.config==B].d10_hi<0).sum()))
P("per farm-year (chance, cohort, daily-sum dev, rule CI, model CI, diff CI, logistic, blend, 10%: rule model):")
for f in folds:
    a=R2[(R2.fold==f)&(R2.config==A)].iloc[0]; b=R2[(R2.fold==f)&(R2.config==B)].iloc[0]; P(f"   {f:12s} n={int(a.cases):5d} {100*a.chance5:5.1f} | {100*p5.loc[f,'R_dailysum_cohort_ratio']:5.1f} | {100*p5.loc[f,'R_dailysum_deviation']:5.1f} | {100*a.case_sens5:5.1f} ({100*a.cs5_lo:.1f}-{100*a.cs5_hi:.1f}) | {100*b.case_sens5:5.1f} ({100*b.cs5_lo:.1f}-{100*b.cs5_hi:.1f}) | {100*b.d5_lo:+.1f}..{100*b.d5_hi:+.1f} | {100*p5.loc[f,'M_logistic']:.1f} | {100*p5.loc[f,'B_rank_mean(rule,model)']:.1f} | {100*a.case_sens10:.1f} {100*b.case_sens10:.1f}")
P("pooled over cases: n=%d rule5 %.4f model5 %.4f blend5 %.4f rule10 %.4f model10 %.4f | early-only (cases with a risk day at -3/-2: %d): rule %.4f model %.4f | daily-sum %.4f cohort %.4f"%(len(cases),cases.rule5.mean(),cases.model5.mean(),cases.blend5.mean(),cases.rule10.mean(),cases.model10.mean(),cases.has_early.sum(),cases.rule5_early[cases.has_early].mean(),cases.model5_early[cases.has_early].mean(),cases.dailysum5.mean(),cases.cohort5.mean()))
pd.DataFrame([dict(n=len(cases),rule5=cases.rule5.mean(),model5=cases.model5.mean(),blend5=cases.blend5.mean(),rule10=cases.rule10.mean(),model10=cases.model10.mean(),n_early=int(cases.has_early.sum()),rule5_early=cases.rule5_early[cases.has_early].mean(),model5_early=cases.model5_early[cases.has_early].mean(),dailysum5=cases.dailysum5.mean(),cohort5=cases.cohort5.mean())]).to_csv(V+"ew_pooled.csv",index=False)
rng=np.random.default_rng(1); CL=[]
for g_,x in cases.groupby("grp"):
    cw=pd.factorize(x.cid)[0]; bs=[]
    for _ in range(1000):
        w=rng.poisson(1.0,cw.max()+1)[cw]; sw=max(w.sum(),1); bs.append(((w*x.rule5).sum()/sw,(w*x.model5).sum()/sw,(w*(x.model5-x.rule5)).sum()/sw,(w*x.rule10).sum()/sw,(w*x.model10).sum()/sw))
    bs=np.array(bs); q=lambda j: np.percentile(bs[:,j],[2.5,97.5]); CL.append(dict(grp=g_,n=len(x),rule5=x.rule5.mean(),rule5_lo=q(0)[0],rule5_hi=q(0)[1],model5=x.model5.mean(),model5_lo=q(1)[0],model5_hi=q(1)[1],diff_lo=q(2)[0],diff_hi=q(2)[1],rule10=x.rule10.mean(),model10=x.model10.mean(),cohort5=x.cohort5.mean(),dailysum5=x.dailysum5.mean()))
CL=pd.DataFrame(CL); CL.to_csv(V+"ew_by_class_pooled.csv",index=False); P("by class, pooled over cases (bootstrap over cows):"); P(CL.round(4).to_string())
gc=[c for c in R2.columns if c.startswith("case_sens5_") and c.split("_")[-1] in ("mastitis","digestive","metabolic","lameness","respiratory","reproductive","other")]; P("by class, mean over farm-years with >=20 cases:"); P((R2[R2.config.isin([A,B,"R_dailysum_cohort_ratio"])].groupby("config")[gc].mean()*100).round(2).T.to_string())
hm=R2[R2.config.isin([A,B])].dropna(subset=["hot_sens10","mild_sens10"]); P("heat: farm-years with both classes: %d"%hm.fold.nunique())
for c,z in hm.groupby("config"):
    d=z.hot_sens10-z.mild_sens10; d5=z.hot_sens5-z.mild_sens5; P(f"   {c}: 10% lists hot {100*z.hot_sens10.mean():.2f} mild {100*z.mild_sens10.mean():.2f} diff {100*d.mean():+.2f} (range {100*d.min():+.1f}..{100*d.max():+.1f}; hot higher in {(d>0).sum()}/{len(d)}; sign P={sign_p((d>0).sum(),len(d)):.3f}) | 5% lists hot {100*z.hot_sens5.mean():.2f} mild {100*z.mild_sens5.mean():.2f} (hot higher in {(d5>0).sum()}/{len(d5)})")
lt=LT.assign(w=LT.on_list*LT.n,wc=LT.chance*LT.n).groupby(["config","days_before"])[["w","wc","n"]].sum(); lt["on_list"]=lt.w/lt.n; lt["chance"]=lt.wc/lt.n; P("lead time (share of cows on the 5% list k days before the record):"); P((lt[["on_list","chance"]]*100).round(2).unstack(0).to_string())
s2=SD[SD.task=="disease"].groupby("seed")[["sens","sens10"]].mean(); P("seeds:",(s2.sens*100).round(2).to_dict(),"SD %.3f pp (5%%), %.3f pp (10%%)"%(100*s2.sens.std(ddof=1),100*s2.sens10.std(ddof=1)))
# calibration
for t,z in CA.groupby("task"):
    r=z.groupby("fold").apply(lambda q:(q.p*q.n).sum()/(q.y*q.n).sum()); P(f"calibration {t}: pooled predicted/observed {(z.p*z.n).sum()/(z.y*z.n).sum():.3f}; by farm-year {r.min():.2f} to {r.max():.2f}")
# importance
for t,z in IM.groupby("task"): P(f"importance (gain share, mean over farm-years) {t}:",z.drop(columns=["task","fold"]).mean().sort_values(ascending=False).head(10).round(3).to_dict())
# sensitivity analyses
SS=S.groupby("analysis",sort=False).apply(lambda q:pd.Series(dict(farm_years=int(q[["rule","model"]].notna().all(axis=1).sum()),event_rate=q.rate.mean(),rule=q.rule.mean(),model=q.model.mean(),model_wins=int((q.model>q.rule).sum())))).reset_index(); SS.to_csv(V+"sens_summary.csv",index=False); P("\nsensitivity analyses:"); P(SS.round(4).to_string())
# folds
P("\nfold composition:"); P(FO.sort_values(["year","farm"]).to_string())
# ---------------- coverage relative to all new records of cows at 10-400 DIM
fl=pd.read_parquet(V+"disease_flow_records.parquet",columns=["cid","dnum","status","milked","risk","farm","year","grp"]); m=cases.merge(fl.rename(columns={"dnum":"rec_dnum"})[["cid","rec_dnum","status","milked"]],on=["cid","rec_dnum"],how="left"); P("\ncases matched to the flow table: %.4f ; status of cases: %s"%(m.status.notna().mean(),m.status.value_counts(dropna=False).to_dict()))
FY=pd.read_csv(V+"tab_disease_flow_farm_year.csv"); FY["fold"]=FY.farm+" "+FY.year.astype(str); det=m[m.status=="10 to 400 days in milk"].groupby("fold").agg(det_rule5=("rule5","sum"),det_model5=("model5","sum"),det_rule10=("rule10","sum"),det_model10=("model10","sum"),ncase=("rule5","size")).reset_index(); FY=FY.merge(det,on="fold",how="left")
FY["overall_rule5"]=FY.det_rule5/FY.target; FY["overall_model5"]=FY.det_model5/FY.target; FY["milked_rule5"]=FY.det_rule5/FY.target_milked; FY["overall_rule10"]=FY.det_rule10/FY.target; FY["milked_rule10"]=FY.det_rule10/FY.target_milked; FY.to_csv(V+"tab_overall_coverage.csv",index=False); P(FY.round(3).to_string())
P("all new records of cows at 10-400 DIM in the test farm-years: %d ; with a recorded milking in the three preceding days: %d (%.3f) ; cases: %d (%.3f) ; detected by the rule at 5%%: %.4f of all, %.4f of those with a recorded milking ; model %.4f of all ; rule at 10%%: %.4f of all, %.4f of milked | farm-year means: rule5/all %.4f rule5/milked %.4f"%(FY.target.sum(),FY.target_milked.sum(),FY.target_milked.sum()/FY.target.sum(),FY.ncase.sum(),FY.ncase.sum()/FY.target.sum(),FY.det_rule5.sum()/FY.target.sum(),FY.det_rule5.sum()/FY.target_milked.sum(),FY.det_model5.sum()/FY.target.sum(),FY.det_rule10.sum()/FY.target.sum(),FY.det_rule10.sum()/FY.target_milked.sum(),FY.overall_rule5.mean(),FY.milked_rule5.mean()))
fa=pd.read_parquet(V+"disease_flow_records.parquet",columns=["farm","status","risk","milked","nD"]); PFm=fa.groupby("farm").apply(lambda q:pd.Series(dict(new=len(q),target=(q.status=="10 to 400 days in milk").sum(),target_milked=((q.status=="10 to 400 days in milk")&q.milked).sum(),risk=q.risk.sum(),target_risk=((q.status=="10 to 400 days in milk")&q.risk).sum()))).reset_index(); PFm.to_csv(V+"tab_disease_flow_by_farm.csv",index=False); P(PFm.to_string())
# ---------------- persistence of the lists and number of days on the list per detected case (Task 2, 5% lists)
def lists(day,score,k):
    o=np.lexsort((-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); n=len(day); cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); s=np.zeros(n,bool); s[o]=rank<np.repeat(np.ceil(cnt*k).clip(min=1),cnt); return s
PR=[]
for f in sorted(glob.glob(C+"pred2_*.parquet")):
    x=pd.read_parquet(f).sort_values(["cid","dnum"]).reset_index(drop=True); day=x.dnum.to_numpy(); r=dict(file=f.split("/")[-1])
    for nm in ("rule","model"):
        S=lists(day,x[nm].to_numpy(),0.05); x["S"]=S; prev=(x.cid.shift(1)==x.cid)&(x.dnum.shift(1)==x.dnum-1); Sp=x.S.shift(1).fillna(False).astype(bool)&prev
        r[nm+"_repeat"]=(S&Sp).sum()/max((S&prev).sum(),1); pos=x[x.y]; cs=(pos.cid.astype(np.int64)*100000+(pos.dnum+pos.dn_dis).astype(np.int64)); g=pos.S.groupby(cs.values).agg(["sum","size"]); d=g[g["sum"]>0]
        r[nm+"_days_on_list"]=d["sum"].mean(); r[nm+"_multi"]=(d["sum"]>=2).mean(); r[nm+"_det"]=len(d)/len(g)
    PR.append(r)
PR=pd.DataFrame(PR); PR.to_csv(V+"ew_list_persistence.csv",index=False); P("\nlist persistence (5% lists, Task 2), mean over farm-years:"); P(PR.drop(columns=["file"]).mean().round(4).to_string())
P("DONE")
