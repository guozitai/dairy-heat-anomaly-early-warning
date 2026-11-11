#!/usr/bin/env python3
"""C5. A cohort-referenced label on daily sums, of the kind commonly derived from daily records, computed on all eligible cow-days:
ratio of the daily sum to the median daily sum of cows of the same farm, day, parity class and 30-day DIM class; flagged when ln(ratio) is more than
1.5 SD below the mean of the farm and parity class. Composition of the flags; persistence (previous-day ratio as a predictor of next-day flags).
Second part (for the response letter): the label and scores of the previously submitted pipeline on its own test folds."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",260); L=np.log(0.8); LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
def lists(day,score,k):
    o=np.lexsort((-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); n=len(day); cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); s=np.zeros(n,bool); s[o]=rank<np.repeat(np.ceil(cnt*k).clip(min=1),cnt); return s
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","y_sum","n_sess","D","par","DIM","elig"]); F=F[F.elig].copy(); F["farm"]=F.farm_id.map(LAB)
F["pc"]=F.par.clip(1,3).astype("int8"); F["dc"]=(F.DIM//30).astype("int16"); g=F.groupby(["farm_id","dnum","pc","dc"]).y_sum; F["med"]=g.transform("median"); F["gn"]=g.transform("size")
F["z"]=np.log(F.y_sum/F.med).where(F.gn>=5); gz=F.groupby(["farm_id","pc"]).z; F["flag"]=(F.z<gz.transform("mean")-1.5*gz.transform("std")).where(F.z.notna())
md=F.groupby(["farm_id","dnum"]).n_sess.transform(lambda x:x.mode().iloc[0]); F["fewer"]=F.n_sess<md
x=F[F.flag.notna()].copy(); x["flag"]=x.flag.astype(bool)
cs=x.groupby("cid").flag.agg(["mean","sum","size"]); ch=cs[(cs["size"]>=60)&(cs["mean"]>0.2)]["sum"].sum()/cs[cs["size"]>=60]["sum"].sum()
xd=x[x.D.notna()]
P("== cohort-referenced daily-sum label, all eligible cow-days with a cohort of at least five cows: n=%d =="%len(x))
P("prevalence %.4f ; share of flags on cow-days with fewer recorded milkings than the farm mode %.4f ; flag rate on such cow-days %.4f vs %.4f on others ; share of cow-days with fewer milkings %.4f"%(x.flag.mean(),x[x.flag].fewer.mean(),x[x.fewer].flag.mean(),x[~x.fewer].flag.mean(),x.fewer.mean()))
P("share of flags from cows flagged on >20%% of their days (cows with >=60 days): %.4f ; of flags with a day deviation, share that are acute drops %.4f ; of acute drops, share flagged %.4f"%(ch,(xd.D<L)[xd.flag].mean(),xd.flag[xd.D<L].mean()))
o=x.groupby("farm").apply(lambda q:pd.Series(dict(n=len(q),prev=q.flag.mean(),flags_on_fewer=q[q.flag].fewer.mean(),rate_fewer=q[q.fewer].flag.mean(),rate_other=q[~q.fewer].flag.mean(),fewer_share=q.fewer.mean()))).reset_index(); P(o.round(4).to_string()); o.to_csv(V+"tab_cohort_label_by_farm.csv",index=False)
# persistence: previous-day ratio as predictor of next-day flag
x=x.sort_values(["cid","dnum"]); nx=x.groupby("cid").dnum.shift(-1); x["flag_next"]=x.groupby("cid").flag.shift(-1).where(nx==x.dnum+1)
y=x.dropna(subset=["flag_next"]); yv=y.flag_next.astype(bool).to_numpy(); S=lists(y.dnum.to_numpy()*100+y.farm_id.to_numpy(),-y.z.to_numpy(),0.10)
P("next-day flags: n=%d rate %.4f ; share on 10%% lists ranked by the ratio of the current day %.4f (lift %.2f) ; P(flag next | flag today) %.4f vs %.4f"%(len(y),yv.mean(),S[yv].mean(),S[yv].mean()*10,yv[y.flag.to_numpy()].mean(),yv[~y.flag.to_numpy()].mean()))
pd.Series(dict(n=len(x),prevalence=x.flag.mean(),flags_on_fewer=x[x.flag].fewer.mean(),rate_fewer=x[x.fewer].flag.mean(),rate_other=x[~x.fewer].flag.mean(),fewer_share=x.fewer.mean(),chronic_share=ch,flags_that_are_drops=(xd.D<L)[xd.flag].mean(),drops_flagged=xd.flag[xd.D<L].mean(),nextday_recall10=S[yv].mean(),p_next_given_flag=yv[y.flag.to_numpy()].mean(),p_next_given_noflag=yv[~y.flag.to_numpy()].mean())).to_csv(V+"tab_cohort_label.csv")
# ---- previously submitted pipeline (response letter)
p=pd.read_parquet(D0+"v2_predictions.parquet"); p["key"]=p.milkDate.astype(str)+"_"+p.farm_id.astype(str); day=pd.factorize(p.key)[0]; ev=p.event.to_numpy()==1; rows=[]
P("\n== previously submitted label on its test folds: n=%d prevalence %.4f =="%(len(p),ev.mean()))
for nm,col,sign in (("previous-day ratio","ratio_lag1",-1),("persistence score","pers",1),("submitted model","mrisk_enriched",1),("submitted model without climate","mrisk_enriched_nocli",1)):
    if col not in p: continue
    v=p[col].to_numpy(float); ok=np.isfinite(v); v=np.where(ok,v,np.nanmedian(v)); S=lists(day,sign*v,0.10); r=dict(score=nm,lift10=S[ev].mean()*10,available=ok.mean())
    for f,ix in p.groupby("farm_label").indices.items(): r[f]=S[ix][ev[ix]].mean()*10
    rows.append(r); P("   %-34s lift at 10%% %.2f (score available %.3f)"%(nm,r["lift10"],ok.mean()))
R=pd.DataFrame(rows); R["mean_of_farms"]=R[[c for c in R.columns if c.startswith("Farm")]].mean(axis=1); P(R.round(2).to_string()); R.to_csv(V+"tab_submitted_label_scores.csv",index=False)
P("pers column summary:",p.pers.describe().round(3).to_dict() if "pers" in p else None)
P("DONE",round(time.time()-t0))
