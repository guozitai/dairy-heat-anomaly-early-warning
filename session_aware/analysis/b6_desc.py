#!/usr/bin/env python3
"""B6. Descriptive table by farm and episode statistics (consecutive drop days)."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}; L=np.log(0.8)
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","D","DIM","par","elig","my24","n_sess","risk_onset","y_onset","y_onset_sus","risk_dis","y_dis","dn_dis","tmax","dev_raw"])
F=F[F.elig].sort_values(["cid","dnum"]).reset_index(drop=True); F["farm"]=F.farm_id.map(LAB)
c=F.cid.to_numpy(); dn=F.dnum.to_numpy(); dr=(F.D<L).to_numpy()
prev_same=np.r_[False,(c[1:]==c[:-1])&(dn[1:]-dn[:-1]==1)]; prev_drop=np.r_[False,dr[:-1]]&prev_same
start=dr&~prev_drop; F["ep"]=np.where(dr,np.cumsum(start),0)
ep=F[F.ep>0].groupby("ep").agg(farm=("farm","first"),dur=("D","size"),depth=("D","min"))
E=ep.groupby("farm").agg(episodes=("dur","size"),median_days=("dur","median"),mean_days=("dur","mean"),share_1d=("dur",lambda x:(x==1).mean()),share_ge3=("dur",lambda x:(x>=3).mean()),median_depth=("depth",lambda x:1-np.exp(x.median()))).reset_index()
dis=pd.read_parquet(V+"disease_cid.parquet"); ct=F[["cid","farm"]].drop_duplicates(); dis=dis.merge(ct,on="cid")
dd=F.groupby("farm").agg(first=("date","min"),last=("date","max"),cows=("cid","nunique"),cow_days=("D","size"),with_D=("D","count"),median_my24=("my24","median"),drop_days=("D",lambda x:(x.dropna()<L).mean()),at_risk=("risk_onset","sum"),onsets=("y_onset","sum"),sustained=("y_onset_sus","sum"),dis_risk=("risk_dis","sum"),thi_days=("tmax",lambda x:x.notna().any()),s3=("n_sess",lambda x:(x>=3).mean()),s2=("n_sess",lambda x:(x==2).mean()),s1=("n_sess",lambda x:(x==1).mean()),flag_raw=("dev_raw",lambda x:(x.dropna()<L).mean())).reset_index()
# evaluable cases: distinct (cow, record day) with >=1 risk-set day in the 3 preceding days
pos=F[(F.y_dis==1)].assign(case=lambda x:x.cid.astype(np.int64)*100000+(x.dnum+x.dn_dis).astype(np.int64)); cases=pos.groupby("farm").case.nunique().rename("evaluable_cases").reset_index()
rec=dis.groupby("farm").size().rename("disease_records").reset_index()
T=dd.merge(E,on="farm").merge(rec,on="farm",how="left").merge(cases,on="farm",how="left"); T["onset_per_1000"]=1000*T.onsets/T.at_risk
P(T.round(3).to_string()); T.to_csv(V+"tab_farm_descriptives.csv",index=False)
tot=dict(s3=(F.n_sess>=3).mean(),s2=(F.n_sess==2).mean(),s1=(F.n_sess==1).mean(),flag_raw=(F.dev_raw.dropna()<L).mean(),flag_D=(F.D.dropna()<L).mean(),first=str(F.date.min().date()),last=str(F.date.max().date()),disease_records=len(dis),cows=F.cid.nunique(),cow_days=len(F),with_D=int(F.D.notna().sum()),episodes=len(ep),median_days=ep.dur.median(),mean_days=ep.dur.mean(),share_1d=(ep.dur==1).mean(),share_ge3=(ep.dur>=3).mean(),cases=pos.case.nunique(),onsets=int(F.y_onset.sum()),at_risk=int(F.risk_onset.sum()))
P("TOTAL",tot); pd.Series(tot).to_csv(V+"tab_totals.csv")
P("episode duration distribution (days: share):",ep.dur.clip(upper=8).value_counts(normalize=True).sort_index().round(3).to_dict())
P("DONE",round(time.time()-t0))
