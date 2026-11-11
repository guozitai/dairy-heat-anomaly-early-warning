#!/usr/bin/env python3
"""D5. Sensitivity of the main results to the cow-days of Farm 3 from 7 June 2025, when part of the herd was milked four times a day.
(a) recorded milkings per cow-day and flags of the two measures, all eligible cow-days and without these cow-days; the cows concerned separately;
(b) acute drops before disease records, cases and controls as in C4, without the records of Farm 3 from that date."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
L8=np.log(0.8); D0=(pd.Timestamp("2025-06-07")-pd.Timestamp("2022-01-01")).days; rs=lambda z: float((z-z.median()).abs().median()*1.4826)
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","shift"]); first4=s[s["shift"]==4].groupby("cid").dnum.min().rename("d4"); del s
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","elig","n_sess","D","dev_raw"]); F=F[F.elig].join(first4,on="cid")
f3=(F.farm_id==16)&(F.dnum>=D0); four=f3&F.d4.notna()&(F.dnum>=F.d4)
def desc(x,lab):
    P("%-46s cow-days %9d | three or more recorded %.4f, two %.4f, one %.4f | D available %.4f | D<ln0.8 %.4f | daily-sum deviation<ln0.8 %.4f | robust SD of D %.4f"%(lab,len(x),(x.n_sess>=3).mean(),(x.n_sess==2).mean(),(x.n_sess==1).mean(),x.D.notna().mean(),(x.D.dropna()<L8).mean(),(x.dev_raw.dropna()<L8).mean(),rs(x.D.dropna())))
desc(F,"all eligible cow-days"); desc(F[~f3],"without Farm 3 from 7 June 2025"); desc(F[f3],"Farm 3 from 7 June 2025"); desc(F[four],"  cows with four daily sessions"); desc(F[f3&~four],"  other cows of Farm 3")
x=F[four]; P("cows with four daily sessions: share of all eligible cow-days %.4f ; recorded milkings per cow-day: %s"%(len(x)/len(F),x.n_sess.clip(upper=5).value_counts(normalize=True).sort_index().round(4).to_dict()))
comp=((F.n_sess>=3)&~four)|((F.n_sess>=4)&four); P("share of cow-days with a complete set of milkings when four are required for these cows: %.4f (three or more for all: %.4f)"%(comp.mean(),(F.n_sess>=3).mean()))
pd.DataFrame([dict(set="all",n=len(F),three=(F.n_sess>=3).mean(),dropD=(F.D.dropna()<L8).mean(),dropraw=(F.dev_raw.dropna()<L8).mean()),dict(set="without_farm3_from_june2025",n=int((~f3).sum()),three=(F[~f3].n_sess>=3).mean(),dropD=(F[~f3].D.dropna()<L8).mean(),dropraw=(F[~f3].dev_raw.dropna()<L8).mean()),dict(set="four_session_cows",n=int(four.sum()),three=(F[four].n_sess>=3).mean(),dropD=(F[four].D.dropna()<L8).mean(),dropraw=(F[four].dev_raw.dropna()<L8).mean()),dict(set="complete_if_four_required",n=len(F),three=comp.mean())]).to_csv(V+"tab_farm3_sensitivity.csv",index=False)
del F
# ---- (b) validation without the disease records of Farm 3 from 7 June 2025 (source of C4, all diseases only)
src=open(V+"c4_validity.py").read()
a='P("new disease records with the record day at least 10 days after the start and 3 days before the end of the milking records:",len(new))'; assert src.count(a)==1
src=src.replace(a,a+'\nn_f3=int(((new.farm_id==16)&(new.dnum>=%d)).sum()); new=new[~((new.farm_id==16)&(new.dnum>=%d))]; P("without the records of Farm 3 from 7 June 2025 (%%d records removed):"%%n_f3,len(new))'%(D0,D0))
a='res.append(study(new,"All diseases",traj,None,True))'; i=src.index(a)+len(a); src=src[:i]+'\npd.DataFrame(res).to_csv(V+"tab2_validity_without_farm3_2025.csv",index=False); P("DONE",round(time.time()-t0))\n'
exec(compile(src,"c4_patched","exec"))
