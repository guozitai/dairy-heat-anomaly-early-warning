from pathlib import Path
import pandas as pd,numpy as np,json
from sklearn.metrics import average_precision_score
V=Path('/home/guozitai/20261111_sat_rescue/v2');W=Path('/home/guozitai/20261114_sat_review_audit');rows=[];null=[]
T={1:pd.read_csv(V/'ew_onset.csv'),2:pd.read_csv(V/'ew_disease.csv')};lab={6:1,17:2,16:3,13:4,14:5,15:6,5:7,3:8}
for t in [1,2]:
 for f in [3,5,6,13,14,15,16,17]:
  for Y in [2024,2025]:
   d=pd.read_parquet(V/f'c1/pred{t}_{f}_{Y}.parquet');tag=f'Farm {lab[f]} {Y}';obs=d.obs.to_numpy() if t==1 else np.ones(len(d),bool);y=d.y.to_numpy().astype(bool)
   for config,col in [('R_deviation','rule'),('M_ref','model')]:
    expected=T[t].query('fold==@tag and config==@config').iloc[0];prev=(expected.rate if t==1 else expected.pos/expected.n);raw=average_precision_score(y[obs],d[col].to_numpy()[obs]);assert abs(raw-expected.pr_lift*prev)<1e-12
    rows.append(dict(task=t,fold=tag,config=config,average_precision=raw,prevalence=prev,pr_lift=expected.pr_lift))
   if t==2:
    _,day=np.unique(d.dnum,return_inverse=True);cnt=np.bincount(day);pos=np.flatnonzero(y);key=d.cid.to_numpy().astype(np.int64)[pos]*100000+d.dnum.to_numpy().astype(np.int64)[pos]+d.dn_dis.to_numpy().astype(np.int64)[pos];cu,ix=np.unique(key,return_inverse=True)
    for k in [.05,.1]:
     p=np.ceil(cnt*k)/cnt;q=np.ones(len(cu));np.multiply.at(q,ix,1-p[day[pos]]);chance=(1-q).mean();expected=T[2].query('fold==@tag and config=="R_deviation"').iloc[0][f'chance{int(k*100)}'];assert abs(chance-expected)<1e-12;null.append(dict(fold=tag,k=k,chance=chance,cases=len(cu)))
pd.DataFrame(rows).to_csv(W/'average_precision_direct_check.csv',index=False);pd.DataFrame(null).to_csv(W/'chance_exact_check.csv',index=False);print('VERIFIED',len(rows),'direct AP values and',len(null),'multi-day random-list expectations',flush=True)
