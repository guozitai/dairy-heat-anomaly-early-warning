"""Pool cross-year case detections by unique cow/event date; fold comparisons unchanged."""
from pathlib import Path
import pandas as pd,numpy as np,json
V=Path('/home/guozitai/20261111_sat_rescue/v2');O=Path('/home/guozitai/20261114_be_v12')
c=pd.concat([pd.read_parquet(f) for f in sorted((V/'c1').glob('cases_*.parquet'))],ignore_index=True)
assert len(c)==20709
for k in ['cid','rec_dnum','grp','farm']:assert c.groupby('case')[k].nunique().max()==1
cols=['rule5','rule10','model5','model10','blend5','rule5_early','model5_early','has_early','dailysum5','cohort5']
a=c.groupby('case',as_index=False).agg(**{k:(k,'max') for k in cols},cid=('cid','first'),rec_dnum=('rec_dnum','first'),grp=('grp','first'),farm=('farm','first'),risk_days=('risk_days','sum'))
assert a.risk_days.max()<=3
pd.DataFrame([dict(n=len(a),**{k:a[k].mean() for k in cols if not k.endswith('_early') and k!='has_early'},n_early=int(a.has_early.sum()),rule5_early=a.loc[a.has_early,'rule5_early'].mean(),model5_early=a.loc[a.has_early,'model5_early'].mean())]).to_csv(O/'ew_pooled_unique.csv',index=False)
rng=np.random.default_rng(1);rows=[]
for group,x in a.groupby('grp'):
 cw=pd.factorize(x.cid)[0];bs=[]
 for _ in range(1000):
  w=rng.poisson(1,cw.max()+1)[cw];den=max(w.sum(),1);bs.append([(w*x.rule5).sum()/den,(w*x.model5).sum()/den,(w*(x.model5-x.rule5)).sum()/den])
 bs=np.array(bs);lo,hi=np.percentile(bs,[2.5,97.5],axis=0)
 rows.append(dict(grp=group,n=len(x),rule5=x.rule5.mean(),rule5_lo=lo[0],rule5_hi=hi[0],model5=x.model5.mean(),model5_lo=lo[1],model5_hi=hi[1],diff_lo=lo[2],diff_hi=hi[2],rule10=x.rule10.mean(),model10=x.model10.mean(),cohort5=x.cohort5.mean(),dailysum5=x.dailysum5.mean()))
pd.DataFrame(rows).to_csv(O/'ew_by_class_pooled_unique.csv',index=False)
flow=pd.read_parquet(V/'disease_flow_records.parquet');flow['case']=flow.cid.to_numpy(np.int64)*100000+flow.dnum.to_numpy(np.int64);assert not flow.case.duplicated().any();m=flow[flow.year.isin([2024,2025])&(flow.status=='10 to 400 days in milk')].merge(a[['case']+cols],on='case',how='left');assert len(m)==35613
summ={'case_fold_evaluations':len(c),'unique_cases':len(a),'cases_spanning_test_years':int((c.groupby('case').size()>1).sum()),'unique_pooled_rule5':a.rule5.mean(),'unique_pooled_model5':a.model5.mean(),'all_target_records':len(m),'target_records_with_milking':int(m.milked.sum()),'target_unique_evaluable':int(m.rule5.notna().sum()),'target_rule5_detected':int(m.rule5.fillna(0).sum()),'target_rule5_coverage':float(m.rule5.fillna(0).sum()/len(m)),'target_milked_rule5_coverage':float(m.rule5.fillna(0).sum()/m.milked.sum()),'risk_days_unique':a.risk_days.value_counts().sort_index().to_dict(),'fold_results_unchanged':True}
(O/'case_boundary_audit.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ),flush=True)
