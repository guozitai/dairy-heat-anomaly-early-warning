"""Audit disease timestamps and replay D-rule eligibility using record creation dates.
End-of-calendar-day sensitivity, not a complete historical database snapshot or prospective trial.
Only aggregate outputs; cow identifiers and clinical free text remain on the data server.
"""
from pathlib import Path
import pandas as pd,numpy as np,json
V=Path('/home/guozitai/20261111_sat_rescue/v2');O=Path('/home/guozitai/20261114_be_v12');O.mkdir(exist_ok=True)
LAB={6:'Farm 1',17:'Farm 2',16:'Farm 3',13:'Farm 4',14:'Farm 5',15:'Farm 6',5:'Farm 7',3:'Farm 8'};BASE=pd.Timestamp('2022-01-01')
ct=pd.read_parquet(V/'cowtab.parquet')[['cid','farm_id','cow']];ct=ct[ct.farm_id.isin(LAB)]
e=pd.read_parquet('/home/guozitai/20261014_event_layer/parquet/e_disease.parquet');columns=e.columns.tolist();e['farm_id']=pd.to_numeric(e.farm_id,errors='coerce');e['cow']=e.cow_number.astype(str);e=e.merge(ct,on=['farm_id','cow']);e['event']=pd.to_datetime(e.event_time,errors='coerce');e['created']=pd.to_datetime(e.create_time,errors='coerce');e['updated']=pd.to_datetime(e.update_time,errors='coerce');e=e[e.event.notna()].copy();e['day']=(e.event.dt.normalize()-BASE).dt.days;e['created_day']=(e.created.dt.normalize()-BASE).dt.days
# One available disease record is sufficient for exclusion on a cow-event day.
ev=e.groupby(['cid','day'],as_index=False).agg(created_day=('created_day','min')).sort_values(['cid','day']);gap=ev.groupby('cid').day.diff();new=ev[gap.isna()|(gap>14)];newkeys=new.cid.to_numpy(np.int64)*100000+new.day.to_numpy(np.int64)
F=pd.read_parquet(V/'feat_v2.parquet',columns=['cid','farm_id','dnum','date','elig','D','risk_dis','y_dis','dn_dis']);F=F[F.elig&F.D.notna()].sort_values(['cid','dnum']).reset_index(drop=True);keys=F.cid.to_numpy(np.int64)*100000+F.dnum.to_numpy(np.int64)
def blocked(use_creation):
 a=ev.copy();start=a.day.to_numpy(np.int64);end=start+14
 if use_creation:
  valid=a.created_day.notna().to_numpy();a=a.loc[valid];start=np.maximum(start[valid],a.created_day.to_numpy(np.int64));end=end[valid]
 cow=a.cid.to_numpy(np.int64);ok=start<=end;lo=np.searchsorted(keys,cow[ok]*100000+start[ok]);hi=np.searchsorted(keys,cow[ok]*100000+end[ok]+1);diff=np.zeros(len(F)+1,np.int32);np.add.at(diff,lo,1);np.add.at(diff,hi,-1);return np.cumsum(diff[:-1])>0
trueblock=blocked(False);assert np.array_equal(~trueblock,F.risk_dis.to_numpy()),'Event-date reconstruction mismatch'
knownblock=blocked(True);assert not np.any(knownblock&~trueblock)
nextkey=keys+np.nan_to_num(F.dn_dis.to_numpy(),nan=99999).astype(np.int64);outcome=(F.dn_dis.to_numpy()<=3)&np.isin(nextkey,newkeys);assert np.array_equal(outcome&~trueblock,F.y_dis.to_numpy().astype(bool))
F['time_risk']=~knownblock;F['new_outcome']=outcome;F['casekey']=nextkey
flow=pd.read_parquet(V/'disease_flow_records.parquet');q=flow[flow.risk&flow.year.isin([2024,2025])];lag=pd.to_numeric(q.entry_lag,errors='coerce')
stats={'event_columns_available':columns,'case_time_fields_absent':['first_observed_symptom_time','first_clinical_examination_time','alert_trigger_source'],'evaluated_flow_records_2024_2025':len(q),'missing_creation_lag':int(lag.isna().sum()),'negative_entry_lag':int((lag<0).sum()),'same_calendar_day':int((lag==0).sum()),'within_0_to_1_days':int(lag.between(0,1).sum()),'within_0_to_3_days':int(lag.between(0,3).sum()),'same_day_fraction':float((lag==0).mean()),'within1_fraction':float(lag.between(0,1).mean()),'event_date_risk_exactly_reproduced':True,'event_date_labels_exactly_reproduced':True,'end_of_day_definition':'Record event date at most t and creation timestamp before midnight ending local calendar day t; excludes disease in event days t-14..t.','limitation':'Uses current exported rows and their creation timestamps; historical edits/deletions and symptom/diagnosis onset are not reconstructed. Milk-upload latency not available; no clinical lead time claimed.'}
old=pd.concat([pd.read_csv(V/'c1'/f'ew2_part{i}.csv') for i in range(6)])
def select(z,tie):
 day=z.dnum.to_numpy();o=np.lexsort((tie,z.D.to_numpy(float),day));st=np.flatnonzero(np.r_[True,np.diff(day[o])!=0]);cnt=np.diff(np.r_[st,len(o)]);s=np.zeros(len(z),bool);s[o]=np.arange(len(o))-np.repeat(st,cnt)<np.repeat(np.ceil(cnt*.05),cnt);return s
def det(z,s):
 pos=z.new_outcome.to_numpy();case,ix=np.unique(z.casekey.to_numpy()[pos],return_inverse=True);h=np.zeros(len(case),bool);np.logical_or.at(h,ix,s[pos]);return pd.Series(h,index=case)
rows=[];bylag=[]
for year in [2024,2025]:
 for farm in LAB:
  z=F[(F.farm_id==farm)&(F.date.dt.year==year)].reset_index(drop=True);orig=z.risk_dis.to_numpy();tim=z.time_risk.to_numpy();tie=np.random.default_rng([20261114,year,farm,22]).random(len(z));tie[orig]=np.random.default_rng([20261111,year,farm,2]).random(orig.sum());h0=None;sel0=None
  for scope,mk in [('event_date',orig),('creation_time',tim)]:
   zz=z[mk].reset_index(drop=True);sel=select(zz,tie[mk]);h=det(zz,sel)
   if scope=='event_date':
    h0=h;sel0=np.zeros(len(z),bool);sel0[mk]=sel
    r=old[(old.farm==LAB[farm])&(old.year==year)&(old.config=='R_deviation')].iloc[0];assert len(h)==r.cases and abs(h.mean()-r.case_sens5)<1e-12
   fullsel=np.zeros(len(z),bool);fullsel[mk]=sel
   rows.append(dict(farm=LAB[farm],year=year,scope=scope,cowdays=len(zz),added_cowdays=int((tim&~orig).sum()),listed=int(sel.sum()),cases=len(h),detected=int(h.sum()),recall=h.mean(),precision=zz.new_outcome.to_numpy()[sel].mean(),entries_per_case=sel.sum()/max(h.sum(),1),detection_on_original_cases=h.reindex(h0.index,fill_value=False).mean(),retained_original_list_entries=int((fullsel&sel0).sum()),original_list_entries=int(sel0.sum())))
  print('timestamp replay',LAB[farm],year,flush=True)
a=pd.DataFrame(rows);a.to_csv(O/'record_time_by_fold.csv',index=False)
for scope,g in a.groupby('scope'):
 stats[scope]={'cowdays':int(g.cowdays.sum()),'cases':int(g.cases.sum()),'detected':int(g.detected.sum()),'macro_recall':float(g.recall.mean()),'pooled_recall':float(g.detected.sum()/g.cases.sum()),'mean_entries_per_case':float(g.entries_per_case.mean()),'macro_detection_original_cases':float(g.detection_on_original_cases.mean())}
stats['added_risk_cowdays']=int(a[a.scope=='creation_time'].added_cowdays.sum());stats['original_list_retention']=float(a[a.scope=='creation_time'].retained_original_list_entries.sum()/a[a.scope=='creation_time'].original_list_entries.sum())
# Direct timestamps for the actual 20,709 evaluation cases, not the separate flow cohort.
actual=F[F.risk_dis & F.new_outcome & F.date.dt.year.isin([2024,2025])].casekey.unique()
meta=ev.assign(casekey=lambda z:z.cid.to_numpy(np.int64)*100000+z.day.to_numpy(np.int64)).set_index('casekey').reindex(actual)
assert meta.created_day.notna().all()
lag=meta.created_day-meta.day
stats['actual_evaluation_case_timestamps']={'n':len(meta),'same_day':int((lag==0).sum()),'within1':int(lag.between(0,1).sum()),'within3':int(lag.between(0,3).sum()),'negative':int((lag<0).sum()),'missing':int(lag.isna().sum())}
# Independent direct-history check of 300 sampled cow-days for creation-date eligibility.
groups={k:g for k,g in ev.groupby('cid')}
for j in np.random.default_rng(12).choice(len(F),300,replace=False):
 row=F.iloc[j];hist=groups.get(row.cid)
 direct=False if hist is None else bool(((hist.day>=row.dnum-14)&(hist.day<=row.dnum)&(hist.created_day<=row.dnum)).any())
 assert direct==bool(knownblock[j])
stats['independent_creation_time_checks']=300
# Metadata counts only, no human names or free text in outputs.
stats['linked_record_update_after_creation_count']=int((e.updated>e.created).sum());stats['linked_records_count']=len(e);stats['missing_creation_rows']=int(e.created.isna().sum());stats['missing_creation_cow_event_days']=int(ev.created_day.isna().sum())
(O/'record_time_audit.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False),flush=True)
