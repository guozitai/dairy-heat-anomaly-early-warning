"""Paired Task 2 rule comparisons on full and common-score risk sets. No model fitting."""
from pathlib import Path
import pandas as pd,numpy as np,json
from scipy.stats import t,binomtest
V=Path('/home/guozitai/20261111_sat_rescue/v2');O=Path('/home/guozitai/20261114_be_v12');O.mkdir(exist_ok=True)
LAB={6:'Farm 1',17:'Farm 2',16:'Farm 3',13:'Farm 4',14:'Farm 5',15:'Farm 6',5:'Farm 7',3:'Farm 8'}
def select(day,score,tie,k=.05):
 o=np.lexsort((tie,score,day));st=np.flatnonzero(np.r_[True,np.diff(day[o])!=0]);cnt=np.diff(np.r_[st,len(o)]);yes=np.arange(len(o))-np.repeat(st,cnt)<np.repeat(np.ceil(cnt*k).astype(int),cnt);out=np.zeros(len(o),bool);out[o]=yes;assert out.sum()==np.ceil(cnt*k).sum();return out
def case_info(z):
 pos=np.flatnonzero(z.y_dis.to_numpy()==1);keys=z.cid.to_numpy(np.int64)[pos]*100000+(z.dnum.to_numpy()[pos]+z.dn_dis.to_numpy()[pos]).astype(np.int64);case,ix=np.unique(keys,return_inverse=True);return pos,case,ix
def hits(sel,pos,ix,n):
 h=np.zeros(n,bool);np.logical_or.at(h,ix,sel[pos]);return h
F=pd.read_parquet(V/'feat_v2.parquet',columns=['cid','farm_id','date','dnum','elig','risk_dis','D','dev_my24','y_dis','dn_dis']);F=F[F.elig&F.risk_dis].sort_values(['cid','dnum']).reset_index(drop=True)
rows=[];decomp=[]
old=pd.concat([pd.read_csv(V/'c1'/f'ew2_part{i}.csv') for i in range(6)])
for year in [2024,2025]:
 for farm in LAB:
  z=F[(F.farm_id==farm)&(F.date.dt.year==year)].reset_index(drop=True);tie=np.random.default_rng([20261111,year,farm,2]).random(len(z));ok=np.isfinite(z.dev_my24.to_numpy());pos,case,ix=case_info(z);sd=select(z.dnum.to_numpy(),z.D.to_numpy(float),tie);v=z.dev_my24.to_numpy(float);v=np.where(ok,v,np.nanmedian(v));sc=select(z.dnum.to_numpy(),v,tie)
  hd=hits(sd,pos,ix,len(case));hc=hits(sc,pos,ix,len(case));anycommon=hits(ok,pos,ix,len(case));anyonly=hits(~ok,pos,ix,len(case));hitcommon=hits(sd&ok,pos,ix,len(case));hitonly=hits(sd&~ok,pos,ix,len(case))
  for nm,h in [('R_deviation',hd),('R_my24_deviation',hc)]:
   orig=old[(old.farm==LAB[farm])&(old.year==year)&(old.config==nm)].iloc[0];assert len(z)==orig['n'] and len(case)==orig.cases;assert abs(h.mean()-orig.case_sens5)<1e-12,(farm,year,nm,h.mean(),orig.case_sens5)
  decomp.append(dict(farm=LAB[farm],year=year,cowdays=len(z),common_cowdays=int(ok.sum()),D_only_cowdays=int((~ok).sum()),cases=len(case),cases_any_common=int(anycommon.sum()),cases_only_D_available=int((~anycommon).sum()),cases_mixed=int((anycommon&anyonly).sum()),D_detected=int(hd.sum()),D_hits_on_common_only=int((hitcommon&~hitonly).sum()),D_hits_on_D_only_only=int((hitonly&~hitcommon).sum()),D_hits_on_both=int((hitcommon&hitonly).sum()),comparator_detected=int(hc.sum())))
  for scope,zz,tt in [('full',z,tie),('common',z.loc[ok].reset_index(drop=True),tie[ok])]:
   pp,cc,ii=case_info(zz)
   for score,col in [('session_aware','D'),('24h_equivalent','dev_my24')]:
    vals=zz[col].to_numpy(float);vals=np.where(np.isfinite(vals),vals,np.nanmedian(vals));sel=select(zz.dnum.to_numpy(),vals,tt);hh=hits(sel,pp,ii,len(cc))
    rows.append(dict(farm=LAB[farm],year=year,scope=scope,score=score,cowdays=len(zz),cases=len(cc),positive_cowdays=len(pp),listed=int(sel.sum()),detected=int(hh.sum()),recall=hh.mean(),precision=float(zz.y_dis.to_numpy()[sel].mean()),entries_per_case=sel.sum()/max(hh.sum(),1)))
  print(LAB[farm],year,'full verified; common',round(ok.mean(),4),flush=True)
a=pd.DataFrame(rows);a.to_csv(O/'common_support_by_fold.csv',index=False);pd.DataFrame(decomp).to_csv(O/'coverage_decomposition.csv',index=False)
sumrows=[]
for scope in ['full','common']:
 z=a[a.scope==scope];p=z.pivot(index=['farm','year'],columns='score',values='recall');delta=p.session_aware-p['24h_equivalent'];fm=delta.groupby(level=0).mean();ci=t.interval(.95,7,loc=fm.mean(),scale=fm.std(ddof=1)/np.sqrt(8));sr=dict(scope=scope,mean_session_aware=float(p.session_aware.mean()),mean_24h=float(p['24h_equivalent'].mean()),mean_difference=float(delta.mean()),farm_t95_low=ci[0],farm_t95_high=ci[1],positive_farmyears=int((delta>0).sum()),positive_farms=int((fm>0).sum()),sign_test_farms_p=binomtest((fm>0).sum(),8,.5).pvalue)
 for sc in ['session_aware','24h_equivalent']:
  q=z[z.score==sc];sr[sc+'_pooled']=q.detected.sum()/q.cases.sum();sr[sc+'_entries_mean']=q.entries_per_case.mean()
 sumrows.append(sr)
pd.DataFrame(sumrows).to_csv(O/'common_support_summary.csv',index=False);print(pd.DataFrame(sumrows).to_string(index=False),flush=True)
(O/'common_support_checks.json').write_text(json.dumps({'original_fold_estimates_reproduced':32,'daily_list_capacity_checked':True,'risk_set_rebuilt_for_common_comparison':True,'note':'D-only detection contributions are descriptive, not an additive causal decomposition of recall differences.'},indent=2))
