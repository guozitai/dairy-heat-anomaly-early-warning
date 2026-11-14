"""Isolated target-day masking with observed yields unchanged; not a clinical validation.
Deterministic sample: up to 120 cows/farm; target dates spaced 21 d apart.
Past references fixed because prior perturbations lie outside t-8..t-2.
Recompute intervals after deletion; no imputation of removed yields.
"""
from pathlib import Path
import pandas as pd,numpy as np,json,time
V=Path('/home/guozitai/20261111_sat_rescue/v2');OUT=Path('/home/guozitai/20261114_be_submission_audit');OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(20261114);farms=[3,5,6,13,14,15,16,17];lab={6:1,17:2,16:3,13:4,14:5,15:6,5:7,3:8}
ct=pd.read_parquet(V/'cowtab.parquet');ids=[]
for f in farms:
 x=np.sort(ct.loc[ct.farm_id==f,'cid'].unique());ids.extend(rng.choice(x,min(120,len(x)),replace=False).tolist())
s=pd.read_parquet(V/'sess_v3.parquet',columns=['cid','dnum','shift','t','y','dt','valid','ref','ref_y'],filters=[('cid','in',ids)])
s=s.sort_values(['cid','t']).reset_index(drop=True);print('sample sessions',len(s),flush=True)
d=pd.read_parquet(V/'cowday_v2.parquet',columns=['cid','dnum','farm_id','DIM','par','D','dev_raw','base_raw'],filters=[('cid','in',ids)])
d=d[d.farm_id.isin(farms)&d.DIM.between(10,400)&d.par.notna()&d.D.notna()&d.dev_raw.notna()].copy()
d=d[(d.dnum>=731)&((d.dnum%21)==(d.farm_id%21))]
d=d[~((d.farm_id==16)&(d.dnum>=(pd.Timestamp('2025-06-07')-pd.Timestamp('2022-01-01')).days))]
g=s.groupby(['cid','dnum']).agg(n=('y','size'),nshift=('shift','nunique'),good=('valid','all'),nr=('ref','count'),ny=('ref_y','count'))
g=g[(g.n==3)&(g.nshift==3)&g.good&(g.nr==3)&(g.ny==3)].reset_index();d=d.merge(g[['cid','dnum']],on=['cid','dnum']).sort_values(['cid','dnum']).reset_index(drop=True)
s=s.merge(d[['cid','dnum']],on=['cid','dnum']).sort_values(['cid','dnum','t']).reset_index(drop=True)
s=s.merge(d.reset_index(names='target')[['target','cid','dnum','farm_id']],on=['cid','dnum']).sort_values(['target','t']).reset_index(drop=True)
assert len(s)==3*len(d)
y=s.y.to_numpy(dtype=np.float64).reshape(-1,3);dt=s.dt.to_numpy(dtype=np.float64).reshape(-1,3);ref=s.ref.to_numpy(dtype=np.float64).reshape(-1,3);refy=s.ref_y.to_numpy(dtype=np.float64).reshape(-1,3)
full=np.log(y.sum(1)/(ref*dt).sum(1));assert np.max(np.abs(full-d.D.to_numpy()))<2e-6
randdrop=rng.integers(0,3,len(d));randomkeep=rng.integers(0,3,len(d))
# same shift code is removed from every sampled cow on a farm-day.
sh=s['shift'].to_numpy().reshape(-1,3);dropcluster=np.zeros(len(d),int)
for (f,day),ix in d.groupby(['farm_id','dnum']).groups.items():
 ix=np.array(list(ix));choices=np.unique(sh[ix]);pick=choices[int((f*31+day)%len(choices))];hit=(sh[ix]==pick);assert hit.sum(1).min()==1;dropcluster[ix]=hit.argmax(1)
rows=[];individual=[];L=np.log(.8)
for scenario,nmissing,drop in [('random_one',1,randdrop),('whole_shift',1,dropcluster),('random_two',2,randomkeep)]:
 keep=np.ones_like(y,dtype=bool)
 if nmissing==1:keep[np.arange(len(d)),drop]=False
 else:keep[:]=False;keep[np.arange(len(d)),drop]=True
 # Interval preceding first observed milking includes its original prior gap;
 # later observed milkings include intervals of all removed predecessors.
 newdt=np.empty_like(dt)
 carry=np.zeros(len(d))
 for j in range(3):
  carry+=dt[:,j];newdt[:,j]=carry;carry=np.where(keep[:,j],0,carry)
 regular=(newdt>=5)&(newdt<=11.5);long=(newdt>11.5)&(newdt<=34);use=keep&(regular|long)
 expected=np.where(regular,ref*newdt,refy)
 for inject in [False,True]:
  yy=y*(.7 if inject else 1.0)
  measure=np.log(np.where(use,yy,0).sum(1)/np.where(use,expected,0).sum(1));raw=np.log(np.where(keep,yy,0).sum(1)/d.base_raw.to_numpy(dtype=np.float64))
  baseline=full+(np.log(.7) if inject else 0.0)
  rawbase=d.dev_raw.to_numpy(dtype=np.float64)+(np.log(.7) if inject else 0.0)
  targetpositive=baseline<L
  for name,val,base in [('session_aware',measure,baseline),('daily_sum',raw,rawbase)]:
   for farm,idx in [('ALL',np.arange(len(d)))]+[(str(lab[f]),np.flatnonzero(d.farm_id.to_numpy()==f)) for f in farms]:
    v=val[idx];bb=base[idx];pos=bb<L;valid=np.isfinite(v);err=v-bb
    rows.append(dict(scenario=scenario,injected_30pct=inject,measure=name,farm=farm,n=len(idx),coverage=valid.mean(),median_abs_D_error=np.nanmedian(abs(err)),p95_abs_D_error=np.nanquantile(abs(err),.95),agreement=np.mean((v<L)==pos),new_drop_rate=np.mean(v[~pos]<L) if (~pos).any() else None,drop_retention=np.mean(v[pos]<L) if pos.any() else None,positive_n=int(pos.sum())))
  if not inject:
   individual.append(pd.DataFrame(dict(cid=d.cid,dnum=d.dnum,farm=d.farm_id,scenario=scenario,D_full=baseline,D_masked=measure,raw_full=rawbase,raw_masked=raw)))
pd.DataFrame(rows).to_csv(OUT/'masking_summary.csv',index=False)
pd.concat(individual).to_parquet(OUT/'masking_pairs_private.parquet',index=False)
pd.concat(individual).to_csv(OUT/'masking_pairs_private.csv.gz',index=False)
meta=dict(seed=20261114,cows_selected=len(ids),cows_with_targets=d.cid.nunique(),target_days=len(d),sampled_sessions=len(s),farm_counts={str(lab[k]):int(v) for k,v in d.groupby('farm_id').size().items()},max_baseline_reproduction_error=float(np.max(abs(full-d.D.to_numpy()))),notes=['Observed-complete days are reference, not biological truth.','Injection scales all target-day yields by 0.7 and is not a disease model.','One/two records deleted per target day. Next retained interval recalculated.','References fixed in isolated t-8..t-2 history; no persistent multi-day missingness tested.','No original parquet or analysis results changed.'])
(OUT/'masking_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta));print(pd.DataFrame(rows).query("farm=='ALL'").round(4).to_string(index=False))
