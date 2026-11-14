"""Repeated deletion with complete recomputation of intervals and historical references.
Same cows/target dates as the isolated-deletion check; existing observations are the reference,
not independently measured biological truth. Private paired outputs stay on the data server.
"""
from pathlib import Path
import pandas as pd,numpy as np,json,warnings
warnings.filterwarnings('ignore',category=RuntimeWarning)
V=Path('/home/guozitai/20261111_sat_rescue/v2');O=Path('/home/guozitai/20261114_be_v12');O.mkdir(exist_ok=True)
LAB={6:1,17:2,16:3,13:4,14:5,15:6,5:7,3:8};farms=[3,5,6,13,14,15,16,17];rng=np.random.default_rng(20261114);ct=pd.read_parquet(V/'cowtab.parquet');ids=[]
for f in farms:
 x=np.sort(ct.loc[ct.farm_id==f,'cid'].unique());ids.extend(rng.choice(x,min(120,len(x)),replace=False).tolist())
s=pd.read_parquet(V/'sess_v3.parquet',columns=['cid','dnum','shift','t','y','dt','valid','ref','ref_y'],filters=[('cid','in',ids)]).sort_values(['cid','t']).reset_index(drop=True);s=s.merge(ct[['cid','farm_id']],on='cid',how='left')
d=pd.read_parquet(V/'cowday_v2.parquet',columns=['cid','dnum','farm_id','DIM','par','D','dev_raw','base_raw'],filters=[('cid','in',ids)])
d=d[d.farm_id.isin(farms)&d.DIM.between(10,400)&d.par.notna()&d.D.notna()&d.dev_raw.notna()].copy();d=d[(d.dnum>=731)&(d.dnum%21==d.farm_id%21)];cut=(pd.Timestamp('2025-06-07')-pd.Timestamp('2022-01-01')).days;d=d[~((d.farm_id==16)&(d.dnum>=cut))]
g=s.groupby(['cid','dnum']).agg(n=('y','size'),nshift=('shift','nunique'),good=('valid','all'),nr=('ref','count'),ny=('ref_y','count'));g=g[(g.n==3)&(g.nshift==3)&g.good&(g.nr==3)&(g.ny==3)].reset_index();d=d.merge(g[['cid','dnum']],on=['cid','dnum']).sort_values(['cid','dnum']).reset_index(drop=True);assert len(d)==4180
# For every observed cow-day choose one record; choices fixed across 3/9-day scenarios.
s['row']=np.arange(len(s));draw=rng.random(len(s));s['random_order']=draw
chosen_random=s.loc[s.groupby(['cid','dnum']).random_order.idxmin(),'row'].to_numpy();chosen_low=s.loc[s.groupby(['cid','dnum']).y.idxmin(),'row'].to_numpy()
# Block endpoint is the next date of the farm-specific 21-day schedule.
s['endpoint']=s.dnum+(s.farm_id-s.dnum)%21;s['distance']=s.endpoint-s.dnum
commonshift={}
for (f,ep),q in s.groupby(['farm_id','endpoint']):
 choices=np.sort(q['shift'].unique());commonshift[(f,ep)]=choices[int((f*31+ep)%len(choices))]
s['block_shift']=[commonshift[(f,ep)] for f,ep in zip(s.farm_id,s.endpoint)]
randommask=np.zeros(len(s),bool);randommask[chosen_random]=True;lowmask=np.zeros(len(s),bool);lowmask[chosen_low]=True

def recompute(a,check_brute=False):
 a=a.sort_values(['cid','t']).copy();c=a.cid.to_numpy();tt=a.t.to_numpy().astype('datetime64[s]').astype(np.int64);dt=np.full(len(a),np.nan);same=c[1:]==c[:-1];dt[1:]=np.where(same,np.diff(tt)/3600,np.nan);a['dt_new']=dt;a['rate_new']=np.where((dt>=5)&(dt<=11.5),a.y.to_numpy(float)/dt,np.nan)
 a=a.sort_values(['cid','shift','t']).reset_index(drop=True);c=a.cid.to_numpy();sh=a['shift'].to_numpy();dn=a.dnum.to_numpy();dt=a.dt_new.to_numpy();y=a.y.to_numpy(float)
 def trailing(x):
  M=np.full((8,len(a)),np.nan)
  for k in range(1,9):
   ok=(c[k:]==c[:-k])&(sh[k:]==sh[:-k])&((dn[k:]-dn[:-k])>=2)&((dn[k:]-dn[:-k])<=8);M[k-1,k:]=np.where(ok,x[:-k],np.nan)
  nr=np.isfinite(M).sum(0);rr=np.nanmedian(M,axis=0);rr[nr<3]=np.nan;return rr
 rr=trailing(a.rate_new.to_numpy());ry=trailing(np.where(dt<=34,y,np.nan));regular=(dt>=5)&(dt<=11.5)&np.isfinite(rr);long=(dt>11.5)&(dt<=34)&np.isfinite(ry);use=regular|long;expected=np.where(regular,rr*dt,np.where(long,ry,np.nan));a['yo']=np.where(use,y,0);a['ye']=np.where(use,expected,0);a['usable']=use
 if check_brute:
  # Independent direct-history selection at random positions (same last-eight convention as source).
  for j in np.random.default_rng(11).choice(len(a),50,replace=False):
   ix=np.arange(max(j-8,0),j);ix=ix[(c[ix]==c[j])&(sh[ix]==sh[j])&(dn[ix]>=dn[j]-8)&(dn[ix]<=dn[j]-2)]
   for xx,out in [(a.rate_new.to_numpy(),rr),(np.where(dt<=34,y,np.nan),ry)]:
    vals=xx[ix];vals=vals[np.isfinite(vals)];val=np.median(vals) if len(vals)>=3 else np.nan;assert np.isclose(val,out[j],equal_nan=True)
 z=a.groupby(['cid','dnum']).agg(yo=('yo','sum'),ye=('ye','sum'),usable=('usable','sum'),records=('y','size'));z['score']=np.log(z.yo/z.ye).where(z.usable>0);return z[['score','usable','records']]
base=recompute(s,True);b=d[['cid','dnum']].join(base,on=['cid','dnum']);err=np.max(abs(b.score-d.D));assert err<2e-6,err;orig=b.score.to_numpy();positive=orig<np.log(.8);assert positive.sum()==175
rows=[];pairs=[];cowidx=pd.factorize(d.cid)[0];nc=cowidx.max()+1;weights=np.random.default_rng(20261115).poisson(1,(500,nc))
def ci_ratio(num,den):
 an=np.bincount(cowidx,weights=num,minlength=nc);ad=np.bincount(cowidx,weights=den,minlength=nc);bb=(weights@an)/(weights@ad);return np.nanquantile(bb,[.025,.975])
for duration in [3,9]:
 for scenario,choice in [('random_one_per_day',randommask),('same_shift_per_block',s['shift'].to_numpy()==s.block_shift.to_numpy()),('lowest_yield_per_day',lowmask)]:
  window=(s.distance.to_numpy()<duration)&(s.endpoint.to_numpy()>=731)&~((s.farm_id.to_numpy()==16)&(s.dnum.to_numpy()>=cut));drop=window&choice;changed=recompute(s.loc[~drop],True);q=d[['cid','dnum']].join(changed,on=['cid','dnum']);val=q.score.to_numpy();avail=np.isfinite(val);flag=avail&(val<np.log(.8));agree=avail&(flag==positive);capture=flag&positive
  for farm,ix in [('ALL',np.arange(len(d)))]+[(str(LAB[f]),np.flatnonzero(d.farm_id.to_numpy()==f)) for f in farms]:
   aa=avail[ix];pp=positive[ix];ff=flag[ix];er=val[ix]-orig[ix]
   r=dict(days=duration,scenario=scenario,farm=farm,n=len(ix),available=int(aa.sum()),coverage=aa.mean(),original_positives=int(pp.sum()),available_positives=int((pp&aa).sum()),retained_positives=int((pp&ff).sum()),positive_retention_all=float((pp&ff).sum()/pp.sum()) if pp.any() else np.nan,positive_retention_available=float((pp&ff).sum()/(pp&aa).sum()) if (pp&aa).any() else np.nan,agreement_available=float((ff[aa]==pp[aa]).mean()) if aa.any() else np.nan,new_drops=int((ff&~pp).sum()),new_drop_rate_allneg=float((ff&~pp).sum()/(~pp).sum()),median_abs_D_error=float(np.nanmedian(abs(er))),p95_abs_D_error=float(np.nanquantile(abs(er),.95)))
   if farm=='ALL':
    for nm,num,den in [('coverage',avail,np.ones(len(d))),('retention',capture,positive),('agreement',agree,avail)]:
     lo,hi=ci_ratio(num,den);r[nm+'_lo']=lo;r[nm+'_hi']=hi
   rows.append(r)
  pairs.append(pd.DataFrame(dict(cid=d.cid,dnum=d.dnum,days=duration,scenario=scenario,original=orig,perturbed=val,available=avail)))
  print(duration,scenario,'available',round(avail.mean(),4),'retention',round(capture.sum()/positive.sum(),4),flush=True)
pd.DataFrame(rows).to_csv(O/'serial_masking_summary.csv',index=False);pd.concat(pairs).to_parquet(O/'serial_masking_pairs_PRIVATE.parquet',index=False)
meta={'seed':20261114,'selected_cows':len(ids),'cows_with_targets':int(d.cid.nunique()),'target_cowdays':len(d),'original_positives':int(positive.sum()),'baseline_max_error':float(err),'brute_reference_checks':350,'cow_cluster_bootstrap_replicates':500,'durations':[3,9],'reference_history_recomputed':True,'intervals_recomputed_from_timestamps':True,'sampling':'Same observed-complete target days as S8; naturally missing records outside/inside reference periods are retained as the starting observation process. Scheduled blocks end 21 days apart.','mechanisms':'Random record each observed cow-day; one common shift throughout a farm block; lowest observed yield each cow-day (constructed yield-dependent stress).','interpretation':'Not a real missingness mechanism or biological ground truth. Unavailable scores are reported separately; positive retention over all original positives counts unavailable scores as not recovered.'};(O/'serial_masking_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
