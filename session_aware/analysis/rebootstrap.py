from pathlib import Path
import ctypes as ct,numpy as np,pandas as pd,json,sys,concurrent.futures,time
W=Path('/home/guozitai/20261114_sat_review_audit');V=Path('/home/guozitai/20261111_sat_rescue/v2');W.mkdir(exist_ok=True)
I=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS');D=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
lib=ct.CDLL(str(W/'weighted_bootstrap.so'));fn=lib.evaluate;fn.restype=ct.c_double
fn.argtypes=[ct.c_int64]*4+[I,I,I,I,D,I,I,ct.c_double,ct.c_int,D,D]
def evaluate(cow,day,order,starts,wc,ci,casecow,k,mode,total=None,selected=False):
 if total is None:total=np.bincount(day,weights=wc[cow],minlength=len(starts)).astype(float)
 out=np.zeros(len(cow),dtype=float)
 value=fn(len(cow),len(starts),len(wc),len(casecow),cow,day,order,starts,wc,ci,casecow,k,mode,total,out)
 return (value,out) if selected else value
# Literal expansion is an independent oracle for repeated-copy boundary selection.
def selftest():
 rng=np.random.default_rng(812);tests=0
 for mode in [1,2]:
  for _ in range(100):
   cows=np.tile(np.arange(8,dtype=np.int64),3);day=np.repeat(np.arange(3,dtype=np.int64),8);score=rng.random(24);order=np.lexsort((cows,-score,day)).astype(np.int64);starts=np.array([0,8,16],np.int64);wc=rng.integers(0,5,8).astype(float)
   ci=np.where(np.isin(cows,[1,3,5]),cows//2,-1).astype(np.int64);casecow=np.array([1,3,5],np.int64)
   value,out=evaluate(cows,day,order,starts,wc,ci,casecow,.25,mode,selected=True)
   oracle=np.zeros(24)
   for d in range(3):
    expanded=[(i,j) for i in order[day[order]==d] for j in range(int(wc[cows[i]]))];take=expanded[:max(1,int(np.ceil(len(expanded)*.25)))]
    for i,j in take:oracle[i]+=1
   assert np.array_equal(oracle,out)
   if mode==1:num=oracle[ci>=0].sum();den=wc[cows[ci>=0]].sum()
   else:
    num=sum(max(oracle[ci==i],default=0) for i in range(3));den=wc[casecow].sum()
   if den:assert abs(value-num/den)<1e-12
   tests+=1
 (W/'bootstrap_selftest.json').write_text(json.dumps({'literal_expansion_cases':tests,'all_passed':True}));print('selftests',tests,flush=True)
def run(job):
 task,f,Y=job;lab={6:1,17:2,16:3,13:4,14:5,15:6,5:7,3:8};tag=f'Farm {lab[f]} {Y}';d=pd.read_parquet(V/f'c1/pred{task}_{f}_{Y}.parquet');n=len(d)
 _,cow=np.unique(d.cid,return_inverse=True);cow=cow.astype(np.int64);_,day=np.unique(d.dnum,return_inverse=True);day=day.astype(np.int64);nc=int(cow.max()+1);starts=np.r_[0,np.cumsum(np.bincount(day))[:-1]].astype(np.int64)
 tb=np.random.default_rng([20261111,Y,f,task]).random(n)
 orders=[np.lexsort((tb,-d[col].to_numpy(),day)).astype(np.int64) for col in ['rule','model']]
 y=d.y.to_numpy().astype(bool)
 if task==1:
  ci=np.where(y,0,-1).astype(np.int64);ys=np.where(d.ys.to_numpy().astype(bool),0,-1).astype(np.int64);cc=np.empty(0,np.int64)
 else:
  pos=np.flatnonzero(y);key=d.cid.to_numpy().astype(np.int64)[pos]*100000+d.dnum.to_numpy().astype(np.int64)[pos]+d.dn_dis.to_numpy().astype(np.int64)[pos];cu,inv=np.unique(key,return_inverse=True);ci=np.full(n,-1,np.int64);ci[pos]=inv;cc=np.zeros(len(cu),np.int64);cc[inv]=cow[pos]
 wc=np.ones(nc);total=np.bincount(day).astype(float)
 if task==1:point=[evaluate(cow,day,o,starts,wc,ci,cc,.1,1,total) for o in orders]
 else:point=[evaluate(cow,day,o,starts,wc,ci,cc,k,2,total) for k in [.05,.1] for o in orders]
 rng=np.random.default_rng([20261114,task,f,Y]);out=[]
 for rep in range(500):
  wc=rng.poisson(1,nc).astype(float);total=np.bincount(day,weights=wc[cow],minlength=len(starts)).astype(float)
  if task==1:r=[evaluate(cow,day,o,starts,wc,c,cc,.1,1,total) for c in [ci,ys] for o in orders]
  else:r=[evaluate(cow,day,o,starts,wc,ci,cc,k,2,total) for k in [.05,.1] for o in orders]
  out.append(r)
 arr=np.array(out);pd.DataFrame(arr).to_csv(W/f'bootstrap_task{task}_farm{lab[f]}_{Y}.csv',index=False)
 stats=dict(task=task,fold=tag,point=point,replicates=500,lo=np.nanquantile(arr,.025,axis=0).tolist(),hi=np.nanquantile(arr,.975,axis=0).tolist(),diff_lo=[float(np.nanquantile(arr[:,j+1]-arr[:,j],.025)) for j in [0,2]],diff_hi=[float(np.nanquantile(arr[:,j+1]-arr[:,j],.975)) for j in [0,2]])
 (W/f'stats_task{task}_farm{lab[f]}_{Y}.json').write_text(json.dumps(stats));return tag,task
if __name__=='__main__':
 selftest();jobs=[(t,f,Y) for t in [1,2] for Y in [2024,2025] for f in [3,5,6,13,14,15,16,17]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
  for z in ex.map(run,jobs):print('DONE',z,flush=True)
 stats=[json.loads(p.read_text()) for p in sorted(W.glob('stats_*.json'))];(W/'all_stats.json').write_text(json.dumps(stats,indent=2));print('FINISHED',len(stats),flush=True)
