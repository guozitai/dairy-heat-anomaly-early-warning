#!/usr/bin/env python3
"""E2. Supervised deep sequence models (GRU, Transformer encoder) for Tasks 1 and 2, with the folds, risk sets, negative sampling and
early-stopping cows of C1. Input: the 14 days t-13..t of each cow (per-day channels below, with a presence mask) and static covariates.
Hyperparameters fixed in advance; no tuning on test farms. Scores are evaluated on the rows of the C1 prediction files, against the
deviation rule and gradient boosting of C1. Usage: e2_dl.py <arch gru|tf> <part> <nparts>"""
import pandas as pd, numpy as np, time, sys, os, torch, torch.nn as nn
t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; C1=V+"c1/"; OUT="/home/guozitai/20261112_aiia_dl/e2/"; os.makedirs(OUT,exist_ok=True)
ARCH=sys.argv[1]; PART=int(sys.argv[2]); NPARTS=int(sys.argv[3]); dev="cuda"
def P(*a): print(*a,flush=True)
L=np.log(0.8); W=14
FARMS=[3,5,6,13,14,15,16,17]; LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
FOLDS=[(Y,f) for Y in (2024,2025) for f in FARMS]; MY=[x for i,x in enumerate(FOLDS) if i%NPARTS==PART]
DYNC=["D","lsum","lmy","n_sess","n_dev","sess_gap","cond_dev","flow_dev","H","tmax"]; STAC=["DIM","lbase","since_dis","pg1","pg2","pg3","pg4"]
cols=["cid","dnum","farm_id","date","elig","y_sum","my24","D","lbase","n_sess","n_dev","sess_gap","cond_dev","flow_dev","H","tmax","DIM","pg","since_dis","risk_onset","y_onset","risk_dis","y_dis"]
A=pd.read_parquet(V+"feat_v2.parquet",columns=cols).sort_values(["cid","dnum"]).reset_index(drop=True)
with np.errstate(all="ignore"):
    A["lsum"]=(np.log(A.y_sum)-A.lbase).astype("float32"); A["lmy"]=(np.log(A.my24)-A.lbase).astype("float32")
A["DIM"]=(A.DIM/100).astype("float32"); A["since_dis"]=np.log1p(A.since_dis.clip(upper=400)).astype("float32")
for k in (1,2,3,4): A[f"pg{k}"]=(A.pg==k).astype("float32")
A["year"]=A.date.dt.year; KEY=A.cid.to_numpy().astype(np.int64)*100000+A.dnum.to_numpy().astype(np.int64)
DY=A[DYNC].to_numpy(np.float32); ST=A[STAC].to_numpy(np.float32); P("loaded",len(A),round(time.time()-t0))
def build(idx):
    k0=KEY[idx]; X=np.zeros((len(idx),W,len(DYNC)+1),np.float32)
    for j in range(W):
        q=k0-(W-1-j); pos=np.searchsorted(KEY,q).clip(max=len(KEY)-1); ok=KEY[pos]==q; v=DY[pos]; v[~ok]=np.nan; X[:,j,:-1]=v; X[:,j,-1]=ok
    return X,ST[idx].copy()
class GRUNet(nn.Module):
    def __init__(s,d,ds,h=64):
        super().__init__(); s.rnn=nn.GRU(d,h,batch_first=True); s.head=nn.Sequential(nn.Linear(h+ds,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(s,x,z): _,hN=s.rnn(x); return s.head(torch.cat([hN[-1],z],1)).squeeze(1)
class TFNet(nn.Module):
    def __init__(s,d,ds,h=64):
        super().__init__(); s.inp=nn.Linear(d,h); s.pos=nn.Parameter(torch.zeros(1,W,h)); s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,4,128,0.1,batch_first=True),2); s.head=nn.Sequential(nn.Linear(h+ds,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,1))
    def forward(s,x,z): e=s.enc(s.inp(x)+s.pos); return s.head(torch.cat([e[:,-1],z],1)).squeeze(1)
def prep(X,Z,mu,sd,zmu,zsd):
    X=X.copy(); m=np.isnan(X[...,:-1]); X[...,:-1]=np.where(m,0,(X[...,:-1]-mu)/sd); X[...,:-1]=np.clip(X[...,:-1],-6,6); X=np.concatenate([X,m.astype(np.float32)],2)
    Z=np.where(np.isnan(Z),0,np.clip((Z-zmu)/zsd,-6,6)).astype(np.float32); return X,Z
def predict(net,X,Z):
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),16384): out.append(torch.sigmoid(net(torch.tensor(X[i:i+16384],device=dev),torch.tensor(Z[i:i+16384],device=dev))).cpu().numpy())
    return np.concatenate(out)
def train(itr,iva,ylab,neg,seed):
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    ytr=ylab[itr]; keep=(ytr==1)|(rng.random(len(itr))<neg); itr=itr[keep]; yva=ylab[iva]; kv=(yva==1)|(rng.random(len(iva))<neg); iva=iva[kv]
    Xt,Zt=build(itr); Xv,Zv=build(iva); mu=np.nanmean(Xt[...,:-1].reshape(-1,len(DYNC)),0); sd=np.nanstd(Xt[...,:-1].reshape(-1,len(DYNC)),0)+1e-6; zmu=np.nanmean(Zt,0); zsd=np.nanstd(Zt,0)+1e-6
    zmu[3:]=0; zsd[3:]=1; norm=(mu,sd,zmu,zsd); Xt,Zt=prep(Xt,Zt,*norm); Xv,Zv=prep(Xv,Zv,*norm); yt=ylab[itr].astype(np.float32); yv=ylab[iva].astype(np.float32)
    d=Xt.shape[2]; net=(GRUNet if ARCH=="gru" else TFNet)(d,Zt.shape[1]).to(dev); opt=torch.optim.Adam(net.parameters(),lr=1e-3); lf=nn.BCEWithLogitsLoss()
    Xtg=torch.tensor(Xt,device=dev); Ztg=torch.tensor(Zt,device=dev); ytg=torch.tensor(yt,device=dev); best=(1e9,None,0); bad=0
    for ep in range(40):
        net.train(); perm=torch.randperm(len(Xtg),device=dev)
        for i in range(0,len(perm),2048):
            b=perm[i:i+2048]; opt.zero_grad(); loss=lf(net(Xtg[b],Ztg[b]),ytg[b]); loss.backward(); opt.step()
        pv=np.clip(predict(net,Xv,Zv),1e-6,1-1e-6); vl=-np.mean(yv*np.log(pv)+(1-yv)*np.log(1-pv))
        if vl<best[0]-1e-5: best=(vl,{k:v.detach().clone() for k,v in net.state_dict().items()},ep); bad=0
        else:
            bad+=1
            if bad>=3: break
    net.load_state_dict(best[1]); del Xtg,Ztg,ytg; torch.cuda.empty_cache(); return net,norm,len(itr),best[2]
def order(day,score,tb):
    o=np.lexsort((tb,-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); return o,st
def sel_fixed(o,st,n,k):
    cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); kk=np.ceil(cnt*k).clip(min=1); s=np.zeros(n,bool); s[o]=rank<np.repeat(kk,cnt); return s
rows1=[];rows2=[]
E=A.elig.to_numpy(); yon=A.y_onset.to_numpy(); ydis=A.y_dis.to_numpy(); ro=A.risk_onset.to_numpy(); rd=A.risk_dis.to_numpy()
for (Y,f) in MY:
    p1=C1+f"pred1_{f}_{Y}.parquet"; p2=C1+f"pred2_{f}_{Y}.parquet"
    if not os.path.exists(p1): continue
    tag=f"{LAB[f]} {Y}"; cut=pd.Timestamp(f"{Y-1}-12-31"); pool=E&(A.farm_id.to_numpy()!=f)&(A.date<=cut).to_numpy()
    cows=np.sort(A.cid[pool].unique()); vc=np.random.default_rng([20261111,Y,f,0]).choice(cows,size=int(round(0.2*len(cows))),replace=False); isv=A.cid.isin(vc).to_numpy()
    ti=pd.Series(np.arange(len(A)),index=KEY)
    for task,risk,ylab,neg,pf,rows in ((1,ro,yon,0.25,p1,rows1),(2,rd,ydis,0.08,p2,rows2)):
        itr=np.flatnonzero(pool&risk&~isv); iva=np.flatnonzero(pool&risk&isv)
        net,norm,nfit,ep=train(itr,iva,ylab,neg,20261112+Y+f+task)
        R=pd.read_parquet(pf); idx=ti.reindex(R.cid.to_numpy().astype(np.int64)*100000+R.dnum.to_numpy().astype(np.int64)).to_numpy()
        X,Z=prep(*build(idx.astype(np.int64)),*norm); s=predict(net,X,Z); R["dl"]=s; R[["cid","dnum","dl"]].to_parquet(OUT+f"pred{task}_{ARCH}_{f}_{Y}.parquet",index=False)
        n=len(R); day=R.dnum.to_numpy(); tb=np.random.default_rng([20261112,Y,f,task]).random(n); sc={"R_deviation":R.rule.to_numpy(),"M_ref":R.model.to_numpy(),"DL_"+ARCH:s}
        if task==1:
            y=R.y.to_numpy()
            for nm,v in sc.items(): S=sel_fixed(*order(day,v,tb),n,0.10); rows.append(dict(fold=tag,farm=LAB[f],year=Y,config=nm,events=int(y.sum()),sens10=S[y].mean(),fit_rows=nfit,best_epoch=ep))
        else:
            y=R.y.to_numpy()==1; dnd=R.dn_dis.to_numpy(); pos=np.flatnonzero(y); case=R.cid.to_numpy().astype(np.int64)[pos]*100000+(day[pos]+dnd[pos]).astype(np.int64); cu,ci=np.unique(case,return_inverse=True)
            for nm,v in sc.items():
                o,st=order(day,v,tb); r=dict(fold=tag,farm=LAB[f],year=Y,config=nm,cases=len(cu),fit_rows=nfit,best_epoch=ep)
                for k in (0.05,0.10): Sk=sel_fixed(o,st,n,k); h=np.zeros(len(cu)); np.maximum.at(h,ci,Sk[pos].astype(float)); r[f"case_sens{int(k*100)}"]=h.mean()
                rows.append(r)
        g=[x for x in rows if x["fold"]==tag]; c="sens10" if task==1 else "case_sens5"
        P(f"[{tag}] T{task} {ARCH} fit={nfit:,} ep={ep}: "+" | ".join(f"{x['config']} {x[c]:.3f}" for x in g)+f"  [{round(time.time()-t0)}s]")
    for nm,x in (("t1",rows1),("t2",rows2)): pd.DataFrame(x).to_csv(OUT+f"{nm}_{ARCH}_part{PART}.csv",index=False)
P("DONE",round(time.time()-t0))
