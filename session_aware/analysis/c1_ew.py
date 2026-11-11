#!/usr/bin/env python3
"""C1. Early-warning evaluation, final run (one run feeds every table). Early stopping on 20% of the cows of the training farms (run 2).
Leave-one-farm-out x two out-of-time years (2024, 2025). Usage: c1_ew.py <part> <nparts> [NB]
Task 1: acute-drop onset on day t+1. Lists are formed among all cows with a day deviation on day t that are not in drop on day t;
        the outcome is known for the cows that have a day deviation on day t+1.
Task 2: new veterinary disease record on days t+1..t+3 (risk set: day deviation on day t, no disease record on day t or in the preceding 14 days).
Scores: rules (no fitting), gradient boosting (six feature sets, five seeds for the reference set), logistic regression (reference set).
Chance levels are analytic expectations for random lists of the same size. Ties are broken at random."""
import pandas as pd, numpy as np, time, warnings, sys, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore"); t0=time.time()
V="/home/guozitai/20261111_sat_rescue/v2/"; OUT=V+"c1/"
import os; os.makedirs(OUT,exist_ok=True)
PART=int(sys.argv[1]); NPARTS=int(sys.argv[2]); NB=int(sys.argv[3]) if len(sys.argv)>3 else 500; NT=14
def P(*a): print(*a,flush=True)
L=np.log(0.8); L9=np.log(0.9)
FARMS=[3,5,6,13,14,15,16,17]; LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
FOLDS=[(Y,f) for Y in (2024,2025) for f in FARMS]
PARTS6=[[(2025,17),(2024,3)],[(2025,13),(2024,15)],[(2025,14),(2024,6),(2024,5)],[(2025,16),(2024,13),(2025,3)],[(2024,17),(2025,6)],[(2024,14),(2024,16),(2025,5),(2025,15)]]
MY=PARTS6[PART] if NPARTS==6 else [x for i,x in enumerate(FOLDS) if i%NPARTS==PART]
F=pd.read_parquet(V+"feat_v2.parquet"); F=F[F.elig].reset_index(drop=True); ex=pd.read_parquet(V+"cowday_extra.parquet",columns=["cid","dnum","D_rate","DL"]); F=F.merge(ex,on=["cid","dnum"],how="left")
F=F.sort_values(["cid","dnum"]).reset_index(drop=True); nxt=(F.cid.shift(-1)==F.cid)&(F.dnum.shift(-1)==F.dnum+1)
F["DL_next"]=F.DL.shift(-1).where(nxt); F["nd_next"]=F.n_dev.shift(-1).where(nxt); F["year"]=F.date.dt.year.astype("int16"); F["warm"]=F.date.dt.month.isin([5,6,7,8,9])
G=pd.read_parquet(V+"disease_cid.parquet",columns=["cid","dnum","grp"]).drop_duplicates(["cid","dnum"]); GRP=dict(zip(G.cid.to_numpy().astype(np.int64)*100000+G.dnum.to_numpy().astype(np.int64),G.grp))
P("part",PART,"folds",MY,"eligible cow-days",len(F),round(time.time()-t0))
YH=["D","D_l1","D_l2","D_l3","D_l4","D_l5","D_l6","ma3","min3","ma7","min7","sd7","slope","drops7","drops30","Dmin","loss_kg","lbase","my24","DIM","pg","n_sess","n_dev","sess_gap"]
SE=["cond_dev","cond_dev_l1","condmax_dev","flow_dev"]; HE=["H","hdrop"]; CL=["tmax","tmax_f1","tmax3","dthi","nmin","h72"]; EV=["since_dis"]
DYN=[c for c in YH if c not in ("lbase","my24","DIM","pg")]+SE+HE; STA=["lbase","my24","DIM","pg","since_dis"]
CONF1={"M_yield":YH,"M_yield+sensors":YH+SE,"M_ref":YH+SE+HE,"M_ref+weather":YH+SE+HE+CL,"M_weather+DIM+parity":CL+["DIM","pg"],"M_no_yield":SE+HE+CL+["DIM","pg"]}
CONF2={"M_yield":YH,"M_yield+sensors":YH+SE,"M_ref":YH+SE+HE+EV,"M_ref+weather":YH+SE+HE+EV+CL,"M_weather+DIM+parity":CL+["DIM","pg"],"M_no_yield":SE+HE+EV+CL+["DIM","pg"],"M_dynamic":DYN,"M_static":STA}
RULES1={"R_deviation":("D",True),"R_loss_kg":("loss_kg",False),"R_mean3":("ma3",True),"R_change3":("slope",True),"R_sd7":("sd7",False),"R_drops30":("drops30",False),"R_deviation_regular_only":("D_rate",True),"R_deviation_lagged_reference":("DL",True),"R_my24_deviation":("dev_my24",True),"R_dailysum_deviation":("dev_raw",True)}
RULES2={"R_deviation":("D",True),"R_loss_kg":("loss_kg",False),"R_mean3":("ma3",True),"R_min3":("min3",True),"R_fewer_milkings":("sess_gap",False),"R_deviation_regular_only":("D_rate",True),"R_deviation_lagged_reference":("DL",True),"R_my24_deviation":("dev_my24",True),"R_dailysum_deviation":("dev_raw",True),"R_dailysum_cohort_ratio":("cohort_ratio",True)}
PAR=dict(objective="binary",learning_rate=0.05,num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,bagging_fraction=0.7,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=NT)
A="R_deviation"; B="M_ref"
def order(day,score,tb):
    o=np.lexsort((tb,-score,day)); ds=day[o]; st=np.flatnonzero(np.r_[True,ds[1:]!=ds[:-1]]); return o,st
def sel_fixed(o,st,n,k):
    cnt=np.diff(np.r_[st,n]); rank=np.arange(n)-np.repeat(st,cnt); kk=np.ceil(cnt*k).clip(min=1); s=np.zeros(n,bool); s[o]=rank<np.repeat(kk,cnt); return s
def frac(o,st,n,k):
    """list size / risk-set size of the farm-day of each row (probability of being on a random list)"""
    cnt=np.diff(np.r_[st,n]); kk=np.ceil(cnt*k).clip(min=1); p=np.zeros(n); p[o]=np.repeat(kk/cnt,cnt); return p
def sel_weighted(o,st,w,k):
    ws=w[o]; cw=np.cumsum(ws)-ws; cnt=np.diff(np.r_[st,len(ws)]); base=np.repeat(cw[st],cnt); tot=np.repeat(np.add.reduceat(ws,st),cnt); s=np.zeros(len(ws),bool); s[o]=(cw-base)<np.maximum(np.ceil(tot*k),1); return s
def fill(v):
    v=np.asarray(v,float); ok=np.isfinite(v); return np.where(ok,v,np.nanmedian(v[ok]) if ok.any() else 0.0)
def fit(tr,va,feats,ycol,neg,rng,seed=0):
    keep=(tr[ycol].to_numpy()==1)|(rng.random(len(tr))<neg); a=tr[keep]; kv=(va[ycol].to_numpy()==1)|(rng.random(len(va))<neg); v=va[kv]; cat=["pg"] if "pg" in feats else "auto"; p=dict(PAR); p["seed"]=20261111+seed
    ds=lgb.Dataset(a[feats],label=a[ycol].astype(int),categorical_feature=cat); dv=lgb.Dataset(v[feats],label=v[ycol].astype(int),reference=ds,categorical_feature=cat)
    m=lgb.train(p,ds,num_boost_round=1500,valid_sets=[dv],callbacks=[lgb.early_stopping(80,verbose=False)]); return m,len(a),int(a[ycol].sum())
def logit(tr,te,feats,ycol,neg,rng):
    keep=(tr[ycol].to_numpy()==1)|(rng.random(len(tr))<neg); a=tr[keep]; num=[c for c in feats if c!="pg"]; Xa=a[num].to_numpy(float); Xt=te[num].to_numpy(float)
    med=np.nanmedian(Xa,axis=0); med=np.where(np.isfinite(med),med,0.0); ma=~np.isfinite(Xa); mt=~np.isfinite(Xt); mi=ma.mean(axis=0)>0.01
    Xa=np.where(ma,med,Xa); Xt=np.where(mt,med,Xt); mu=Xa.mean(axis=0); sd=Xa.std(axis=0); sd[sd==0]=1; Xa=np.clip((Xa-mu)/sd,-6,6); Xt=np.clip((Xt-mu)/sd,-6,6)
    pa=np.column_stack([(a.pg.to_numpy()==k).astype(float) for k in (2,3,4)]); pt=np.column_stack([(te.pg.to_numpy()==k).astype(float) for k in (2,3,4)])
    Xa=np.column_stack([Xa,ma[:,mi].astype(float),pa]); Xt=np.column_stack([Xt,mt[:,mi].astype(float),pt])
    m=LogisticRegression(C=1.0,max_iter=300,solver="lbfgs",n_jobs=NT).fit(Xa,a[ycol].astype(int).to_numpy()); return m.predict_proba(Xt)[:,1]
rows1=[];rows2=[];sens=[];cal=[];lead=[];imp=[];seedrows=[];foldinfo=[]
for (Y,f) in MY:
    teA=F[(F.farm_id==f)&(F.year==Y)]
    if len(teA)<30000: continue
    cut=pd.Timestamp(f"{Y-1}-12-31"); pool=F[(F.farm_id!=f)&(F.date<=cut)]; cows=np.sort(pool.cid.unique()); vc=np.random.default_rng([20261111,Y,f,0]).choice(cows,size=int(round(0.2*len(cows))),replace=False); isv=pool.cid.isin(vc).to_numpy(); trA=pool[~isv]; vaA=pool[isv]
    tb_=pool.assign(dimb=(pool.DIM//30).astype(int)).groupby(["pg","dimb"]).y_sum.median().rename("exp_raw").reset_index(); tag=f"{LAB[f]} {Y}"; fi=dict(fold=tag,farm=LAB[f],year=Y,train_farms=int(pool.farm_id.nunique()),train_cows=int(len(cows)-len(vc)),valid_cows=int(len(vc)),train_cowdays=len(trA),valid_cowdays=len(vaA),test_first=str(teA.date.min().date()),test_last=str(teA.date.max().date()),test_days=int(teA.dnum.nunique()))
    # ================= Task 1 =================
    tr=trA[trA.risk_onset]; va=vaA[vaA.risk_onset]; te=teA[teA.D.notna()&(teA.D>=L)].reset_index(drop=True); obs=te.D_next.notna().to_numpy(); y=(te.D_next.to_numpy()<L)&obs
    if obs.sum()>=20000 and y.sum()>=200 and len(tr)>200000:
        rng=np.random.default_rng([20261111,Y,f,1]); n=len(te); day=te.dnum.to_numpy(); tbk=rng.random(n); cowi=pd.factorize(te.cid)[0]; nc=cowi.max()+1
        ys=y&(te.D_next2.to_numpy()<L9); oa=te.DL_next.notna().to_numpy(); ya=(te.DL_next.to_numpy()<L)&oa; ndv=te.n_dev.clip(upper=3).to_numpy(); warm=te.warm.to_numpy()
        fi.update(t1_train=len(tr),t1_train_events=int(tr.y_onset.sum()),t1_test=n,t1_test_observed=int(obs.sum()),t1_onsets=int(y.sum()),t1_confirmed=int(ys.sum()))
        sc={}
        for nm,(col,asc) in RULES1.items(): v=fill(te[col]); sc[nm]=-v if asc else v
        for nm,feats in CONF1.items():
            m,na,ne=fit(tr,va,feats,"y_onset",0.25,rng); sc[nm]=m.predict(te[feats],num_iteration=m.best_iteration)
            if nm==B:
                fi.update(t1_fit_rows=na,t1_fit_events=ne,t1_iterations=int(m.best_iteration)); g=pd.Series(m.feature_importance("gain"),index=feats); imp.append(dict(task="onset",fold=tag,**(g/g.sum()).round(4).to_dict()))
                pc=sc[nm]*0.25/(sc[nm]*0.25+1-sc[nm]); q=pd.qcut(pc[obs],10,labels=False,duplicates="drop"); cal.append(pd.DataFrame({"task":"onset","fold":tag,"bin":q,"p":pc[obs],"y":y[obs].astype(int)}).groupby(["task","fold","bin"]).agg(p=("p","mean"),y=("y","mean"),n=("y","size")).reset_index())
        sc["M_logistic"]=logit(tr,te,CONF1[B],"y_onset",0.25,rng)
        pre={nm:order(day,s,tbk) for nm,s in sc.items()}; pch=frac(*pre[A],n,0.10)
        for nm,s in sc.items():
            o,st=pre[nm]; S=sel_fixed(o,st,n,0.10); det=(S&y).sum()
            r=dict(fold=tag,farm=LAB[f],year=Y,config=nm,n=n,n_obs=int(obs.sum()),events=int(y.sum()),rate=y.sum()/obs.sum(),sens10=S[y].mean(),chance10=pch[y].mean(),prec10=y[S&obs].mean(),entries_per_detection=S.sum()/max(det,1),unobs_listed=(~obs)[S].mean(),unobs_unlisted=(~obs)[~S].mean(),
                   sens10_conf=S[ys].mean() if ys.sum() else np.nan,n_conf=int(ys.sum()),sens10_alt=S[ya].mean() if ya.sum() else np.nan,n_alt=int(ya.sum()),pr_lift=average_precision_score(y[obs],s[obs])/y[obs].mean(),auc=roc_auc_score(y[obs],s[obs]))
            for k in (1,2,3): r[f"sens10_u{k}"]=S[y&(ndv==k)].mean() if (y&(ndv==k)).sum()>=30 else np.nan; r[f"n_u{k}"]=int((y&(ndv==k)).sum())
            for nmk,mk in (("warm",warm),("cool",~warm)): r["sens10_"+nmk]=S[y&mk].mean() if (y&mk).sum()>=100 else np.nan
            rows1.append(r)
        bs=[]
        for _ in range(NB):
            w=rng.poisson(1.0,nc).astype(float)[cowi]; Sa=sel_weighted(*pre[A],w,0.10); Sb=sel_weighted(*pre[B],w,0.10); wy=w*y; wys=w*ys; bs.append(((wy*Sa).sum()/wy.sum(),(wy*Sb).sum()/wy.sum(),(wys*Sa).sum()/max(wys.sum(),1e-9),(wys*Sb).sum()/max(wys.sum(),1e-9)))
        bs=np.array(bs)
        for r in rows1[-len(sc):]:
            if r["config"]==A: r["lo"],r["hi"]=np.percentile(bs[:,0],[2.5,97.5]); r["conf_lo"],r["conf_hi"]=np.percentile(bs[:,2],[2.5,97.5])
            if r["config"]==B: r["lo"],r["hi"]=np.percentile(bs[:,1],[2.5,97.5]); r["diff_lo"],r["diff_hi"]=np.percentile(bs[:,1]-bs[:,0],[2.5,97.5]); r["conf_lo"],r["conf_hi"]=np.percentile(bs[:,3],[2.5,97.5]); r["conf_diff_lo"],r["conf_diff_hi"]=np.percentile(bs[:,3]-bs[:,2],[2.5,97.5])
        for sd in range(1,5):
            m,na,ne=fit(tr,va,CONF1[B],"y_onset",0.25,np.random.default_rng([20261111,Y,f,1,sd]),seed=sd); p=m.predict(te[CONF1[B]],num_iteration=m.best_iteration); S=sel_fixed(*order(day,p,tbk),n,0.10); seedrows.append(dict(task="onset",fold=tag,seed=sd,sens=S[y].mean(),sens_conf=S[ys].mean() if ys.sum() else np.nan,iterations=int(m.best_iteration)))
        seedrows.append(dict(task="onset",fold=tag,seed=0,sens=[x for x in rows1[-len(sc):] if x["config"]==B][0]["sens10"],sens_conf=[x for x in rows1[-len(sc):] if x["config"]==B][0]["sens10_conf"],iterations=fi["t1_iterations"]))
        pd.DataFrame({"cid":te.cid.to_numpy(),"dnum":day,"obs":obs,"y":y,"ys":ys,"D":te.D.to_numpy(),"n_dev":te.n_dev.to_numpy(),"rule":sc[A],"model":sc[B],"model_weather":sc["M_ref+weather"],"logistic":sc["M_logistic"]}).to_parquet(OUT+f"pred1_{f}_{Y}.parquet",index=False)
        g1=lambda nm,c: [x for x in rows1[-len(sc):] if x["config"]==nm][0][c]
        P(f"[{tag}] T1 n={n:,} obs={int(obs.sum()):,} onsets={int(y.sum()):,}: rule {g1(A,'sens10'):.3f} | model {g1(B,'sens10'):.3f} | +weather {g1('M_ref+weather','sens10'):.3f} | logistic {g1('M_logistic','sens10'):.3f} | chance {g1(A,'chance10'):.3f} | conf rule {g1(A,'sens10_conf'):.3f} model {g1(B,'sens10_conf'):.3f}  [{round(time.time()-t0)}s]")
        # sensitivity analyses with refit: thresholds, >=2 usable milkings on both days, DIM 60-330
        for thr in (0.85,0.75,0.70):
            Lt=np.log(thr); rk=lambda d: d[d.D.notna()&(d.D>=Lt)&d.D_next.notna()].assign(yy=lambda q:(q.D_next<Lt).astype("int8")); t2=teA[teA.D.notna()&(teA.D>=Lt)].reset_index(drop=True); o2=t2.D_next.notna().to_numpy(); y2=(t2.D_next.to_numpy()<Lt)&o2
            if y2.sum()<150: continue
            m,_,_=fit(rk(trA),rk(vaA),CONF1[B],"yy",0.25,np.random.default_rng([20261111,Y,f,1,int(thr*100)])); d2=t2.dnum.to_numpy(); tb2=rng.random(len(t2)); Sr=sel_fixed(*order(d2,-fill(t2.D),tb2),len(t2),0.10); Sm=sel_fixed(*order(d2,m.predict(t2[CONF1[B]],num_iteration=m.best_iteration),tb2),len(t2),0.10)
            sens.append(dict(analysis=f"Task 1, threshold {int(thr*100)}%",fold=tag,rate=y2.sum()/o2.sum(),rule=Sr[y2].mean(),model=Sm[y2].mean()))
        for nm,mk in (("Task 1, at least two usable milkings on both days",lambda d:(d.n_dev>=2)),("Task 1, DIM 60 to 330",lambda d:(d.DIM>=60)&(d.DIM<=330))):
            two=nm.startswith("Task 1, at least"); t2=teA[teA.D.notna()&(teA.D>=L)&mk(teA)].reset_index(drop=True); o2=t2.D_next.notna().to_numpy()&((t2.nd_next.to_numpy()>=2) if two else True); y2=(t2.D_next.to_numpy()<L)&o2
            if y2.sum()<150: continue
            rk=lambda d: d[d.risk_onset&mk(d)&((d.nd_next>=2) if two else True)]
            m,_,_=fit(rk(trA),rk(vaA),CONF1[B],"y_onset",0.25,np.random.default_rng([20261111,Y,f,1,7 if two else 8])); d2=t2.dnum.to_numpy(); tb2=rng.random(len(t2)); Sr=sel_fixed(*order(d2,-fill(t2.D),tb2),len(t2),0.10); Sm=sel_fixed(*order(d2,m.predict(t2[CONF1[B]],num_iteration=m.best_iteration),tb2),len(t2),0.10)
            sens.append(dict(analysis=nm,fold=tag,rate=y2.sum()/o2.sum(),rule=Sr[y2].mean(),model=Sm[y2].mean()))
        for nmk in ("warm","cool"): sens.append(dict(analysis="Task 1, May to September" if nmk=="warm" else "Task 1, October to April",fold=tag,rate=np.nan,rule=g1(A,"sens10_"+nmk),model=g1(B,"sens10_"+nmk)))
        for k in (1,2,3): sens.append(dict(analysis=f"Task 1, onsets of cows with {k} usable milking{'s' if k>1 else ''} on day t",fold=tag,rate=np.nan,rule=g1(A,f"sens10_u{k}"),model=g1(B,f"sens10_u{k}")))
        sens.append(dict(analysis="Task 1, confirmed onsets",fold=tag,rate=ys.sum()/obs.sum(),rule=g1(A,"sens10_conf"),model=g1(B,"sens10_conf"))); sens.append(dict(analysis="Task 1, next-day drop defined with the reference window t - 20 to t - 8",fold=tag,rate=ya.sum()/max(oa.sum(),1),rule=g1(A,"sens10_alt"),model=g1(B,"sens10_alt")))
    # ================= Task 2 =================
    tr=trA[trA.risk_dis]; va=vaA[vaA.risk_dis]; te=teA[teA.risk_dis].copy(); te["dimb"]=(te.DIM//30).astype(int); te=te.merge(tb_,on=["pg","dimb"],how="left"); te["cohort_ratio"]=te.y_sum/te.exp_raw; te=te.reset_index(drop=True)
    if te.y_dis.sum()>=30 and len(tr)>200000:
        rng=np.random.default_rng([20261111,Y,f,2]); y=te.y_dis.to_numpy()==1; n=len(te); day=te.dnum.to_numpy(); tbk=rng.random(n); cowi=pd.factorize(te.cid)[0]; nc=cowi.max()+1; dnd=te.dn_dis.to_numpy()
        pos=np.flatnonzero(y); case=te.cid.to_numpy().astype(np.int64)[pos]*100000+(day[pos]+dnd[pos]).astype(np.int64); cu,ci=np.unique(case,return_inverse=True); ncase=len(cu); case_cow=np.zeros(ncase,int); case_cow[ci]=cowi[pos]
        grp=np.array([GRP.get(int(k),"other") for k in cu],dtype=object); nrd=np.bincount(ci,minlength=ncase); early=dnd[pos]>=2; d1=dnd[pos]==1; has_early=np.bincount(ci,weights=early.astype(float),minlength=ncase)>0; has_d1=np.bincount(ci,weights=d1.astype(float),minlength=ncase)>0
        hot=(te.tmax3.to_numpy()>=77); mild=(te.tmax3.to_numpy()<=72); warm=te.warm.to_numpy(); nd2=te.n_dev.to_numpy()>=2
        fi.update(t2_train=len(tr),t2_train_events=int(tr.y_dis.sum()),t2_test=n,t2_cases=ncase,t2_pos=int(y.sum()),t2_cases_1day=int((nrd==1).sum()),t2_cases_2days=int((nrd==2).sum()),t2_cases_3days=int((nrd==3).sum()))
        sc={}
        for nm,(col,asc) in RULES2.items(): v=fill(te[col]); sc[nm]=-v if asc else v
        for nm,feats in CONF2.items():
            m,na,ne=fit(tr,va,feats,"y_dis",0.08,rng); sc[nm]=m.predict(te[feats],num_iteration=m.best_iteration)
            if nm==B:
                fi.update(t2_fit_rows=na,t2_fit_events=ne,t2_iterations=int(m.best_iteration)); g=pd.Series(m.feature_importance("gain"),index=feats); imp.append(dict(task="disease",fold=tag,**(g/g.sum()).round(4).to_dict()))
                pc=sc[nm]*0.08/(sc[nm]*0.08+1-sc[nm]); q=pd.qcut(pc,10,labels=False,duplicates="drop"); cal.append(pd.DataFrame({"task":"disease","fold":tag,"bin":q,"p":pc,"y":y.astype(int)}).groupby(["task","fold","bin"]).agg(p=("p","mean"),y=("y","mean"),n=("y","size")).reset_index())
        sc["M_logistic"]=logit(tr,te,CONF2[B],"y_dis",0.08,rng)
        pct=lambda s: pd.Series(s).groupby(day).rank(pct=True).to_numpy(); sc["B_rank_mean(rule,model)"]=pct(sc[A])+pct(sc[B])
        pre={nm:order(day,s,tbk) for nm,s in sc.items()}
        def csens(S,wcase=None,mask=None,rows=None):
            h=np.zeros(ncase); sel=S[pos].astype(float) if rows is None else (S[pos]&rows).astype(float); np.maximum.at(h,ci,sel)
            if mask is not None: h=h[mask]; wcase=None if wcase is None else wcase[mask]
            return h.mean() if wcase is None else (h*wcase).sum()/max(wcase.sum(),1e-9)
        det={}
        for nm,s in sc.items():
            o,st=pre[nm]; r=dict(fold=tag,farm=LAB[f],year=Y,config=nm,n=n,cases=ncase,pos=int(y.sum()),auc=roc_auc_score(y,s),pr_lift=average_precision_score(y,s)/y.mean())
            for k in (0.02,0.05,0.10):
                S=sel_fixed(o,st,n,k); kk=int(k*100); cs=csens(S); r[f"case_sens{kk}"]=cs; r[f"cowday_sens{kk}"]=S[y].mean(); r[f"prec{kk}"]=y[S].mean(); r[f"entries_per_case{kk}"]=S.sum()/max(cs*ncase,1)
                if nm==A:
                    p=frac(o,st,n,k); q=np.ones(ncase); np.multiply.at(q,ci,1-p[pos]); r[f"chance{kk}"]=(1-q).mean(); r[f"chance_prec{kk}"]=y.mean(); r[f"chance_entries{kk}"]=S.sum()/max((1-q).sum(),1e-9)
                if k in (0.05,0.10):
                    r[f"case_sens{kk}_early"]=csens(S,mask=has_early,rows=early); r[f"case_sens{kk}_d1"]=csens(S,mask=has_d1,rows=d1); h=np.zeros(ncase); np.maximum.at(h,ci,S[pos].astype(float)); det[(nm,kk)]=h; he=np.zeros(ncase); np.maximum.at(he,ci,(S[pos]&early).astype(float)); det[(nm,kk,"early")]=he
                    for g_ in ("mastitis","digestive","metabolic","lameness","respiratory","reproductive","other"):
                        mk=grp==g_; r[f"ncase_{g_}"]=int(mk.sum())
                        if mk.sum()>=20: r[f"case_sens{kk}_{g_}"]=csens(S,mask=mk)
                    for nmk,rw in (("warm",warm),("cool",~warm),("u2",nd2)):
                        mk=np.bincount(ci,weights=rw[pos].astype(float),minlength=ncase)>0
                        if mk.sum()>=20: r[f"case_sens{kk}_{nmk}"]=csens(S,mask=mk,rows=rw[pos])
                    if (y&hot).sum()>=30: r[f"hot_sens{kk}"]=S[y&hot].mean()
                    if (y&mild).sum()>=30: r[f"mild_sens{kk}"]=S[y&mild].mean()
            r["n_early"]=int(has_early.sum()); r["n_d1"]=int(has_d1.sum()); rows2.append(r)
            if nm in (A,B):
                S=sel_fixed(o,st,n,0.05); p=frac(o,st,n,0.05)
                for kd in range(1,8):
                    mk=(dnd==kd)
                    if mk.sum()>0: lead.append(dict(fold=tag,config=nm,days_before=kd,n=int(mk.sum()),on_list=float(S[mk].mean()),chance=float(p[mk].mean())))
        # window 1-7 d (same lists)
        y7=(dnd<=7); p7=np.flatnonzero(y7); c7=te.cid.to_numpy().astype(np.int64)[p7]*100000+(day[p7]+dnd[p7]).astype(np.int64); cu7,ci7=np.unique(c7,return_inverse=True)
        def cs7(S): h=np.zeros(len(cu7)); np.maximum.at(h,ci7,S[p7].astype(float)); return h.mean()
        Sa5=sel_fixed(*pre[A],n,0.05); Sb5=sel_fixed(*pre[B],n,0.05); sens.append(dict(analysis="Task 2, records within 1 to 7 days",fold=tag,rate=y7.mean(),rule=cs7(Sa5),model=cs7(Sb5)))
        g2=lambda nm,c: [x for x in rows2[-len(sc):] if x["config"]==nm][0].get(c,np.nan)
        for lab_,c_ in (("Task 2, May to September","case_sens5_warm"),("Task 2, October to April","case_sens5_cool"),("Task 2, days with at least two usable milkings","case_sens5_u2"),("Task 2, lists of days -3 and -2 only","case_sens5_early"),("Task 2, list of day -1 only","case_sens5_d1")): sens.append(dict(analysis=lab_,fold=tag,rate=np.nan,rule=g2(A,c_),model=g2(B,c_)))
        bs=[]
        for _ in range(NB):
            wc=rng.poisson(1.0,nc).astype(float); w=wc[cowi]; wcase=wc[case_cow]; out=[]
            for k in (0.05,0.10):
                for nm in (A,B): out.append(csens(sel_weighted(*pre[nm],w,k),wcase))
            bs.append(out)
        bs=np.array(bs)
        for r in rows2[-len(sc):]:
            if r["config"]==A: r["cs5_lo"],r["cs5_hi"]=np.percentile(bs[:,0],[2.5,97.5]); r["cs10_lo"],r["cs10_hi"]=np.percentile(bs[:,2],[2.5,97.5])
            if r["config"]==B: r["cs5_lo"],r["cs5_hi"]=np.percentile(bs[:,1],[2.5,97.5]); r["cs10_lo"],r["cs10_hi"]=np.percentile(bs[:,3],[2.5,97.5]); r["d5_lo"],r["d5_hi"]=np.percentile(bs[:,1]-bs[:,0],[2.5,97.5]); r["d10_lo"],r["d10_hi"]=np.percentile(bs[:,3]-bs[:,2],[2.5,97.5])
        for sd in range(1,5):
            m,na,ne=fit(tr,va,CONF2[B],"y_dis",0.08,np.random.default_rng([20261111,Y,f,2,sd]),seed=sd); p=m.predict(te[CONF2[B]],num_iteration=m.best_iteration); oo=order(day,p,tbk); seedrows.append(dict(task="disease",fold=tag,seed=sd,sens=csens(sel_fixed(*oo,n,0.05)),sens10=csens(sel_fixed(*oo,n,0.10)),iterations=int(m.best_iteration)))
        seedrows.append(dict(task="disease",fold=tag,seed=0,sens=g2(B,"case_sens5"),sens10=g2(B,"case_sens10"),iterations=fi["t2_iterations"]))
        pd.DataFrame({"fold":tag,"farm":LAB[f],"year":Y,"case":cu,"cid":cu//100000,"rec_dnum":cu%100000,"grp":grp,"risk_days":nrd,"has_early":has_early,"rule5":det[(A,5)],"rule10":det[(A,10)],"model5":det[(B,5)],"model10":det[(B,10)],"rule5_early":det[(A,5,"early")],"model5_early":det[(B,5,"early")],"blend5":det[("B_rank_mean(rule,model)",5)],"dailysum5":det[("R_dailysum_deviation",5)],"cohort5":det[("R_dailysum_cohort_ratio",5)]}).to_parquet(OUT+f"cases_{f}_{Y}.parquet",index=False)
        pd.DataFrame({"cid":te.cid.to_numpy(),"dnum":day,"y":y,"dn_dis":dnd,"D":te.D.to_numpy(),"rule":sc[A],"model":sc[B],"model_weather":sc["M_ref+weather"],"logistic":sc["M_logistic"]}).to_parquet(OUT+f"pred2_{f}_{Y}.parquet",index=False)
        P(f"[{tag}] T2 n={n:,} cases={ncase}: chance {g2(A,'chance5'):.3f} | cohort {g2('R_dailysum_cohort_ratio','case_sens5'):.3f} | daily-sum dev {g2('R_dailysum_deviation','case_sens5'):.3f} | rule {g2(A,'case_sens5'):.3f} | model {g2(B,'case_sens5'):.3f} | logistic {g2('M_logistic','case_sens5'):.3f} | blend {g2('B_rank_mean(rule,model)','case_sens5'):.3f} | early rule {g2(A,'case_sens5_early'):.3f} model {g2(B,'case_sens5_early'):.3f}  [{round(time.time()-t0)}s]")
    foldinfo.append(fi)
    for nm,x in (("ew1",rows1),("ew2",rows2),("sens",sens),("lead",lead),("imp",imp),("seeds",seedrows),("folds",foldinfo)): pd.DataFrame(x).to_csv(OUT+f"{nm}_part{PART}.csv",index=False)
    if cal: pd.concat(cal).to_csv(OUT+f"cal_part{PART}.csv",index=False)
P("DONE",round(time.time()-t0))
