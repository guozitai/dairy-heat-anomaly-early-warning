#!/usr/bin/env python3
"""C4. Acute drops before veterinary and management records, with matched controls free of the event (controls selected by the same rule as the records).
Controls: same farm, day, parity class and 30-day DIM class, with a day deviation on the record day; for disease records no disease record of the control
cow from 14 days before to 3 days after the record day; for management records no record of the same kind within 3 days.
Shares by window (days -3..-1, -1, -2, -3, -3..-2, -7..-4, 0), thresholds 85/80/75/70%, lagged-reference deviation, daily-sum deviation; cow-clustered bootstrap."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; E="/home/guozitai/20261014_event_layer/parquet/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",260); L=np.log(0.8); rng=np.random.default_rng(20261111)
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","D","dev_raw","elig","risk_onset","y_onset","DIM","par","dp_dis","dn_dis"]); F=F[F.elig&F.D.notna()].copy(); F["pc"]=F.par.clip(1,3).astype("int8"); F["DIM"]=F.DIM.astype("float32")
ex=pd.read_parquet(V+"cowday_extra.parquet",columns=["cid","dnum","DL"]); F=F.merge(ex,on=["cid","dnum"],how="left")
X=F[["cid","dnum","D","dev_raw","DL"]]; ct=pd.read_parquet(V+"cowtab.parquet"); cov=F.groupby("farm_id").dnum.agg(["min","max"])
def load(name):
    e=pd.read_parquet(E+name+".parquet",columns=["farm_id","cow_number","event_time"]); e["farm_id"]=pd.to_numeric(e.farm_id,errors="coerce"); e=e.dropna(subset=["farm_id"]); e["farm_id"]=e.farm_id.astype(int); e["cow"]=e.cow_number.astype(str)
    e["dnum"]=(pd.to_datetime(e.event_time,errors="coerce").dt.normalize()-pd.Timestamp("2022-01-01")).dt.days; e=e.dropna(subset=["dnum"]).merge(ct[["cid","farm_id","cow"]],on=["farm_id","cow"]); e["dnum"]=e.dnum.astype(int)
    return e[["cid","farm_id","dnum"]].drop_duplicates()
OFF=list(range(-10,4))
def table(cases):
    parts=[]
    for k in OFF:
        t=cases[["case","cid","ev"]].copy(); t["k"]=k; t["dnum"]=t.ev+k; parts.append(t)
    return pd.concat(parts,ignore_index=True).merge(X,on=["cid","dnum"],how="inner")
WIN={"pre3":(-3,-1,2),"d1":(-1,-1,1),"d2":(-2,-2,1),"d3":(-3,-3,1),"pre32":(-3,-2,1),"pre7_4":(-7,-4,3),"day0":(0,0,1)}
def study(cases,label,traj_out,allev,disease,ctrl_n=3):
    cases=cases.rename(columns={"dnum":"ev"}).drop_duplicates(["cid","ev"]).reset_index(drop=True); cases["case"]=np.arange(len(cases)); n_all=len(cases)
    parts=[]
    for k in (1,2,3):
        t=cases[["case","cid","ev"]].copy(); t["dnum"]=t.ev-k; t=t.merge(F[["cid","dnum","pc","DIM"]],on=["cid","dnum"],how="inner"); t["dimr"]=t.DIM+k; t["k"]=k; parts.append(t)
    st=pd.concat(parts).sort_values(["case","k"]).drop_duplicates("case"); st["dc"]=(st.dimr//30).astype(int); cases=cases.merge(st[["case","pc","dc"]],on="case",how="inner")
    # controls are selected by the same rule as the records: a value of D on at least one of days -3..-1 (strata from the latest such day)
    days=cases[["farm_id","ev"]].drop_duplicates(); pp=[]
    for k in (1,2,3):
        t=days.assign(dnum=days.ev-k).merge(F[["cid","farm_id","dnum","pc","DIM","dp_dis","dn_dis"]],on=["farm_id","dnum"],how="inner"); t["k"]=k
        if disease: t=t[(t.dp_dis.isna()|(t.dp_dis>14-k))&(t.dn_dis.isna()|(t.dn_dis>3+k))]
        pp.append(t)
    pool=pd.concat(pp).sort_values(["farm_id","ev","cid","k"]).drop_duplicates(["farm_id","ev","cid"]); pool["dc"]=((pool.DIM+pool.k)//30).astype(int)
    if disease:
        bad=pd.concat(pp).groupby(["farm_id","ev","cid"]).size().rename("nk").reset_index(); allk=pd.concat([days.assign(dnum=days.ev-k).merge(F[["cid","farm_id","dnum","dp_dis","dn_dis"]],on=["farm_id","dnum"],how="inner").assign(k=k) for k in (1,2,3)]); allk["dis"]=~((allk.dp_dis.isna()|(allk.dp_dis>14-allk.k))&(allk.dn_dis.isna()|(allk.dn_dis>3+allk.k))); dz=allk.groupby(["farm_id","ev","cid"]).dis.any().rename("anydis").reset_index(); pool=pool.merge(dz,on=["farm_id","ev","cid"],how="left"); pool=pool[~pool.anydis.fillna(False)]
    else:
        near=pd.concat([allev.assign(ev=allev.dnum+k)[["cid","ev"]] for k in range(-3,4)]).drop_duplicates().assign(hit=1); pool=pool.merge(near,on=["cid","ev"],how="left"); pool=pool[pool.hit.isna()]
    pool=pool.merge(cases[["cid","ev"]].assign(is_case=1),on=["cid","ev"],how="left"); pool=pool[pool.is_case.isna()]
    pool=pool[["cid","farm_id","ev","pc","dc"]]; pool["u"]=rng.random(len(pool)); cnt=cases.groupby(["farm_id","ev","pc","dc"]).size().rename("nc").reset_index(); pool=pool.merge(cnt,on=["farm_id","ev","pc","dc"]); pool["r"]=pool.groupby(["farm_id","ev","pc","dc"]).u.rank(method="first")
    ctl=pool[pool.r<=pool.nc*ctrl_n][["cid","farm_id","ev"]].reset_index(drop=True); ctl["case"]=np.arange(len(ctl))
    Lc=table(cases); Lk=table(ctl)
    tr=Lc.groupby("k").D.agg(["mean","sem","size"]).reset_index().assign(group=label,arm="case"); tk=Lk.groupby("k").D.agg(["mean","sem","size"]).reset_index().assign(group=label,arm="control"); traj_out.append(pd.concat([tr,tk]))
    def share(Lg,a,b,need,col="D",thr=L):
        w=Lg[(Lg.k>=a)&(Lg.k<=b)].dropna(subset=[col]); w=w.assign(dr=(w[col]<thr)).groupby("case").dr.agg(["size","sum"]); w=w[w["size"]>=need]; return (w["sum"]>0)
    out=dict(group=label,records=n_all,examined=len(cases),controls=len(ctl))
    cow_of=cases.set_index("case").cid
    for nm,(a,b,need) in WIN.items():
        x=share(Lc,a,b,need); y=share(Lk,a,b,need); out[nm+"_n"]=len(x); out[nm]=x.mean() if len(x) else np.nan; out[nm+"_ctrl"]=y.mean() if len(y) else np.nan; out[nm+"_nctrl"]=len(y)
        if nm in ("pre3","pre32","pre7_4") and len(x)>30:
            cw=pd.factorize(cow_of.reindex(x.index).to_numpy())[0]; xv=x.to_numpy(float); bs=[]
            for _ in range(500):
                w=rng.poisson(1.0,cw.max()+1)[cw]; bs.append((w*xv).sum()/max(w.sum(),1))
            out[nm+"_lo"],out[nm+"_hi"]=np.percentile(bs,[2.5,97.5])
    for thr in (0.85,0.75,0.70):
        out[f"pre3_t{int(thr*100)}"]=share(Lc,-3,-1,2,thr=np.log(thr)).mean(); out[f"pre3_t{int(thr*100)}_ctrl"]=share(Lk,-3,-1,2,thr=np.log(thr)).mean()
    out["pre3_raw"]=share(Lc,-3,-1,2,"dev_raw").mean(); out["pre3_raw_ctrl"]=share(Lk,-3,-1,2,"dev_raw").mean(); xl=share(Lc,-3,-1,2,"DL"); out["pre3_DL"]=xl.mean(); out["pre3_DL_ctrl"]=share(Lk,-3,-1,2,"DL").mean(); out["pre3_DL_n"]=len(xl)
    for k in (-5,-3,-2,-1,0,1):
        out[f"D_{k}"]=Lc[Lc.k==k].D.mean(); out[f"Dc_{k}"]=Lk[Lk.k==k].D.mean(); out[f"n_{k}"]=int((Lc.k==k).sum())
    P(f"[{label}] new records {n_all}, examined {len(cases)}, evaluated {out['pre3_n']}: drop d-3..-1 {out['pre3']:.3f} ({out.get('pre3_lo',np.nan):.3f}-{out.get('pre3_hi',np.nan):.3f}) vs {out['pre3_ctrl']:.3f} | d-1 {out['d1']:.3f}/{out['d1_ctrl']:.3f} d-2 {out['d2']:.3f}/{out['d2_ctrl']:.3f} d-3 {out['d3']:.3f}/{out['d3_ctrl']:.3f} | d-3..-2 {out['pre32']:.3f}/{out['pre32_ctrl']:.3f} | d-7..-4 {out['pre7_4']:.3f}/{out['pre7_4_ctrl']:.3f} | daily-sum {out['pre3_raw']:.3f}/{out['pre3_raw_ctrl']:.3f} | DL {out['pre3_DL']:.3f}/{out['pre3_DL_ctrl']:.3f} | thr85 {out['pre3_t85']:.3f}/{out['pre3_t85_ctrl']:.3f} thr70 {out['pre3_t70']:.3f}/{out['pre3_t70_ctrl']:.3f} | D d-1 {(np.exp(out['D_-1'])-1)*100:+.1f} ({(np.exp(out['Dc_-1'])-1)*100:+.1f}) d0 {(np.exp(out['D_0'])-1)*100:+.1f} ({(np.exp(out['Dc_0'])-1)*100:+.1f})")
    return out
traj=[]; res=[]
dis=pd.read_parquet(V+"disease_cid.parquet"); dis=dis.sort_values(["cid","dnum"]); gap=dis.groupby("cid").dnum.diff(); dis["new"]=gap.isna()|(gap>14)
new=dis[dis.new].merge(cov,left_on="farm_id",right_index=True); new=new[(new.dnum>=new["min"]+10)&(new.dnum<=new["max"]-3)]
P("new disease records with the record day at least 10 days after the start and 3 days before the end of the milking records:",len(new))
res.append(study(new,"All diseases",traj,None,True))
for g,lab in (("mastitis","Udder"),("digestive","Digestive"),("metabolic","Abomasal and metabolic"),("lameness","Locomotor"),("respiratory","Respiratory"),("reproductive","Reproductive tract"),("other","Other")):
    res.append(study(new[new.grp==g],lab,traj,None,True))
for name,lab in (("e_oestrus","Oestrus recorded"),("e_mating","Insemination"),("e_immune","Vaccination"),("e_trimming","Hoof trimming"),("e_group_transformation","Pen change"),("e_first_inspection","Pregnancy check")):
    ea=load(name); e=ea.merge(cov,left_on="farm_id",right_index=True); e=e[(e.dnum>=e["min"]+10)&(e.dnum<=e["max"]-3)]
    if len(e)>300000: e=e.sample(300000,random_state=3)
    res.append(study(e,lab,traj,ea,False))
pd.DataFrame(res).to_csv(V+"tab2_validity.csv",index=False); pd.concat(traj).to_csv(V+"fig2_trajectories.csv",index=False)
# attribution of onsets to recorded events
on=F[(F.risk_onset)]
P("\n[attribution] recorded events around acute-drop onsets vs at-risk cow-days without onset")
o1=on[on.y_onset==1][["cid","dnum"]].copy(); o1["dnum"]+=1; o0=on[on.y_onset==0][["cid","dnum"]].sample(2000000,random_state=5).copy(); o0["dnum"]+=1
def near(tab,e,a,b):
    hit=np.zeros(len(tab),bool); ee=e[["cid","dnum"]].drop_duplicates().rename(columns={"dnum":"ev"})
    for k in range(a,b+1):
        t=tab.copy(); t["ev"]=t.dnum+k; hit|=(t.merge(ee,on=["cid","ev"],how="left",indicator=True)["_merge"]=="both").to_numpy()
    return hit
EVS={"Disease record (day -1 to +3)":(["DIS"],-1,3),"Oestrus or insemination (day -1 to +1)":(["e_oestrus","e_mating"],-1,1),"Pen change (day -1 to +1)":(["e_group_transformation"],-1,1),"Vaccination (day -2 to 0)":(["e_immune"],-2,0),"Hoof trimming (day -2 to 0)":(["e_trimming"],-2,0),"Dry-off (day 0 to +7)":(["e_dried"],0,7)}
a1=np.zeros(len(o1),bool); a0=np.zeros(len(o0),bool); rows=[]
for lab,(names,a,b) in EVS.items():
    e=pd.concat([dis[["cid","dnum"]] if n=="DIS" else load(n)[["cid","dnum"]] for n in names]); h1=near(o1,e,a,b); h0=near(o0,e,a,b); a1|=h1; a0|=h0
    rows.append(dict(event=lab,onsets=h1.mean(),non_onsets=h0.mean())); P(f"   {lab:40s} onsets {h1.mean():.4f} | others {h0.mean():.4f}")
rows.append(dict(event="Any of the above",onsets=a1.mean(),non_onsets=a0.mean(),n_onsets=len(o1))); P(f"   any: onsets {a1.mean():.4f} | others {a0.mean():.4f} ; n onsets {len(o1)}")
pd.DataFrame(rows).to_csv(V+"tab_attribution.csv",index=False)
P("DONE",round(time.time()-t0))
