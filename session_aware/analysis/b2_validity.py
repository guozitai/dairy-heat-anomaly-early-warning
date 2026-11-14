#!/usr/bin/env python3
"""B2. Correspondence between acute drops (session-aware) and recorded veterinary / management events.
Trajectories of the day deviation D around event days with same-farm same-day control cows; share of cases with >=1 drop day before the record
(bootstrap CI over cases); attribution of drop onsets to recorded events. Outputs CSVs for Table 2 / Fig 2."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time()
V="/home/guozitai/20261111_sat_rescue/v2/"; E="/home/guozitai/20261014_event_layer/parquet/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",250); L=np.log(0.8); rng=np.random.default_rng(11)
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","D","dev_raw","elig","risk_onset","y_onset","DIM","par"]); F=F[F.elig&F.D.notna()].copy(); F["pc"]=F.par.clip(1,3).astype("int8"); F["DIM"]=F.DIM.astype("float32")
F["drop"]=(F.D<L).astype("int8"); F["drop_raw"]=np.where(F.dev_raw.notna(),(F.dev_raw<L).astype(float),np.nan)
X=F[["cid","dnum","D","drop","drop_raw"]]; ct=pd.read_parquet(V+"cowtab.parquet")
cov=F.groupby("farm_id").dnum.agg(["min","max"]); fa=F[["cid","farm_id"]].drop_duplicates()
def load(name):
    e=pd.read_parquet(E+name+".parquet",columns=["farm_id","cow_number","event_time"]); e["farm_id"]=pd.to_numeric(e.farm_id,errors="coerce"); e=e.dropna(subset=["farm_id"]); e["farm_id"]=e.farm_id.astype(int); e["cow"]=e.cow_number.astype(str)
    e["dnum"]=(pd.to_datetime(e.event_time,errors="coerce").dt.normalize()-pd.Timestamp("2022-01-01")).dt.days; e=e.dropna(subset=["dnum"]).merge(ct[["cid","farm_id","cow"]],on=["farm_id","cow"]); e["dnum"]=e.dnum.astype(int)
    return e[["cid","farm_id","dnum"]].drop_duplicates()
OFF=list(range(-10,8))
def table(cases):
    parts=[]
    for k in OFF:
        t=cases[["case","cid","ev"]].copy(); t["k"]=k; t["dnum"]=t.ev+k; parts.append(t)
    return pd.concat(parts,ignore_index=True).merge(X,on=["cid","dnum"],how="inner")
def study(cases,label,traj_out,ctrl_n=3):
    cases=cases.rename(columns={"dnum":"ev"}).drop_duplicates(["cid","ev"]).reset_index(drop=True); cases["case"]=np.arange(len(cases))
    # strata of each case from its latest row on days -3..-1: parity class and 30-day DIM class at the record day
    parts=[]
    for k in (1,2,3):
        t=cases[["case","cid","ev"]].copy(); t["dnum"]=t.ev-k; t=t.merge(F[["cid","dnum","pc","DIM"]],on=["cid","dnum"],how="inner"); t["dimr"]=t.DIM+k; t["k"]=k; parts.append(t)
    st=pd.concat(parts).sort_values(["case","k"]).drop_duplicates("case"); st["dc"]=(st.dimr//30).astype(int); cases=cases.merge(st[["case","pc","dc"]],on="case",how="inner")
    days=cases[["farm_id","ev","pc","dc"]].drop_duplicates(); P0=F[["cid","farm_id","dnum","pc","DIM"]].copy(); P0["dc"]=(P0.DIM//30).astype(int)
    pool=P0.merge(days,left_on=["farm_id","dnum","pc","dc"],right_on=["farm_id","ev","pc","dc"])[["cid","farm_id","ev","pc","dc"]]
    pool=pool.merge(cases[["cid","ev"]].assign(is_case=1),on=["cid","ev"],how="left"); pool=pool[pool.is_case.isna()]
    pool["u"]=rng.random(len(pool)); cnt=cases.groupby(["farm_id","ev","pc","dc"]).size().rename("nc").reset_index(); pool=pool.merge(cnt,on=["farm_id","ev","pc","dc"]); pool["r"]=pool.groupby(["farm_id","ev","pc","dc"]).u.rank(method="first")
    ctl=pool[pool.r<=pool.nc*ctrl_n][["cid","farm_id","ev"]].reset_index(drop=True); ctl["case"]=np.arange(len(ctl))
    Lc=table(cases); Lk=table(ctl)
    tr=Lc.groupby("k").D.agg(["mean","sem","size"]).reset_index().assign(group=label,arm="case"); tk=Lk.groupby("k").D.agg(["mean","sem","size"]).reset_index().assign(group=label,arm="control"); traj_out.append(pd.concat([tr,tk]))
    def share(Lg,a,b,need,col="drop"):
        w=Lg[(Lg.k>=a)&(Lg.k<=b)].dropna(subset=[col]).groupby("case")[col].agg(["size","sum"]); w=w[w["size"]>=need]; return (w["sum"]>0).to_numpy()
    out=dict(group=label,records=len(cases))
    for nm,(a,b,need) in {"pre3":(-3,-1,2),"pre7_4":(-7,-4,3),"day0":(0,0,1),"pre1":(-1,-1,1),"pre2":(-2,-2,1),"pre3only":(-3,-3,1)}.items():
        x=share(Lc,a,b,need); y=share(Lk,a,b,need); out[nm+"_n"]=len(x); out[nm]=x.mean() if len(x) else np.nan; out[nm+"_ctrl"]=y.mean() if len(y) else np.nan
        if nm=="pre3" and len(x)>30:
            bs=[x[rng.integers(0,len(x),len(x))].mean() for _ in range(500)]; out["pre3_lo"],out["pre3_hi"]=np.percentile(bs,[2.5,97.5])
            xr=share(Lc,a,b,need,"drop_raw"); yr=share(Lk,a,b,need,"drop_raw"); out["pre3_raw"]=xr.mean(); out["pre3_raw_ctrl"]=yr.mean()
    out["D_m1"]=Lc[Lc.k==-1].D.mean(); out["D_0"]=Lc[Lc.k==0].D.mean()
    P(f"[{label}] records {len(cases)}, evaluable {out['pre3_n']}: drop in d-3..-1 {out['pre3']:.3f} ({out.get('pre3_lo',np.nan):.3f}-{out.get('pre3_hi',np.nan):.3f}) vs control {out['pre3_ctrl']:.3f} | raw-sum label {out.get('pre3_raw',np.nan):.3f} vs {out.get('pre3_raw_ctrl',np.nan):.3f} | mean D d-1 {out['D_m1']*100:+.1f}, d0 {out['D_0']*100:+.1f}")
    return out
traj=[]; res=[]
dis=pd.read_parquet(V+"disease_cid.parquet"); dis=dis.merge(cov,left_on="farm_id",right_index=True); dis=dis[(dis.dnum>=dis["min"]+10)&(dis.dnum<=dis["max"]-7)]
dis=dis.sort_values(["cid","dnum"]); gap=dis.groupby("cid").dnum.diff(); new=dis[gap.isna()|(gap>14)]
P("new disease records (no record in previous 14 d) within milking-data coverage:",len(new))
res.append(study(new,"All diseases",traj))
for g,lab in (("mastitis","Mastitis"),("digestive","Digestive"),("lameness","Lameness"),("respiratory","Respiratory"),("metabolic","Metabolic"),("reproductive","Reproductive tract")):
    res.append(study(new[new.grp==g],lab,traj))
for name,lab in (("e_oestrus","Oestrus recorded"),("e_mating","Insemination"),("e_immune","Vaccination"),("e_trimming","Hoof trimming"),("e_group_transformation","Pen change"),("e_first_inspection","Pregnancy check")):
    e=load(name); e=e.merge(cov,left_on="farm_id",right_index=True); e=e[(e.dnum>=e["min"]+10)&(e.dnum<=e["max"]-7)]
    if len(e)>300000: e=e.sample(300000,random_state=3)
    res.append(study(e,lab,traj))
pd.DataFrame(res).to_csv(V+"tab2_validity.csv",index=False); pd.concat(traj).to_csv(V+"fig2_trajectories.csv",index=False)
# attribution of onsets
on=F[(F.risk_onset)]; 
P("\n[attribution] recorded events around acute-drop onsets (day t+1 of at-risk rows) vs at-risk cow-days without onset")
o1=on[on.y_onset==1][["cid","dnum"]].copy(); o1["dnum"]+=1; o0=on[on.y_onset==0][["cid","dnum"]].sample(2000000,random_state=5).copy(); o0["dnum"]+=1
def near(tab,e,a,b):
    hit=np.zeros(len(tab),bool); ee=e[["cid","dnum"]].drop_duplicates().rename(columns={"dnum":"ev"})
    for k in range(a,b+1):
        t=tab.copy(); t["ev"]=t.dnum+k; hit|=(t.merge(ee,on=["cid","ev"],how="left",indicator=True)["_merge"]=="both").to_numpy()
    return hit
EVS={"Disease record (d-1..+3)":(["DIS"],-1,3),"Oestrus or insemination (d-1..+1)":(["e_oestrus","e_mating"],-1,1),"Pen change (d-1..+1)":(["e_group_transformation"],-1,1),"Vaccination (d-2..0)":(["e_immune"],-2,0),"Hoof trimming (d-2..0)":(["e_trimming"],-2,0),"Dry-off within 7 d":(["e_dried"],0,7)}
a1=np.zeros(len(o1),bool); a0=np.zeros(len(o0),bool); rows=[]
for lab,(names,a,b) in EVS.items():
    e=pd.concat([dis[["cid","dnum"]] if n=="DIS" else load(n)[["cid","dnum"]] for n in names]); h1=near(o1,e,a,b); h0=near(o0,e,a,b); a1|=h1; a0|=h0
    rows.append(dict(event=lab,onsets=h1.mean(),non_onsets=h0.mean())); P(f"   {lab:36s} onsets {h1.mean():.4f} | others {h0.mean():.4f}")
rows.append(dict(event="Any of the above",onsets=a1.mean(),non_onsets=a0.mean())); P(f"   any: onsets {a1.mean():.4f} | others {a0.mean():.4f} ; n onsets {len(o1)}")
pd.DataFrame(rows).to_csv(V+"tab_attribution.csv",index=False)
P("DONE",round(time.time()-t0))
