#!/usr/bin/env python3
"""C3. Flow of veterinary disease records: from all records of the study cows to the records evaluated.
Status of the cow on the record day from the lactation number in the record, calving records and dry-off records.
Writes disease_flow_records.parquet (one row per new record) and summary CSVs."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; E="/home/guozitai/20261014_event_layer/parquet/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",260); pd.set_option("display.max_columns",40)
LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}; D0=pd.Timestamp("2022-01-01")
ct=pd.read_parquet(V+"cowtab.parquet"); ct=ct[ct.farm_id.isin(LAB)]
F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","D","elig","risk_dis","DIM","par","n_sess","calvingDate"])
cov=F[F.elig].groupby("farm_id").dnum.agg(["min","max"]); study=F[F.elig][["cid"]].drop_duplicates()
def ev(name,cols):
    e=pd.read_parquet(E+name+".parquet",columns=["farm_id","cow_number","event_time"]+cols); e["farm_id"]=pd.to_numeric(e.farm_id,errors="coerce"); e=e.dropna(subset=["farm_id"]); e["farm_id"]=e.farm_id.astype(int); e["cow"]=e.cow_number.astype(str)
    e["day"]=pd.to_datetime(e.event_time,errors="coerce").dt.normalize(); e=e.dropna(subset=["day"]).merge(ct[["cid","farm_id","cow"]],on=["farm_id","cow"]); e["dnum"]=(e.day-D0).dt.days.astype("int32"); return e
d=ev("e_disease",["disease_name","pro_con","cowshed_name","create_time","work_name","lact","healing_state"])
G=pd.read_parquet(V+"disease_cid.parquet",columns=["cid","dnum","disease_name","grp"]).drop_duplicates(["cid","dnum","disease_name"]); d=d.merge(G,on=["cid","dnum","disease_name"],how="left"); d["grp"]=d.grp.fillna("other")
d=d.sort_values(["cid","dnum"]).reset_index(drop=True); gap=d.groupby("cid").dnum.diff(); d["new"]=gap.isna()|(gap>14)
d=d.merge(study,on="cid").merge(cov,left_on="farm_id",right_index=True); d=d[(d.dnum>=d["min"])&(d.dnum<=d["max"])].copy()
P("disease records of study cows within the milking-record period of the farm:",len(d),"| new (no record of the cow in the preceding 14 d):",int(d.new.sum()))
# calving and dry-off history
cv=ev("e_calving",[])[["cid","dnum"]]; fc=F.dropna(subset=["calvingDate"])[["cid","calvingDate"]].drop_duplicates(); fc["dnum"]=(pd.to_datetime(fc.calvingDate)-D0).dt.days.astype("int32")
cv=pd.concat([cv,fc[["cid","dnum"]]]).drop_duplicates().sort_values("dnum").rename(columns={"dnum":"calv"}); dr=ev("e_dried",[])[["cid","dnum"]].drop_duplicates().sort_values("dnum").rename(columns={"dnum":"dry"})
n=d[d.new].copy().sort_values("dnum"); n=pd.merge_asof(n,cv,left_on="dnum",right_on="calv",by="cid",direction="backward"); n=pd.merge_asof(n,dr,left_on="dnum",right_on="dry",by="cid",direction="backward")
n["dim"]=n.dnum-n.calv; lact=pd.to_numeric(n.lact,errors="coerce")
isdry=n.dry.notna()&(n.calv.isna()|(n.dry>=n.calv))
n["status"]=np.select([n.calv.isna(),isdry,n.dim<10,n.dim<=400],["before first calving","dry period","first 9 days after calving","10 to 400 days in milk"],"more than 400 days in milk")
P("lactation number in the record, by status:"); P(pd.crosstab(n.status,lact.clip(upper=3)).to_string())
# milking data on the three preceding days
n=n.reset_index(drop=True); n["rec"]=np.arange(len(n)); parts=[]
for k in (1,2,3):
    t=n[["rec","cid","dnum"]].copy(); t["dnum"]=t.dnum-k; parts.append(t)
Lk=pd.concat(parts).merge(F[["cid","dnum","D","elig","risk_dis","n_sess"]],on=["cid","dnum"],how="left"); Lk["De"]=Lk.D.where(Lk.elig==True)
g=Lk.groupby("rec").agg(milked=("n_sess",lambda x:x.notna().any()),elig=("elig",lambda x:(x==True).any()),withD=("De",lambda x:x.notna().any()),nD=("De","count"),risk=("risk_dis",lambda x:((x==True)).any()))
Lk["rk"]=(Lk.risk_dis==True)&(Lk.elig==True); g["risk"]=Lk.groupby("rec").rk.any()
n=n.join(g,on="rec"); n["late"]=n.dnum>=n["min"]+3; n["year"]=(D0+pd.to_timedelta(n.dnum,unit="D")).dt.year; n["farm"]=n.farm_id.map(LAB)
# days since the last recorded milking for records without milking in the three preceding days
last=F[["cid","dnum"]].sort_values("dnum").rename(columns={"dnum":"lastmilk"}); n=pd.merge_asof(n.sort_values("dnum"),last,left_on="dnum",right_on="lastmilk",by="cid",direction="backward",allow_exact_matches=False); n["since_milk"]=n.dnum-n.lastmilk
n["hosp"]=n.cowshed_name.astype(str).str.contains("病"); n["treated"]=n.pro_con.notna()&(n.pro_con.astype(str).str.strip()!="")
ctm=pd.to_datetime(n.create_time,errors="coerce"); n["entry_lag"]=(ctm.dt.normalize()-n.day).dt.days; n["live"]=ctm>pd.Timestamp("2022-09-07")
n.drop(columns=["min","max"]).to_parquet(V+"disease_flow_records.parquet",index=False)
def flow(x):
    tg=x[x.status=="10 to 400 days in milk"]; return pd.Series({"new records":len(x),"before first calving":(x.status=="before first calving").sum(),"dry period":(x.status=="dry period").sum(),"first 9 days after calving":(x.status=="first 9 days after calving").sum(),"more than 400 days in milk":(x.status=="more than 400 days in milk").sum(),
        "10 to 400 days in milk":len(tg),"  with a recorded milking on days -3..-1":int(tg.milked.sum()),"  with a day deviation on days -3..-1":int(tg.withD.sum()),"  in the Task 2 risk set on at least one of these days":int(tg.risk.sum()),"  with at least two day deviations":int((tg.nD>=2).sum()),
        "outside 10-400 DIM but in risk set":int(x[x.status!="10 to 400 days in milk"].risk.sum()),"all in risk set":int(x.risk.sum()),"all with >=2 deviations":int((x.nD>=2).sum())})
FL=pd.concat([flow(n).to_frame("All")]+[flow(n[n.grp==g]).to_frame(g) for g in ("mastitis","digestive","lameness","metabolic","respiratory","reproductive","other")],axis=1); P("\n== flow of new records =="); P(FL.to_string()); FL.to_csv(V+"tab_disease_flow.csv")
tg=n[n.status=="10 to 400 days in milk"]; nm=tg[~tg.milked]
P("\nrecords at 10-400 DIM without a recorded milking on days -3..-1: %d ; in a hospital pen at the record %.3f ; days since last recorded milking: median %.0f (IQR %.0f-%.0f) ; never milked before %.3f"%(len(nm),nm.hosp.mean(),nm.since_milk.median(),nm.since_milk.quantile(.25),nm.since_milk.quantile(.75),nm.since_milk.isna().mean()))
P("   pens of these records (top 12):",nm.cowshed_name.value_counts().head(12).to_dict())
P("   since_milk distribution: 4-7 d %.3f | 8-30 d %.3f | >30 d %.3f | none %.3f"%(nm.since_milk.between(4,7).mean(),nm.since_milk.between(8,30).mean(),(nm.since_milk>30).mean(),nm.since_milk.isna().mean()))
P("target records (10-400 DIM): %d ; in risk set %d (%.3f) ; by class share in risk set:"%(len(tg),tg.risk.sum(),tg.risk.mean()),tg.groupby("grp").risk.mean().round(3).to_dict())
ev_=n[n.risk]; P("\nrecords in the risk set: %d ; treatment prescription recorded %.3f ; healing state recorded %.3f ; distinct veterinarians %d ; hospital pen at record %.3f"%(len(ev_),ev_.treated.mean(),ev_.healing_state.notna().mean(),ev_.work_name.nunique(),ev_.hosp.mean()))
lv=ev_[ev_.live]; P("   entered after the system went live (n=%d): entry on the record day %.3f ; within 1 day %.3f ; within 3 days %.3f"%(len(lv),(lv.entry_lag==0).mean(),lv.entry_lag.between(0,1).mean(),lv.entry_lag.between(0,3).mean()))
P("   treated share by class:",ev_.groupby("grp").treated.mean().round(3).to_dict())
# denominators by farm-year for the test years
FY=n[n.year.isin([2024,2025])].groupby(["farm","year"]).apply(lambda x:pd.Series(dict(new=len(x),target=(x.status=="10 to 400 days in milk").sum(),target_milked=((x.status=="10 to 400 days in milk")&x.milked).sum(),risk=x.risk.sum(),target_risk=((x.status=="10 to 400 days in milk")&x.risk).sum()))).reset_index(); P("\nby farm-year (test years):"); P(FY.to_string()); FY.to_csv(V+"tab_disease_flow_farm_year.csv",index=False)
# diagnoses within classes (records in the risk set)
DN=ev_.groupby(["grp","disease_name"]).size().rename("n").reset_index().sort_values(["grp","n"],ascending=[True,False]); DN.to_csv(V+"tab_diagnoses.csv",index=False); P("\ndiagnoses by class (records in the risk set):")
for g_,x in DN.groupby("grp"): P("  ",g_,int(x.n.sum()),dict(zip(x.disease_name.head(8),x.n.head(8))))
P("DONE",round(time.time()-t0))
