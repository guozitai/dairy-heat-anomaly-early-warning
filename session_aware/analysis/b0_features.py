#!/usr/bin/env python3
"""B0. Feature table at information-day granularity (data up to and including day t), outcomes, exposure.
Outcomes: (1) acute-drop onset on day t+1 among cows observed and not in drop on day t; (2) new veterinary disease record on days t+1..t+3."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time()
V="/home/guozitai/20261111_sat_rescue/v2/"; E="/home/guozitai/20261014_event_layer/parquet/"
def P(*a): print(*a,flush=True)
L=np.log(0.8)
d=pd.read_parquet(V+"cowday_v2.parquet")
d=d[d.farm_id.isin([3,5,6,13,14,15,16,17])].sort_values(["cid","dnum"]).reset_index(drop=True)
c=d.cid.to_numpy(); dn=d.dnum.to_numpy(); N=len(d)
def lag(col,k):
    """value of col exactly k days earlier (same cow), NaN if no record that day"""
    x=d[col].to_numpy(dtype="float32"); out=np.full(N,np.nan,dtype="float32")
    for j in range(1,k+1):
        ok=(c[j:]==c[:-j])&((dn[j:]-dn[:-j])==k); idx=np.flatnonzero(ok)+j; out[idx]=x[idx-j]
    return out
def lead(col,k):
    x=d[col].to_numpy(dtype="float32"); out=np.full(N,np.nan,dtype="float32")
    for j in range(1,k+1):
        ok=(c[j:]==c[:-j])&((dn[j:]-dn[:-j])==k); idx=np.flatnonzero(ok); out[idx]=x[idx+j]
    return out
for k in range(1,7): d[f"D_l{k}"]=lag("D",k)
M=d[["D"]+[f"D_l{k}" for k in range(1,7)]].to_numpy(dtype="float32")
d["ma3"]=np.nanmean(M[:,:3],axis=1); d["min3"]=np.nanmin(M[:,:3],axis=1); d["ma7"]=np.nanmean(M,axis=1); d["min7"]=np.nanmin(M,axis=1); d["sd7"]=np.nanstd(M,axis=1); d["slope"]=d.D-d.D_l2
d["drops7"]=np.nansum((M<L),axis=1).astype("float32")
dr=(d.D<L).astype("float32").where(d.D.notna()); d["_dr"]=dr
acc=np.zeros(N,dtype="float32")
for k in range(0,30):
    v=d._dr.to_numpy(dtype="float32") if k==0 else lag("_dr",k); acc+=np.nan_to_num(v)
d["drops30"]=acc
d["loss_kg"]=(d.base_my24-d.my24).astype("float32"); d["lbase"]=np.log(d.base_my24).astype("float32")
# sensors: deviation of conductivity / flow from own trailing median (days t-8..t-2, >=4)
def trailing_med(col,minn=4):
    x=d[col].to_numpy(dtype="float32"); Mx=np.full((8,N),np.nan,dtype="float32")
    for k in range(1,9):
        ok=(c[k:]==c[:-k]); dd=dn[k:]-dn[:-k]; ok&=(dd>=2)&(dd<=8); Mx[k-1,k:]=np.where(ok,x[:-k],np.nan)
    cnt=np.sum(~np.isnan(Mx),axis=0); med=np.nanmedian(Mx,axis=0); med[cnt<minn]=np.nan; return med
d["cond"]=d.cond.where(d.cond>0); d["flow"]=d.flow.where(d.flow>0)
d["cond_dev"]=np.log(d.cond/trailing_med("cond")).astype("float32"); d["condmax_dev"]=np.log(d.cond_max.where(d.cond_max>0)/trailing_med("cond_max")).astype("float32"); d["flow_dev"]=np.log(d.flow/trailing_med("flow")).astype("float32")
d["cond_dev_l1"]=lag("cond_dev",1)
# herd context (same day): mean deviation and share in drop; sessions relative to herd median
h=d[d.D.notna()].groupby(["farm_id","dnum"]).agg(H=("D","mean"),Hmed=("D","median"),hdrop=("_dr","mean"),hn=("D","size")).reset_index()
d=d.merge(h,on=["farm_id","dnum"],how="left"); d.loc[d.hn<100,["H","Hmed","hdrop"]]=np.nan
md=d.groupby(["farm_id","dnum"]).n_sess.transform("median"); d["sess_gap"]=(md-d.n_sess).astype("float32")
d["pg"]=d.par.clip(1,4).astype("float32")
c=d.cid.to_numpy(); dn=d.dnum.to_numpy()
# outcome 1: onset tomorrow
d["D_next"]=lead("D",1); d["D_next2"]=lead("D",2)
d["risk_onset"]=d.D.notna()&(d.D>=L)&d.D_next.notna()
d["y_onset"]=(d.risk_onset&(d.D_next<L)).astype("int8"); d["y_onset_sus"]=(d.y_onset.eq(1)&(d.D_next2<np.log(0.9))).astype("int8")
# outcome 2: veterinary disease records
ct=pd.read_parquet(V+"cowtab.parquet")
dis=pd.read_parquet(E+"e_disease.parquet",columns=["farm_id","cow_number","disease_name","event_time"])
dis["farm_id"]=pd.to_numeric(dis.farm_id,errors="coerce"); dis=dis.dropna(subset=["farm_id"]); dis["farm_id"]=dis.farm_id.astype(int); dis["cow"]=dis.cow_number.astype(str)
dis["day"]=pd.to_datetime(dis.event_time,errors="coerce").dt.normalize(); dis=dis.dropna(subset=["day"]).merge(ct[["cid","farm_id","cow"]],on=["farm_id","cow"],how="inner")
dis["dnum"]=(dis.day-pd.Timestamp("2022-01-01")).dt.days.astype("int32")
def grp(s):
    s=str(s)
    for g,keys in (("mastitis",["乳房","血乳","乳头","乳区"]),("digestive",["腹泻","消化","前胃","瘤胃","肠","积食","胃炎","臌气"]),("respiratory",["肺","感冒","支气管","呼吸"]),("lameness",["蹄","关节","跛","肢"]),("metabolic",["酮","瘫","真胃","脂肪肝","低钙"]),("reproductive",["子宫","胎衣","产道","阴道","流产"])):
        if any(k in s for k in keys): return g
    return "other"
dis["grp"]=dis.disease_name.map(grp)
dis=dis.sort_values(["cid","dnum"]); dis.to_parquet(V+"disease_cid.parquet",index=False)
P("disease records linked to cows with milking data:",len(dis)); P(dis.grp.value_counts().to_string())
def next_prev(tab,suf):
    a=d[["cid","dnum"]].copy(); a["idx"]=np.arange(N); a=a.sort_values("dnum")
    b=tab[["cid","dnum"]].drop_duplicates().sort_values("dnum").rename(columns={"dnum":"ev"})
    f=pd.merge_asof(a,b,left_on="dnum",right_on="ev",by="cid",direction="forward",allow_exact_matches=False).set_index("idx").sort_index()
    p=pd.merge_asof(a,b,left_on="dnum",right_on="ev",by="cid",direction="backward",allow_exact_matches=True).set_index("idx").sort_index()
    d["dn_"+suf]=(f.ev-f.dnum).to_numpy(dtype="float32"); d["dp_"+suf]=(p.dnum-p.ev).to_numpy(dtype="float32")
next_prev(dis,"dis")
for g in ("mastitis","digestive","respiratory","lameness","metabolic","reproductive"): next_prev(dis[dis.grp==g],g[:4])
d["risk_dis"]=d.D.notna()&~(d.dp_dis==0)&~((d.dp_dis>=1)&(d.dp_dis<=14))
d["y_dis"]=(d.risk_dis&(d.dn_dis<=3)).astype("int8")
d["since_dis"]=d.dp_dis.clip(upper=400)
# exposure at farm location
LOC={3:"A",6:"B",13:"C",14:"C",17:"C",15:"D",16:"D",5:"E"}; d["loc"]=d.farm_id.map(LOC)
T=pd.read_csv(V+"era5_farm_daily.csv",parse_dates=["date"]).sort_values(["loc","date"]); g=T.groupby("loc")
T["tmax_l1"]=g.tmax.shift(1); T["tmax_l2"]=g.tmax.shift(2); T["tmax_f1"]=g.tmax.shift(-1); T["tmax3"]=g.tmax.transform(lambda s:s.rolling(3,min_periods=3).mean())
T["tmax_ref"]=g.tmax.transform(lambda s:s.shift(2).rolling(7,min_periods=5).mean()); T["dthi"]=T.tmax-T.tmax_ref
d=d.merge(T[["loc","date","tmax","tmean","nmin","h72","h79","tmax_l1","tmax_l2","tmax_f1","tmax3","dthi"]],on=["loc","date"],how="left")
d["elig"]=d.DIM.notna()&(d.DIM>=10)&(d.DIM<=400)&d.par.notna()
keep=[x for x in d.columns if x not in ("_dr","key","yield_suspicious","qc_flag","daily_n_sessions")]
d[keep].to_parquet(V+"feat_v2.parquet",index=False)
e=d[d.elig]
P("\ncow-days",N,"eligible (DIM 10-400)",len(e),"with D",int(e.D.notna().sum()),round(time.time()-t0))
P("onset risk set",int(e.risk_onset.sum()),"onset rate",round(float(e.y_onset[e.risk_onset].mean()),4),"sustained",round(float(e.y_onset_sus[e.risk_onset].mean()),4))
P("disease risk set",int(e.risk_dis.sum()),"P(record in t+1..t+3)",round(float(e.y_dis[e.risk_dis].mean()),5))
P(e.groupby("farm_id").agg(cowdays=("D","size"),cows=("cid","nunique"),d0=("date","min"),d1=("date","max"),withD=("D","count"),drop=("D",lambda s:(s<L).mean()),thi=("tmax","count")).to_string())
P("DONE",round(time.time()-t0))
