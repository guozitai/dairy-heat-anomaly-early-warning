#!/usr/bin/env python3
"""D2. Two checks of the measurement against veterinary disease records (cases and controls as in C4, all diseases).
(a) Day deviation without the milkings that follow an interval of 11.5-14 h (Dx), and from regular-interval milkings only (D_rate):
    share of acute drops and share of records preceded by an acute drop.
(b) Records removed at cleaning for a yield of 0.5 kg or less: frequency on the days before a disease record and in controls,
    and the share of records preceded by an acute drop or by such a record."""
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"; D0="/home/guozitai/20260908_thi_alignment/derived/"
def P(*a): print(*a,flush=True)
L8=np.log(0.8)
# ---- (a) deviation without milkings after 11.5-14 h
s=pd.read_parquet(V+"sess_v3.parquet",columns=["cid","dnum","y","dt","ref","ref_y","use","use_rate"]); s=s[s.use].copy()
b=(s.dt>11.5)&(s.dt<=14); SH_B=b.mean(); P("usable milkings %d ; after an interval of 11.5-14 h: %d (%.4f) ; after any long gap: %.4f"%(len(s),b.sum(),b.mean(),(~s.use_rate).mean()))
s["e"]=np.where(s.use_rate,s.ref*s.dt,s.ref_y); allv=s.groupby(["cid","dnum"]).agg(anyb=("dt",lambda x:((x>11.5)&(x<=14)).any())) if False else None
s["b"]=b; g=s.groupby(["cid","dnum"]); hasb=g.b.any().rename("hasb"); k=s[~b].groupby(["cid","dnum"]).agg(yo=("y","sum"),ye=("e","sum"),nx=("y","size")); k["Dx"]=np.log(k.yo/k.ye).astype("float32")
DX=k[["Dx"]].join(hasb,how="outer").reset_index(); del s,g,k
F0=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","dnum","D","elig"]); F0=F0[F0.elig].merge(DX,on=["cid","dnum"],how="left").merge(pd.read_parquet(V+"cowday_extra.parquet",columns=["cid","dnum","D_rate"]),on=["cid","dnum"],how="left")
w=F0[F0.D.notna()]; P("eligible cow-days with D: %d ; D uses a milking after 11.5-14 h on %.4f of them ; these lose their value on %.4f"%(len(w),w.hasb.fillna(False).mean(),(w.Dx.isna()).mean()))
P("acute drops: D %.4f | Dx %.4f (n=%d) | D_rate %.4f (n=%d)"%((w.D<L8).mean(),(w.Dx.dropna()<L8).mean(),w.Dx.notna().sum(),(F0.D_rate.dropna()<L8).mean(),F0.D_rate.notna().sum()))
pd.DataFrame([dict(usable_milkings=int(len(F0)*0+0),share_milkings_115_14=float(SH_B),cowdays_with_D=len(w),share_cowdays_using=float(w.hasb.fillna(False).mean()),drop_D=float((w.D<L8).mean()),drop_Dx=float((w.Dx.dropna()<L8).mean()),drop_Drate=float((F0.D_rate.dropna()<L8).mean()))]).to_csv(V+"tab_boundary_check.csv",index=False)
bb=w[w.Dx.notna()]; P("agreement on cow-days with both: drop by D and by Dx %.4f of D drops ; correlation %.4f"%(((bb.D<L8)&(bb.Dx<L8)).sum()/max((bb.D<L8).sum(),1),np.corrcoef(bb.D,bb.Dx)[0,1]))
# ---- (b) low-yield records per cow-day
r=pd.read_parquet(D0+"unified_session.parquet",columns=["farm_id","bullNumber","milkDate","milkYield"]); r["y"]=pd.to_numeric(r.milkYield,errors="coerce"); r=r[(r.y<=0.5)|r.y.isna()]
r["key"]=r.farm_id.astype(str)+"_"+r.bullNumber.astype(str); ct=pd.read_parquet(V+"cowtab.parquet"); r=r.merge(ct[["cid","key"]],on="key",how="inner"); r["dnum"]=(pd.to_datetime(r.milkDate).dt.normalize()-pd.Timestamp("2022-01-01")).dt.days
LOW=r.groupby(["cid","dnum"]).size().rename("nlow").reset_index(); P("cow-days with a record of 0.5 kg or less: %d (records matched to study cows: %d)"%(len(LOW),len(r))); del r
el=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","dnum","elig"]); el=el[el.elig].merge(LOW,on=["cid","dnum"],how="left"); P("share of eligible cow-days with such a record: %.4f"%(el.nlow.fillna(0)>0).mean()); del el
# ---- cases and controls exactly as in C4 (source patched: extra columns, extra outputs, all diseases only)
src=open(V+"c4_validity.py").read()
a='X=F[["cid","dnum","D","dev_raw","DL"]]'; assert src.count(a)==1
src=src.replace(a,'F=F.merge(F0[["cid","dnum","Dx","D_rate"]],on=["cid","dnum"],how="left"); X=F[["cid","dnum","D","dev_raw","DL","Dx","D_rate"]]')
a='    P(f"[{label}] new records {n_all}'; assert src.count(a)==1
extra='''    if label=="All diseases":
        for col in ("Dx","D_rate"):
            xs=share(Lc,-3,-1,2,col); ys=share(Lk,-3,-1,2,col); x2=share(Lc,-3,-2,1,col); y2=share(Lk,-3,-2,1,col)
            P("  [%s] records with >=2 values on days -3..-1: %d ; acute drop %.4f vs controls %.4f ; days -3..-2: %.4f vs %.4f"%(col,len(xs),xs.mean(),ys.mean(),x2.mean(),y2.mean()))
        def lowshare(G):
            t=pd.concat([G[["case","cid","ev"]].assign(dnum=G.ev+k,k=k) for k in (-3,-2,-1,0)]).merge(LOW,on=["cid","dnum"],how="left"); t["low"]=t.nlow.fillna(0)>0
            return t[t.k<0].groupby("case").low.any(), t[t.k==0].groupby("case").low.any(), t[t.k<0].low.mean()
        la,lb,lr=lowshare(cases); ka,kb,kr=lowshare(ctl)
        P("  low-yield records: cases with one on days -3..-1 %.4f vs controls %.4f ; on the record day %.4f vs %.4f ; per cow-day of days -3..-1 %.4f vs %.4f"%(la.mean(),ka.mean(),lb.mean(),kb.mean(),lr,kr))
        xs=share(Lc,-3,-1,2); ys=share(Lk,-3,-1,2)
        P("  evaluated records %d: acute drop %.4f vs %.4f ; acute drop or low-yield record on days -3..-1 %.4f vs %.4f ; low-yield record among records without an acute drop %.4f vs %.4f"%(len(xs),xs.mean(),ys.mean(),(xs|la.reindex(xs.index).fillna(False)).mean(),(ys|ka.reindex(ys.index).fillna(False)).mean(),la.reindex(xs.index).fillna(False)[~xs].mean(),ka.reindex(ys.index).fillna(False)[~ys].mean()))
        le=la.reindex(xs.index).fillna(False); ke=ka.reindex(ys.index).fillna(False)
        P("  evaluated records: low-yield record on days -3..-1 %.4f vs %.4f ; among records with an acute drop %.4f vs %.4f"%(le.mean(),ke.mean(),le[xs].mean(),ke[ys].mean()))
        xd=share(Lc,-3,-1,2,"Dx"); yd=share(Lk,-3,-1,2,"Dx"); xr=share(Lc,-3,-1,2,"D_rate"); yr=share(Lk,-3,-1,2,"D_rate")
        pd.DataFrame([dict(measure="D",n=len(xs),records=xs.mean(),controls=ys.mean(),pre32=share(Lc,-3,-2,1).mean(),pre32_ctrl=share(Lk,-3,-2,1).mean()),
                      dict(measure="Dx",n=len(xd),records=xd.mean(),controls=yd.mean(),pre32=share(Lc,-3,-2,1,"Dx").mean(),pre32_ctrl=share(Lk,-3,-2,1,"Dx").mean()),
                      dict(measure="D_rate",n=len(xr),records=xr.mean(),controls=yr.mean(),pre32=share(Lc,-3,-2,1,"D_rate").mean(),pre32_ctrl=share(Lk,-3,-2,1,"D_rate").mean()),
                      dict(measure="low_record",n=len(xs),records=le.mean(),controls=ke.mean()),
                      dict(measure="D_or_low_record",n=len(xs),records=(xs|le).mean(),controls=(ys|ke).mean())]).to_csv(V+"tab_validity_checks.csv",index=False)
'''
src=src.replace(a,extra+a)
a='res.append(study(new,"All diseases",traj,None,True))'; i=src.index(a)+len(a); src=src[:i]+'\nP("DONE",round(time.time()-t0))\n'
exec(compile(src,"c4_patched","exec"))
