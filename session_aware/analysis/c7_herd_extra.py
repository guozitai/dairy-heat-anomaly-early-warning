#!/usr/bin/env python3
"""C7. Herd-level response, additional analyses: per-farm slopes at three thresholds with t-based intervals (clusters = farm-months),
tests of subgroup differences (paired differences of subgroup herd deviations), share of cows below 85/80/75/70% of expected yield,
numbers of dates and hot spells behind the highest THI classes."""
import pandas as pd, numpy as np, time, warnings
from scipy import stats
warnings.filterwarnings("ignore"); t0=time.time(); V="/home/guozitai/20261111_sat_rescue/v2/"
def P(*a): print(*a,flush=True)
pd.set_option("display.width",260); LAB={6:"Farm 1",17:"Farm 2",16:"Farm 3",13:"Farm 4",14:"Farm 5",15:"Farm 6",5:"Farm 7",3:"Farm 8"}
fd=pd.read_parquet(V+"farmday_v2.parquet"); warm=fd[fd.m.isin([5,6,7,8,9])].copy()
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs,cl="fym"):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]; sc=pd.DataFrame(S).groupby(z[cl].values).sum().values
    return b,np.sqrt(np.diag(XtXi@(sc.T@sc)@XtXi)),len(z),z[cl].nunique()
rows=[]
for f,s in warm.groupby("farm"):
    for tau in (72,76,79):
        s=s.copy(); s["h"]=np.maximum(s.tmax2p-tau,0); r=dict(farm=f,tau=tau,days=len(s),days_above=int((s.tmax2p>tau).sum()))
        for y in ("H","drop","onset"):
            b,se,n,G=fe(s,y,["h"]); tc=stats.t.ppf(0.975,G-1); r.update({y:b[0]*100,y+"_se":se[0]*100,y+"_lo":(b[0]-tc*se[0])*100,y+"_hi":(b[0]+tc*se[0])*100,"clusters":G,y+"_p":2*stats.t.sf(abs(b[0]/se[0]),G-1)})
        rows.append(r)
PF=pd.DataFrame(rows); PF.to_csv(V+"tab3_per_farm_all.csv",index=False); P(PF.round(3).to_string())
for tau in (72,76,79):
    z=PF[PF.tau==tau]; P(f"tau={tau}: herd deviation slope negative on {(z.H<0).sum()}/8 farms, t-interval excluding zero on {((z.H_hi<0)|(z.H_lo>0)).sum()}; drop-share slope positive on {(z['drop']>0).sum()}/8 (interval excl. zero {((z.drop_lo>0)|(z.drop_hi<0)).sum()}); onset slope positive on {(z.onset>0).sum()}/8 (interval excl. zero {((z.onset_lo>0)|(z.onset_hi<0)).sum()})")
# dates and spells behind the highest classes (previous-day THI)
for lo,hi in ((79,81),(81,90)):
    z=warm[(warm.tmax_l1>lo)&(warm.tmax_l1<=hi)]; d=np.sort(z.date.unique()); sp=1+int((np.diff(d).astype("timedelta64[D]").astype(int)>1).sum()) if len(d) else 0
    P(f"previous-day THI in ({lo},{hi}]: {len(z)} farm-days on {len(d)} dates in {sp} spells; farms {z.farm.nunique()}; by year {z.date.dt.year.value_counts().sort_index().to_dict()}")
# subgroups
L=np.log(0.8); F=pd.read_parquet(V+"feat_v2.parquet",columns=["cid","farm_id","dnum","date","loc","D","elig","DIM","par","base_my24"]); F=F[F.elig&F["loc"].notna()&F.D.notna()].copy()
F["par2"]=np.where(F.par>=2,"multiparous","primiparous"); F["stage"]=pd.cut(F.DIM,[9,100,200,400],labels=["DIM 10-100","DIM 101-200","DIM 201-400"]).astype(str); med=F.groupby(["farm_id","dnum"]).base_my24.transform("median"); F["ylev"]=np.where(F.base_my24>=med,"higher-yielding half","lower-yielding half")
T=fd[["farm_id","date","tmax2p","tmax_l1","fym","m"]]; SG=[]; DF=[]
for col,a,b_ in (("par2","multiparous","primiparous"),("stage","DIM 201-400","DIM 10-100"),("stage","DIM 101-200","DIM 10-100"),("ylev","higher-yielding half","lower-yielding half")):
    g=F.groupby(["farm_id","date",col]).D.agg(["mean","size"]).reset_index(); pv=g.pivot(index=["farm_id","date"],columns=col,values="mean"); nn=g.pivot(index=["farm_id","date"],columns=col,values="size")
    z=pd.DataFrame({"a":pv[a],"b":pv[b_],"na":nn[a],"nb":nn[b_]}).reset_index().merge(T,on=["farm_id","date"]); z=z[z.m.isin([5,6,7,8,9])&(z.na>=80)&(z.nb>=80)].copy(); z["d"]=z.a-z.b; z["h"]=np.maximum(z.tmax2p-76,0)
    ba,sa,n,G=fe(z,"a",["h"]); bb,sb,_,_=fe(z,"b",["h"]); bd,sd,_,_=fe(z,"d",["h"]); tc=2*stats.t.sf(abs(bd[0]/sd[0]),G-1)
    DF.append(dict(contrast=f"{a} minus {b_}",n=n,slope_a=ba[0]*100,se_a=sa[0]*100,slope_b=bb[0]*100,se_b=sb[0]*100,diff=bd[0]*100,se_diff=sd[0]*100,p=tc)); P(f"{a} {ba[0]*100:+.3f} ({sa[0]*100:.3f}) | {b_} {bb[0]*100:+.3f} ({sb[0]*100:.3f}) | difference {bd[0]*100:+.3f} (SE {sd[0]*100:.3f}; P={tc:.3f}; n={n})")
pd.DataFrame(DF).to_csv(V+"tab3_subgroup_diff.csv",index=False)
# share of cows below alternative thresholds
A=F.assign(**{f"s{int(t*100)}":(F.D<np.log(t)).astype(float) for t in (0.85,0.80,0.75,0.70)}).groupby(["farm_id","date"]).agg(n=("D","size"),s85=("s85","mean"),s80=("s80","mean"),s75=("s75","mean"),s70=("s70","mean")).reset_index(); A=A[A.n>=200].merge(T,on=["farm_id","date"]); A=A[A.m.isin([5,6,7,8,9])].copy(); A["h"]=np.maximum(A.tmax2p-76,0); A["hot"]=(A.tmax_l1>81).astype(float); A["cool"]=(A.tmax_l1<=68).astype(float); TH=[]
for c in ("s85","s80","s75","s70"):
    b,se,n,G=fe(A,c,["h"]); TH.append(dict(threshold=c[1:],mean_share=A[c].mean()*100,slope=b[0]*100,se=se[0]*100)); P(f"share below {c[1:]}%: mean {A[c].mean()*100:.2f}% ; slope {b[0]*100:+.3f} pp per unit above 76 (SE {se[0]*100:.3f})")
pd.DataFrame(TH).to_csv(V+"tab3_threshold_shares.csv",index=False)
P("DONE",round(time.time()-t0))
