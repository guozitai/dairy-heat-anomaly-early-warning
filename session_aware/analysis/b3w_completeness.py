#!/usr/bin/env python3
"""B3w. Completeness of recording (share of cow-days with three recorded milkings) by class of daily maximum THI, within-estimates as in B3y.
Runs on the exported farm-day table (out/farmday_all.csv); appends outcome 'c3' to out/fig3_dose_response.csv."""
import pandas as pd, numpy as np
O="/work/cea_sat/rescue/v2/out/"; fd=pd.read_csv(O+"farmday_all.csv",parse_dates=["date"]); fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d"); warm=fd[fd.m.isin([5,6,7,8,9])].copy()
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]
    def vc(G): sc=pd.DataFrame(S).groupby(G).sum().values; return XtXi@(sc.T@sc)@XtXi
    Vf=vc(z["fym"].values); Vd=vc(z["dkey"].values); V2=Vf+Vd-XtXi@(S.T@S)@XtXi; return b,np.sqrt(np.diag(Vf)),np.sqrt(np.clip(np.diag(V2),0,None)),len(z)
bins=[0,68,72,75,77,79,81,90]; names=[str(i) for i in pd.cut(pd.Series([1.0]),bins).cat.categories]; rows=[]
for lagc in ("tmax","tmax_l1","tmax_l2"):
    z=warm.dropna(subset=["c3",lagc]).copy(); z["b"]=pd.cut(z[lagc],bins).astype(str); xs=[]
    for j,nm in enumerate(names[1:]): z[f"I{j}"]=(z.b==nm).astype(float); xs.append(f"I{j}")
    b,se,se2,n=fe(z,"c3",xs); cnt=z.b.value_counts(); rows.append(dict(outcome="c3",lag=lagc,b=names[0],size=int(cnt.get(names[0],0)),mean=0.0,sem=0.0,sem_twoway=0.0))
    for nm,bb,s1,s2 in zip(names[1:],b,se,se2): rows.append(dict(outcome="c3",lag=lagc,b=nm,size=int(cnt.get(nm,0)),mean=bb,sem=s1,sem_twoway=s2))
    print(lagc," | ".join(f"{nm}: {bb*100:+.2f} ({s*100:.2f})" for nm,bb,s in zip(names[1:],b,se)))
B=pd.read_csv(O+"fig3_dose_response.csv"); B=pd.concat([B[B.outcome!="c3"],pd.DataFrame(rows)],ignore_index=True); B.to_csv(O+"fig3_dose_response.csv",index=False)
print("warm-season mean share with three recorded milkings: %.3f"%warm.c3.mean(), "| outcomes:",sorted(B.outcome.unique()))
