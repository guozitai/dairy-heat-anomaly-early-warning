#!/usr/bin/env python3
"""D6. Sensitivity to Farm 3 from 7 June 2025 (part of the herd milked four times a day), local part:
(a) herd-level response without the farm-days of Farm 3 from that date; (b) early-warning means without the farm-year Farm 3 2025.
Runs on out/farmday_all.csv, out/ew_onset.csv and out/ew_disease.csv."""
import pandas as pd, numpy as np
O="/work/cea_sat/rescue/v2/out/"; pd.set_option("display.width",250)
fd=pd.read_csv(O+"farmday_all.csv",parse_dates=["date"]); fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d"); warm=fd[fd.m.isin([5,6,7,8,9])].copy()
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]
    sc=pd.DataFrame(S).groupby(z["fym"].values).sum().values; Vf=XtXi@(sc.T@sc)@XtXi; return b,np.sqrt(np.diag(Vf)),len(z)
bins=[0,68,72,75,77,79,81,90]; names=[str(i) for i in pd.cut(pd.Series([1.0]),bins).cat.categories]; rows=[]
for lab,w in (("all farm-days",warm),("without Farm 3 from 7 June 2025",warm[~((warm.farm=="Farm 3")&(warm.date>="2025-06-07"))])):
    z=w.dropna(subset=["H","tmax2p"]).copy(); z["x"]=np.maximum(z.tmax2p-76,0); b,se,n=fe(z,"H",["x"])
    y=w.dropna(subset=["H","tmax_l1"]).copy(); y["b"]=pd.cut(y.tmax_l1,bins).astype(str); xs=[]
    for j,nm in enumerate(names[1:]): y[f"I{j}"]=(y.b==nm).astype(float); xs.append(f"I{j}")
    bc,sec,nc=fe(y,"H",xs)
    print("%-34s farm-days %d | slope above 76: %.3f (SE %.3f) | THI 1 day earlier, 79-81: %.2f (%.2f), >81: %.2f (%.2f)"%(lab,n,100*b[0],100*se[0],100*bc[-2],100*sec[-2],100*bc[-1],100*sec[-1]))
    rows.append(dict(set=lab,n=n,slope76=b[0],se76=se[0],c79_81=bc[-2],se79_81=sec[-2],c81=bc[-1],se81=sec[-1]))
f3=warm[(warm.farm=="Farm 3")]; print("Farm 3 warm-season farm-days with H: %d, of which from 7 June 2025: %d"%(f3.H.notna().sum(),(f3.H.notna()&(f3.date>="2025-06-07")).sum()))
for lab,w in (("Farm 3, all",f3),("Farm 3, before 7 June 2025",f3[f3.date<"2025-06-07"])):
    z=w.dropna(subset=["H","tmax2p"]).copy(); z["x"]=np.maximum(z.tmax2p-76,0); b,se,n=fe(z,"H",["x"]); print("  %-28s farm-days %d (above 76: %d) slope %.3f (SE %.3f)"%(lab,n,int((z.x>0).sum()),100*b[0],100*se[0]))
pd.DataFrame(rows).to_csv(O+"tab3_without_farm3_2025.csv",index=False)
A="R_deviation"; B="M_ref"; out=[]
for nm,f,col in (("Task 1","ew_onset.csv","sens10"),("Task 2","ew_disease.csv","case_sens5")):
    e=pd.read_csv(O+f); p=e.pivot(index="fold",columns="config",values=col)
    for lab,q in (("16 farm-years",p),("without Farm 3 2025",p.drop(index="Farm 3 2025"))):
        print("%s %-22s rule %.4f | model %.4f | model above rule in %d of %d"%(nm,lab,q[A].mean(),q[B].mean(),int((q[B]>q[A]).sum()),len(q))); out.append(dict(task=nm,set=lab,rule=q[A].mean(),model=q[B].mean(),wins=int((q[B]>q[A]).sum()),n=len(q)))
    print("   Farm 3 2025: rule %.4f model %.4f"%(p.loc["Farm 3 2025",A],p.loc["Farm 3 2025",B]))
pd.DataFrame(out).to_csv(O+"ew_without_farm3_2025.csv",index=False)
