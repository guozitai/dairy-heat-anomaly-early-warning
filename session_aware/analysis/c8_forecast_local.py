#!/usr/bin/env python3
"""C8. Archived forecasts against the reanalysis: skill by lead time without bias correction and with the mean bias of the other year removed
(bias estimated in 2024 applied to 2025 and vice versa); difference in herd deviation over the two following days after days forecast to reach 79.
Runs locally on out/farmday_all.csv, out/fcst_farm_daily.csv, out/era5_farm_daily.csv."""
import pandas as pd, numpy as np
O="/work/cea_sat/rescue/v2/out/"; pd.set_option("display.width",250)
T=pd.read_csv(O+"era5_farm_daily.csv",parse_dates=["date"])[["loc","date","tmax"]]; FC=pd.read_csv(O+"fcst_farm_daily.csv",parse_dates=["date"]).merge(T,on=["loc","date"]); FC=FC[FC.date.dt.month.isin([5,6,7,8,9])].copy(); FC["yr"]=FC.date.dt.year
print("grid-cell days with forecasts, warm seasons:",len(FC),FC.groupby("yr").size().to_dict(),"cells",FC["loc"].nunique(),"| days reaching 79 (ERA5):",int((FC.tmax>=79).sum()))
rows=[]
for k in range(1,8):
    c=f"fc_tmax_l{k}"; x=FC.dropna(subset=[c]).copy(); e=x[c]-x.tmax; bias_y=x.assign(e=e).groupby("yr").e.mean(); other={2024:bias_y.get(2025,np.nan),2025:bias_y.get(2024,np.nan)}; x["fc_cx"]=x[c]-x.yr.map(other); ob=x.tmax>=79; f0=x[c]>=79; f1=x.fc_cx>=79
    xp=x.merge(T.assign(date=T.date+pd.Timedelta(days=k)).rename(columns={'tmax':'pers'}),on=['loc','date'],how='left'); fp=xp.pers>=79; obp=xp.tmax>=79
    rows.append(dict(lead=k,pod_pers=(obp&fp).sum()/obp.sum(),far_pers=((~obp)&fp).sum()/max(fp.sum(),1),mae_pers=(xp.pers-xp.tmax).abs().mean(),n=len(x),mae=e.abs().mean(),bias=e.mean(),bias_2024=bias_y.get(2024,np.nan),bias_2025=bias_y.get(2025,np.nan),r=np.corrcoef(x[c],x.tmax)[0,1],pod_raw=(ob&f0).sum()/ob.sum(),far_raw=((~ob)&f0).sum()/max(f0.sum(),1),pod_cx=(ob&f1).sum()/ob.sum(),far_cx=((~ob)&f1).sum()/max(f1.sum(),1),mae_cx=(x.fc_cx-x.tmax).abs().mean()))
FS=pd.DataFrame(rows); print(FS.round(3).to_string()); FS.to_csv(O+"fig3_forecast_skill.csv",index=False)
fd=pd.read_csv(O+"farmday_all.csv",parse_dates=["date"]); fd["dkey"]=fd.date.dt.strftime("%Y-%m-%d")
def absorb(v,k): return np.asarray(v,float)-pd.Series(np.asarray(v,float)).groupby(k).transform("mean").values
def fe(df,y,xs):
    z=df.dropna(subset=[y]+xs); k=z["fym"].values; yd=absorb(z[y].values,k); Xm=np.column_stack([absorb(z[x].values,k) for x in xs]); b=np.linalg.lstsq(Xm,yd,rcond=None)[0]; e=yd-Xm@b; XtXi=np.linalg.pinv(Xm.T@Xm); S=Xm*e[:,None]
    def vc(G): sc=pd.DataFrame(S).groupby(G).sum().values; return XtXi@(sc.T@sc)@XtXi
    Vf=vc(z["fym"].values); Vd=vc(z["dkey"].values); V2=Vf+Vd-XtXi@(S.T@S)@XtXi; return b,np.sqrt(np.diag(Vf)),np.sqrt(np.clip(np.diag(V2),0,None)),len(z),z
h=fd[["farm_id","date","H"]]; h1=h.assign(date=h.date-pd.Timedelta(days=1)).rename(columns={"H":"Hp1"}); h2=h.assign(date=h.date-pd.Timedelta(days=2)).rename(columns={"H":"Hp2"})
f2=fd.merge(h1,on=["farm_id","date"],how="left").merge(h2,on=["farm_id","date"],how="left").merge(FC.drop(columns=["tmax"]),on=["loc","date"],how="inner"); f2["Hn"]=(f2.Hp1+f2.Hp2)/2; AN=[]
z0=f2.dropna(subset=["fc_tmax_l1"]).copy(); z0["x"]=(z0.tmax>=79).astype(float); b,se,se2,n,zz=fe(z0,"Hn",["x"]); AN.append(dict(lead=0,n=n,n_hot=int(zz.x.sum()),diff=b[0],se=se[0],se_twoway=se2[0]))
for k in range(1,8):
    c=f"fc_tmax_l{k}"; z=f2.dropna(subset=[c]).copy(); r=FS[FS.lead==k].iloc[0]; z["x"]=((z[c]-z.yr.map({2024:r.bias_2025,2025:r.bias_2024}))>=79).astype(float); b,se,se2,n,zz=fe(z,"Hn",["x"]); z["xr"]=(z[c]>=79).astype(float); br,ser,_,_,zr=fe(z,"Hn",["xr"])
    AN.append(dict(lead=k,n=n,n_hot=int(zz.x.sum()),diff=b[0],se=se[0],se_twoway=se2[0],n_hot_raw=int(zr.xr.sum()),diff_raw=br[0],se_raw=ser[0]))
AN=pd.DataFrame(AN); AN.to_csv(O+"tab3_forecast_announced.csv",index=False); print(AN.assign(diff=AN["diff"]*100,se=AN.se*100,se_twoway=AN.se_twoway*100,diff_raw=AN.diff_raw*100,se_raw=AN.se_raw*100).round(3).to_string())
