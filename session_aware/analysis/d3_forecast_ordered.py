#!/usr/bin/env python3
"""D3. Forecast skill in temporal order: forecasts of the warm season of 2025 corrected with the mean bias of 2024 (same lead time),
against the uncorrected forecasts and the persistence forecast of 2025. Runs locally on out/fcst_farm_daily.csv and out/era5_farm_daily.csv."""
import pandas as pd, numpy as np
O="/work/cea_sat/rescue/v2/out/"; pd.set_option("display.width",250)
T=pd.read_csv(O+"era5_farm_daily.csv",parse_dates=["date"])[["loc","date","tmax"]]; FC=pd.read_csv(O+"fcst_farm_daily.csv",parse_dates=["date"]).merge(T,on=["loc","date"]); FC=FC[FC.date.dt.month.isin([5,6,7,8,9])].copy(); FC["yr"]=FC.date.dt.year
rows=[]
for k in range(1,8):
    c=f"fc_tmax_l{k}"; x=FC.dropna(subset=[c]).copy(); b24=(x[c]-x.tmax)[x.yr==2024].mean(); y=x[x.yr==2025].copy(); y["cx"]=y[c]-b24; ob=y.tmax>=79; f0=y[c]>=79; f1=y.cx>=79
    yp=y.merge(T.assign(date=T.date+pd.Timedelta(days=k)).rename(columns={"tmax":"pers"}),on=["loc","date"],how="left"); fp=yp.pers>=79; obp=yp.tmax>=79
    rows.append(dict(lead=k,n=len(y),days79=int(ob.sum()),bias_2024=b24,pod_raw=(ob&f0).sum()/ob.sum(),far_raw=((~ob)&f0).sum()/max(f0.sum(),1),pod_cx=(ob&f1).sum()/ob.sum(),far_cx=((~ob)&f1).sum()/max(f1.sum(),1),pod_pers=(obp&fp).sum()/obp.sum(),far_pers=((~obp)&fp).sum()/max(fp.sum(),1),mae_raw=(y[c]-y.tmax).abs().mean(),mae_cx=(y.cx-y.tmax).abs().mean()))
R=pd.DataFrame(rows); print(R.round(3).to_string()); R.to_csv(O+"fig3_forecast_skill_2025.csv",index=False)
print("grid-cell days 2024/2025 reaching 79:",FC.groupby("yr").apply(lambda g:int((g.tmax>=79).sum())).to_dict())
