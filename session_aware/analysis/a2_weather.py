#!/usr/bin/env python3
"""A2. Heat exposure at farm locations. ERA5 (0.25 deg) hourly 2-m temperature and relative humidity from the Open-Meteo archive API,
local time (Asia/Shanghai); THI after NRC (1971); daily aggregation. Archived ECMWF IFS 0.25 deg forecasts by lead day (1-7) from the Previous Runs API."""
import urllib.request, json, time, sys
import pandas as pd, numpy as np
# Centres of the 0.25-degree grid cells nearest to the farm locations and elevations of the farm locations (m; Supplementary Table S1).
# The exact farm coordinates are not given. A request for the cell centre with the elevation of the farm returns the series of a request
# for the farm location itself: the API takes the values of the grid cell and adjusts air temperature to the stated elevation.
LOC={"A":(38.25,106.00,1127,[3]),"B":(38.50,106.25,1116,[6]),"C":(37.75,106.50,1333,[13,14,17]),"D":(39.00,106.25,1097,[15,16]),"E":(38.50,106.00,1123,[5])}
def thi(T,RH): return (1.8*T+32)-(0.55-0.0055*RH)*(1.8*T-26)
def get(url):
    for a in range(5):
        try: return json.loads(urllib.request.urlopen(url,timeout=180).read())
        except Exception as e: print("retry",repr(e)[:120],flush=True); time.sleep(6)
    raise SystemExit("failed "+url)
era=[]; meta=[]
for k,(la,lo,el,farms) in LOC.items():
    for (a,b) in [("2022-01-01","2022-12-31"),("2023-01-01","2023-12-31"),("2024-01-01","2024-12-31"),("2025-01-01","2025-10-05")]:
        j=get(f"https://archive-api.open-meteo.com/v1/archive?latitude={la}&longitude={lo}&elevation={el}&start_date={a}&end_date={b}&hourly=temperature_2m,relative_humidity_2m&models=era5&timezone=Asia%2FShanghai")
        h=pd.DataFrame(j["hourly"]); h["time"]=pd.to_datetime(h.time); h["loc"]=k; era.append(h)
        if a.startswith("2022"): meta.append(dict(loc=k,req_lat=la,req_lon=lo,grid_lat=j.get("latitude"),grid_lon=j.get("longitude"),elev=j.get("elevation"),farms=str(farms)))
        print("era5",k,a,len(h),int(h.temperature_2m.notna().sum()),flush=True); time.sleep(1)
H=pd.concat(era,ignore_index=True); H["thi"]=thi(H.temperature_2m,H.relative_humidity_2m); H["date"]=H.time.dt.normalize()
g=H.groupby(["loc","date"])
Dd=g.agg(tmax=("thi","max"),tmean=("thi","mean"),tmin=("thi","min"),h72=("thi",lambda s:(s>72).sum()),h79=("thi",lambda s:(s>79).sum()),tair_max=("temperature_2m","max"),rh_mean=("relative_humidity_2m","mean"),n=("thi","count")).reset_index()
hr=H.time.dt.hour; night=H[(hr>=20)|(hr<6)].copy(); night["nd"]=(night.time+pd.Timedelta(hours=4)).dt.normalize()
nm=night.groupby(["loc","nd"]).thi.min().rename("nmin").reset_index().rename(columns={"nd":"date"}); Dd=Dd.merge(nm,on=["loc","date"],how="left")
Dd.to_csv("era5_farm_daily.csv",index=False); pd.DataFrame(meta).to_csv("era5_farm_grid.csv",index=False)
print(pd.DataFrame(meta).to_string()); print(Dd.groupby("loc").agg(d0=("date","min"),d1=("date","max"),n=("date","size"),tmax_max=("tmax","max"),days79=("tmax",lambda s:(s>=79).sum())).to_string(),flush=True)
# forecasts by lead
leads=[1,2,3,4,5,6,7]; vars_=["temperature_2m","relative_humidity_2m"]
hourly=",".join(vars_+[f"{v}_previous_day{k}" for v in vars_ for k in leads]); out=[]
for k,(la,lo,el,farms) in LOC.items():
    for (a,b) in [("2024-02-01","2024-06-30"),("2024-07-01","2024-12-31"),("2025-01-01","2025-06-30"),("2025-07-01","2025-10-05")]:
        j=get(f"https://previous-runs-api.open-meteo.com/v1/forecast?latitude={la}&longitude={lo}&elevation={el}&hourly={hourly}&start_date={a}&end_date={b}&timezone=Asia%2FShanghai&models=ecmwf_ifs025")
        h=pd.DataFrame(j["hourly"]); h["time"]=pd.to_datetime(h.time); h["loc"]=k; out.append(h); print("fc",k,a,len(h),flush=True); time.sleep(1)
F=pd.concat(out,ignore_index=True); F["date"]=F.time.dt.normalize(); F["thi_l0"]=thi(F.temperature_2m,F.relative_humidity_2m)
for k in leads: F[f"thi_l{k}"]=thi(F[f"temperature_2m_previous_day{k}"],F[f"relative_humidity_2m_previous_day{k}"])
agg={f"fc_tmax_l{k}":(f"thi_l{k}","max") for k in [0]+leads}; agg.update({f"fc_n_l{k}":(f"thi_l{k}","count") for k in [0]+leads})
Fd=F.groupby(["loc","date"]).agg(**agg).reset_index()
for k in [0]+leads: Fd.loc[Fd[f"fc_n_l{k}"]<20,f"fc_tmax_l{k}"]=np.nan
Fd.drop(columns=[c for c in Fd.columns if c.startswith("fc_n_")]).to_csv("fcst_farm_daily.csv",index=False)
print("forecast days",len(Fd),"DONE",flush=True)
