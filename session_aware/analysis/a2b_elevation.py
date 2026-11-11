#!/usr/bin/env python3
"""A2b. Elevation adjustment applied by the weather archive: series requested for the farm elevation (as used in the analyses) against the
unadjusted values of the same 0.25-degree grid cell (elevation=nan), May to September 2024. Writes era5_elevation_adjustment.csv."""
import urllib.request, json, time
import pandas as pd, numpy as np
LOC={"A":(38.25,106.00,1127,"Farm 8"),"B":(38.50,106.25,1116,"Farm 1"),"C":(37.75,106.50,1333,"Farms 2, 4 and 5"),"D":(39.00,106.25,1097,"Farms 3 and 6"),"E":(38.50,106.00,1123,"Farm 7")}
def thi(T,RH): return (1.8*T+32)-(0.55-0.0055*RH)*(1.8*T-26)
def get(url):
    for a in range(5):
        try: return json.loads(urllib.request.urlopen(url,timeout=180).read())
        except Exception as e: print("retry",repr(e)[:120],flush=True); time.sleep(6)
    raise SystemExit("failed "+url)
era="https://archive-api.open-meteo.com/v1/archive?start_date=2024-05-01&end_date=2024-09-30&hourly=temperature_2m,relative_humidity_2m&models=era5&timezone=Asia%2FShanghai"
fc="https://previous-runs-api.open-meteo.com/v1/forecast?start_date=2024-07-01&end_date=2024-07-07&hourly=temperature_2m&timezone=Asia%2FShanghai&models=ecmwf_ifs025"
rows=[]
for k,(la,lo,el,farms) in LOC.items():
    a=get(era+f"&latitude={la}&longitude={lo}&elevation={el}"); n=get(era+f"&latitude={la}&longitude={lo}&elevation=nan"); f=get(fc+f"&latitude={la}&longitude={lo}&elevation=nan"); time.sleep(1)
    A=pd.DataFrame(a["hourly"]); N=pd.DataFrame(n["hourly"]); A["thi"]=thi(A.temperature_2m,A.relative_humidity_2m); N["thi"]=thi(N.temperature_2m,N.relative_humidity_2m)
    day=pd.to_datetime(A.time).dt.normalize(); dmax=A.groupby(day).thi.max()-N.groupby(day).thi.max()
    rows.append(dict(loc=k,farms=farms,grid_lat=a["latitude"],grid_lon=a["longitude"],elev_farm=a["elevation"],elev_cell_era5=n["elevation"],elev_cell_ifs=f["elevation"],
        dT_mean=(A.temperature_2m-N.temperature_2m).mean(),dT_sd=(A.temperature_2m-N.temperature_2m).std(),dRH_mean=(A.relative_humidity_2m-N.relative_humidity_2m).mean(),
        dTHImax_mean=dmax.mean(),dTHImax_min=dmax.min(),dTHImax_max=dmax.max(),lapse_per_100m=(A.temperature_2m-N.temperature_2m).mean()/((n["elevation"]-a["elevation"])/100) if n["elevation"]!=a["elevation"] else np.nan))
R=pd.DataFrame(rows); R.to_csv("/work/cea_sat/rescue/v2/out/era5_elevation_adjustment.csv",index=False); pd.set_option("display.width",250); print(R.round(3).to_string())
