#!/usr/bin/env python3
"""C11. Availability and dispersion of the three daily measures on eligible cow-days (values quoted in Section 3.1)."""
import pandas as pd, numpy as np
V="/home/guozitai/20261111_sat_rescue/v2/"; F=pd.read_parquet(V+"feat_v2.parquet",columns=["D","dev_my24","dev_raw","elig","n_dev"]); F=F[F.elig]; L=np.log(0.8); rs=lambda z: float((z-z.median()).abs().median()*1.4826)
print("eligible cow-days %d | D: available %.4f, robust SD %.5f, below ln0.8 %.4f | deviation of the 24-h equivalent yield: available %.4f, robust SD %.5f, below %.4f | deviation of the daily sum: available %.4f, robust SD %.5f, below %.4f"%(len(F),F.D.notna().mean(),rs(F.D.dropna()),(F.D.dropna()<L).mean(),F.dev_my24.notna().mean(),rs(F.dev_my24.dropna()),(F.dev_my24.dropna()<L).mean(),F.dev_raw.notna().mean(),rs(F.dev_raw.dropna()),(F.dev_raw.dropna()<L).mean()))
