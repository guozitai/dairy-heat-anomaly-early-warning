#!/usr/bin/env python3
"""C2. Disease classes from the recorded diagnosis (explicit keyword table; replaces the first grouping).
udder: mastitis, blood in milk, udder and teat lesions; abomasal_metabolic: all abomasal disorders, ketosis, milk fever, fatty liver;
digestive: forestomach and intestinal disorders; respiratory; locomotor: hoof and joint disorders incl. white line disease; reproductive tract; other."""
import pandas as pd, numpy as np, shutil, os
V="/home/guozitai/20261111_sat_rescue/v2/"
if not os.path.exists(V+"disease_cid_v1.parquet"): shutil.copy(V+"disease_cid.parquet",V+"disease_cid_v1.parquet")
d=pd.read_parquet(V+"disease_cid_v1.parquet")
RULES=(("mastitis",["乳房","血乳","乳头","乳区"]),("metabolic",["真胃","酮","瘫","脂肪肝","低钙"]),("digestive",["腹泻","消化","前胃","瘤胃","瓣胃","肠","积食","臌气","鼓气","胃炎"]),("respiratory",["肺","感冒","支气管","呼吸"]),("lameness",["蹄","关节","跛","肢","白线"]),("reproductive",["子宫","胎衣","产道","阴道","流产","早产"]))
def grp(s):
    s=str(s)
    if "滑倒" in s: return "other"
    for g,keys in RULES:
        if any(k in s for k in keys): return g
    return "other"
d["grp_v1"]=d.grp; d["grp"]=d.disease_name.map(grp); print(pd.crosstab(d.grp_v1,d.grp).to_string())
d.to_parquet(V+"disease_cid.parquet",index=False); print("rewritten",len(d))
