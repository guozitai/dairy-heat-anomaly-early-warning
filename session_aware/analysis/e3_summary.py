#!/usr/bin/env python3
"""E3. Summary of E1 (Chronos-Bolt zero-shot) and E2 (GRU, Transformer) over the 16 farm-years: mean, range and farm-years above the rule."""
import pandas as pd, numpy as np, glob
D="/home/guozitai/20261112_aiia_dl/"; OUT=D+"summary/"
import os; os.makedirs(OUT,exist_ok=True)
def rd(p): return pd.concat([pd.read_csv(x) for x in sorted(glob.glob(p))],ignore_index=True)
res=[]
for task,col,pats in ((1,"sens10",[D+"e1/t1_*.csv",D+"e2/t1_*.csv"]),(2,"case_sens5",[D+"e1/t2_*.csv",D+"e2/t2_*.csv"])):
    for p in pats:
        for fn in sorted(glob.glob(p)):
            z=pd.read_csv(fn); tagm=fn.split("/")[-1].split("_",1)[1].rsplit("_part",1)[0]
            rule=z[z.config=="R_deviation"].set_index("fold")[col]
            for cfg,g in z.groupby("config"):
                g=g.set_index("fold")[col]; above=(g>rule.reindex(g.index)).sum()
                res.append(dict(task=task,run=tagm,config=cfg if cfg in ("R_deviation","M_ref","M_ref+weather") else f"{cfg} [{tagm}]",folds=len(g),mean=g.mean(),min=g.min(),max=g.max(),above_rule=int(above) if cfg!="R_deviation" else np.nan,diff_vs_rule=(g-rule.reindex(g.index)).mean()))
R=pd.DataFrame(res).drop_duplicates(["task","config"]).sort_values(["task","mean"],ascending=[True,False]); R.to_csv(OUT+"e_detection.csv",index=False)
pd.set_option("display.width",200); print(R.round(4).to_string(index=False))
C=pd.concat([pd.read_csv(f).assign(run=f.split("/")[-1].split("_",1)[1].rsplit("_part",1)[0]) for f in sorted(glob.glob(D+"e1/comp_*.csv"))])
w=lambda g: pd.Series({"n":g.n.sum(),"flag_rate":np.average(g.flag_rate,weights=g.n),"flags_on_incomplete":np.average(g.flags_on_incomplete,weights=g.n*g.flag_rate),"incomplete_share":np.average(g.incomplete_share,weights=g.n),"flags_with_D_drop":np.average(g.flags_with_D_drop,weights=g.n*g.flag_rate),"folds":g.fold.nunique()})
S=C.groupby(["run","score"]).apply(w).reset_index(); S.to_csv(OUT+"e_flags.csv",index=False); print(S.round(4).to_string(index=False))
