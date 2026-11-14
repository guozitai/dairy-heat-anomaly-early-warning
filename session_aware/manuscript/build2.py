#!/usr/bin/env python3
"""Assemble manuscript, supplementary material, response letter and highlights (HTML -> docx/pdf via soffice).
All table numbers are read from the result CSVs in ../v2/out. Usage: build2.py ms|supp|response|highlights|all"""
import re, html, subprocess, sys, os, base64
import pandas as pd, numpy as np
MS='/work/cea_sat/rescue/ms/'; DAT='/work/cea_sat/rescue/v2/out/'
BASE='body{font-family:"Times New Roman",serif;font-size:12pt;line-height:2.0;margin:0}h1{font-size:15pt;line-height:1.4}h2{font-size:13pt;margin-top:16pt}h3{font-size:12pt;font-style:italic;margin-top:12pt}p{margin-top:6pt;margin-bottom:6pt;text-align:left}p.eq{text-align:center}p.cap{font-weight:bold;margin-top:14pt;line-height:1.4}p.note{font-size:10pt;line-height:1.3}td,th{font-size:9.5pt;line-height:1.25}'
CSS='@page{size:21cm 29.7cm;margin:2.5cm}'+BASE
CSS2=CSS.replace("line-height:2.0","line-height:1.4")
CSSL='@page{size:29.7cm 21cm;margin:1.6cm}'+BASE.replace("line-height:2.0","line-height:1.4").replace("td,th{font-size:9.5pt","td,th{font-size:9pt")
M=lambda s: str(s).replace("-","−")
def fi(n): return f"{int(round(float(n))):,}"
def table(header, rows, widths=None, align=None, note=None, caption=None, newpage=False):
    n=len(header); align=align or ["left"]+["right"]*(n-1); out=[]
    docxfix.TW.append(list(widths) if widths else None)
    if caption: out.append(f'<p class="cap" style="page-break-before:always">{caption}</p>' if newpage else f'<p class="cap">{caption}</p>')
    out.append('<table width="100%" cellspacing="0" cellpadding="3" style="border-collapse:collapse">')
    def cell(tag,txt,j,first=False,last=False):
        st="padding:2.5pt 4pt;vertical-align:top;"
        if first: st+="border-top:1.5pt solid #000000;border-bottom:0.75pt solid #000000;font-weight:bold;"
        if last: st+="border-bottom:1.5pt solid #000000;"
        w=f' width="{widths[j]}%"' if widths else ""
        nw=' nowrap' if (j==0 and tag=="td" and len(re.sub(r"<.*?>|&nbsp;","",str(txt)))<=14) else ""
        return f'<{tag}{w}{nw} align="{align[j]}" style="{st}text-align:{align[j]};">{txt}</{tag}>'
    out.append("<tr>"+"".join(cell("th",h,j,first=True) for j,h in enumerate(header))+"</tr>")
    for i,r in enumerate(rows): out.append("<tr>"+"".join(cell("td",x,j,last=(i==len(rows)-1)) for j,x in enumerate(r))+"</tr>")
    out.append("</table>")
    if note: out.append(f'<p class="note">{note}</p>')
    return "\n".join(out)
def md2html(t):
    out=[]
    for blk in re.split(r"\n\s*\n",t.strip()):
        b=blk.strip()
        if not b: continue
        if b.startswith("{{") or b.startswith("<table") or b.startswith("<p"): out.append(b); continue
        b=html.escape(b,quote=False)
        b=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",b); b=re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])",r"<i>\1</i>",b); b=re.sub(r"\^(.+?)\^",r"<sup>\1</sup>",b)
        b=re.sub(r"(\S)_(ref|it|fd|fm)\b",r"\1<sub>\2</sub>",b)
        if b.startswith("### "): out.append(f"<h3>{b[4:]}</h3>")
        elif b.startswith("## "): out.append(f"<h2>{b[3:]}</h2>")
        elif b.startswith("# "): out.append(f"<h1>{b[2:]}</h1>")
        elif re.match(r"^(e = |D<sub>it</sub> = |THI = \(|H<sub>fd</sub> = )",b): out.append(f'<p class="eq">{b}</p>')
        else: out.append("<p>"+b.replace("\n","<br>")+"</p>")
    return "\n".join(out)
import docxfix
from docxfix import fix_docx
def to_docx(name, body, css=CSS, lines=False, pages=True):
    h=f'<html><head><meta charset="utf-8"><style>{css}</style></head><body>\n{body}\n</body></html>'
    open(MS+name+".html","w").write(h)
    subprocess.run(["soffice","--headless","--convert-to",'docx:MS Word 2007 XML',"--outdir",MS,MS+name+".html"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
    tw=list(docxfix.TW); docxfix.TW.clear(); fix_docx(MS+name+".docx",line_numbers=lines,page_numbers=pages,widths=tw)
    subprocess.run(["soffice","--headless","--convert-to","pdf","--outdir",MS,MS+name+".docx"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
R=lambda f,**k: pd.read_csv(DAT+f,**k)
FARMS=[f"Farm {i}" for i in range(1,9)]
COUNTY={"Farm 1":"Xixia","Farm 2":"Lingwu","Farm 3":"Pingluo","Farm 4":"Lingwu","Farm 5":"Lingwu","Farm 6":"Pingluo","Farm 7":"Xixia","Farm 8":"Xixia"}
fkey=lambda s:(int(s.split()[1]),int(s.split()[2]))
p1=lambda x: f"{100*x:.1f}"
def mr(x,nd=1): x=np.asarray(x,float); return f"{100*np.nanmean(x):.{nd}f} ({100*np.nanmin(x):.{nd}f} to {100*np.nanmax(x):.{nd}f})"
A="R_deviation"; B="M_ref"
N1={"R_deviation":"Deviation rule (D of day t)","R_mean3":"Mean of D, days t − 2 to t","R_loss_kg":"Loss in kg on day t","R_sd7":"Standard deviation of D, 7 days","R_drops30":"Drop days in 30 days","R_change3":"Change of D, day t − 2 to t",
    "R_min3":"Minimum of D, days t − 2 to t","R_fewer_milkings":"Fewer recorded milkings than herd median",
    "R_my24_deviation":"Deviation of the 24-h equivalent yield","R_deviation_regular_only":"D from regular-interval milkings only","R_deviation_lagged_reference":"D with reference window t − 21 to t − 9","R_dailysum_deviation":"Deviation of the daily sum","R_dailysum_cohort_ratio":"Daily sum, training-cohort ratio",
    "M_logistic":"Logistic regression, features of the main model","M_yield":"Boosting: yield history","M_yield+sensors":"Boosting: yield history and sensors","M_ref":"Boosting: main model","M_ref+weather":"Boosting: main model and weather","M_weather+DIM+parity":"Boosting: weather, DIM and parity only","M_no_yield":"Boosting: all features except yield history",
    "M_dynamic":"Boosting: deviations, sensors and herd context only","M_static":"Boosting: baseline yield, DIM, parity and disease history only","B_rank_mean(rule,model)":"Mean of the within-day ranks of deviation rule and main model"}
# ---------------------------------------------------------------- main tables
def t1():
    FD=R("tab_farm_descriptives.csv").set_index("farm"); TOT=R("tab_totals.csv",index_col=0).iloc[:,0]; FL=R("tab_disease_flow_by_farm.csv").set_index("farm"); rows=[]
    for f in FARMS:
        r=FD.loc[f]; rows.append([f,r["first"][:7],fi(r.cows),f"{r.cow_days/1e3:,.0f}",p1(r.s3),p1(r.s2),p1(r.s1),f"{r.episodes/1e3:.1f} ({p1(r.share_1d)})",fi(FL.loc[f,"target"]),fi(FL.loc[f,"target_risk"])])
    rows.append(["All",str(TOT["first"])[:7],fi(TOT.cows),f"{float(TOT.cow_days)/1e3:,.0f}",p1(float(TOT.s3)),p1(float(TOT.s2)),p1(float(TOT.s1)),f"{float(TOT.episodes)/1e3:.1f} ({p1(float(TOT.share_1d))})",fi(FL.target.sum()),fi(FL.target_risk.sum())])
    return table(["Farm","First record","Cows","Cow-days,<br>10<sup>3</sup>","Three or more recorded milkings, % of cow-days","Two,<br>%","One,<br>%","Drop episodes, 10<sup>3</sup><br>(of one day, %)","New disease records at 10 to 400 DIM","Of which with a preceding day in the Task 2 risk set"],rows,widths=[9,9,8,12,12,7,7,14,10,12],
        caption="Table 1. Farms, records, recorded milkings per cow-day, episodes of acute drop and veterinary disease records.",
        note="Farms 1, 7 and 8 are in Xixia District, Farms 2, 4 and 5 in Lingwu City and Farms 3 and 6 in Pingluo County. Records ended on 27 or 28 September 2025 (10 March 2025 on Farm 6). Cow-days are those at 10 to 400 days in milk (DIM) with known parity. An episode is a run of consecutive days with a yield below 80% of the expected yield. The Task 2 risk set is defined in Section 2.8; a record counts when the cow was in it on at least one of the three days before the record.")
def t2():
    rows=[["Deviation rule","Session-aware deviation D of day t, lowest first","Yield of the day relative to the yield expected from the cow's milkings of the preceding week"],
          ["Other rules","Mean of D over days t − 2 to t (lowest first); loss in kg, baseline yield minus 24-h equivalent yield of day t (largest first). Task 1 only: change of D from day t − 2 to t (largest decrease first); standard deviation of D over days t − 6 to t (largest first); number of days with an acute drop in the last 30 days (most first). Task 2 only: minimum of D over days t − 2 to t (lowest first); recorded milkings relative to the herd median (fewest first)","Persistence, trend and variability of the cow's own yield; completeness of her records"],
          ["Comparators","Deviation of the 24-h equivalent yield; deviation of the daily sum; D from regular-interval milkings only; D with reference window t − 21 to t − 9; training-cohort ratio of the daily sum (Task 2) (all lowest first)","Other measures of daily yield and another reference window"],
          ["Yield history (24 features, with stage and records)","D of days t to t − 6 and their mean, minimum, standard deviation and change; days with an acute drop in 7 and 30 days; lowest deviation among the milkings of day t; loss in kg; baseline yield and 24-h equivalent yield; DIM; parity; recorded and usable milkings; recorded milkings relative to herd median","Level, trend and variability of the cow's recent yield, stage of lactation and completeness of records"],
          ["Sensors (4)","Deviation of mean and maximum conductivity and of flow rate from the cow's median of days t − 8 to t − 2; conductivity deviation of day t − 1","Udder health and milk let-down as recorded by the parlour at each milking"],
          ["Herd context (2)","Herd mean of D and share of cows in acute drop on day t","Events common to the herd on that day"],
          ["Disease history (1; Task 2)","Days since the last veterinary record of the cow","Recurrence of disease"],
          ["Weather (6)","Daily maximum THI of day t and of day t + 1; mean of days t − 2 to t; difference from the mean of days t − 8 to t − 2; night minimum; hours above 72","Heat load on the day of the list and the day ahead"]]
    return table(["Score or feature group","Variables and ranking direction","What it represents"],rows,widths=[18,54,28],align=["left","left","left"],caption="Table 2. Rules, comparators and feature groups of the models used to rank cows at the end of day t.",
        note="Rules and comparators rank cows directly. The feature groups enter gradient-boosting models in six combinations: yield history; yield history and sensors; these with herd context and, in Task 2, disease history (main model); the main model with weather; weather with DIM and parity only; all groups except yield history. A logistic regression used the features of the main model. All variables are defined in Supplementary Table S8.")
VN=[("All diseases","All disease records"),("Udder","Udder disorders"),("Digestive","Digestive disorders"),("Abomasal and metabolic","Abomasal and metabolic disorders"),("Locomotor","Locomotor disorders"),("Respiratory","Respiratory disease"),("Reproductive tract","Reproductive tract disorders"),("Other","Other diagnoses"),("Oestrus recorded","Oestrus observation"),("Insemination","Insemination"),("Vaccination","Vaccination"),("Hoof trimming","Hoof trimming"),("Pen change","Pen change"),("Pregnancy check","Pregnancy check")]
def t3():
    V=R("tab2_validity.csv").set_index("group"); rows=[]
    for g,lab in VN:
        r=V.loc[g]; rows.append([lab,fi(r.pre3_n),f"{100*r.pre3:.1f} ({100*r.pre3_lo:.1f} to {100*r.pre3_hi:.1f})",f"{100*r.pre3_ctrl:.1f}",f"{100*r.d1:.1f} / {100*r.d1_ctrl:.1f}",f"{100*r.pre32:.1f} / {100*r.pre32_ctrl:.1f}",f"{100*r.pre7_4:.1f} / {100*r.pre7_4_ctrl:.1f}"])
    return table(["Record","Records evaluated","With acute drop on days −3 to −1, % (95% CI)","Controls, %","On day −1: records / controls, %","On days −3 to −2: records / controls, %","On days −7 to −4: records / controls, %"],rows,
        caption="Table 3. Acute drops before veterinary disease records and management records, with matched controls.",
        note="Disease records are new records (no record of the cow in the preceding 14 days). Controls were cows of the same farm, day, parity class and 30-day class of days in milk, with a value of D on at least one of the three preceding days and without a disease record from 14 days before to 3 days after the record day (disease records) or without a record of the same kind within 3 days (management records). Records evaluated had at least two values of D on days −3 to −1; the other columns require one value of D in the window (three for days −7 to −4). Confidence intervals from a bootstrap over cows.")
def t4():
    O=R("ew_onset.csv"); folds=sorted(O.fold.unique(),key=fkey); pv=O.pivot(index="fold",columns="config",values="sens10").loc[folds]; rows=[]; a=O[O.config==A]
    rows.append(["Lists drawn at random",f"{100*a.chance10.mean():.1f}",f"{100*a.chance10.mean():.1f}","","",""])
    order=["R_deviation","R_mean3","R_loss_kg","R_sd7","R_drops30","R_change3","R_my24_deviation","R_deviation_regular_only","R_deviation_lagged_reference","R_dailysum_deviation","M_logistic","M_yield","M_yield+sensors","M_ref","M_ref+weather","M_weather+DIM+parity","M_no_yield"]
    for c in order:
        z=O[O.config==c].set_index("fold").loc[folds]; wins="" if c==A else f"{int((pv[c]>pv[A]).sum())}/{len(pv)}"
        rows.append([N1[c],mr(z.sens10),f"{100*z.sens10_conf.mean():.1f}",f"{100*z.prec10.mean():.1f}",f"{z.entries_per_detection.mean():.1f}",wins])
    return table(["Score","Onsets on the lists, % (range)","Confirmed onsets on the lists, %","Precision, %","List entries per onset detected","Farm-years above the deviation rule"],rows,caption=f"Table 4. Early warning of acute drops on the next day (Task 1): daily lists of 10% of the cows at risk in {len(folds)} farm-years of leave-one-farm-out validation.",
        note="Values are means over farm-years. An onset is confirmed when the yield of the following day is still at least 10% below expectation. Precision is the share of list entries with a known outcome that were followed by an onset. Models were fitted to the other farms with records up to the end of the preceding year.")
def t5():
    Dz=R("ew_disease.csv"); folds=sorted(Dz.fold.unique(),key=fkey); p5=Dz.pivot(index="fold",columns="config",values="case_sens5").loc[folds]; a=Dz[Dz.config==A]; rows=[]
    rows.append(["Lists drawn at random",f"{100*a.chance5.mean():.1f}",f"{100*a.chance10.mean():.1f}","",f"{100*a.chance_prec5.mean():.1f}",f"{a.chance_entries5.mean():.0f}",""])
    order=["R_dailysum_cohort_ratio","R_dailysum_deviation","R_my24_deviation","R_deviation_regular_only","R_deviation_lagged_reference","R_deviation","R_min3","R_mean3","R_loss_kg","R_fewer_milkings","M_logistic","M_yield","M_yield+sensors","M_ref","M_ref+weather","M_weather+DIM+parity","M_no_yield","B_rank_mean(rule,model)"]
    for c in order:
        z=Dz[Dz.config==c].set_index("fold").loc[folds]; wins="" if c==A else f"{int((p5[c]>p5[A]).sum())}/{len(p5)}"
        rows.append([N1[c],mr(z.case_sens5),f"{100*z.case_sens10.mean():.1f}",f"{100*z.case_sens5_early.mean():.1f}",f"{100*z.prec5.mean():.1f}",f"{z.entries_per_case5.mean():.0f}",wins])
    return table(["Score","Cases detected, lists of 5% of cows, % (range)","Lists of 10%, %","Lists of days −3 and −2 only, 5%, %","Precision, 5%, %","List entries per detected case, 5%","Farm-years above the deviation rule (5%)"],rows,caption=f"Table 5. Early warning of new veterinary disease records within three days (Task 2) in {len(folds)} farm-years of leave-one-farm-out validation.",
        note="Values are means over farm-years. A case is detected when the cow was on the daily list on at least one of the days before the record on which she was in the risk set; the fourth column counts the lists of two and three days before the record only, among cases in the risk set on these days. Precision is the share of list entries followed by a new disease record within three days.")
def t6():
    ED="/work/cea_sat/rescue/v2/out/e_dl/"; col={1:"sens10",2:"case_sens5"}
    pv={(t,r):pd.read_csv(ED+f"t{t}_{r}_part0.csv").pivot(index="fold",columns="config",values=col[t]) for t in (1,2) for r in ("gru","tf","chronos-bolt-base")}
    spec=[("Deviation rule (D of day t)","chronos-bolt-base","R_deviation"),("Gradient boosting, main model","chronos-bolt-base","M_ref"),("Recurrent network (GRU)","gru","DL_gru"),("Transformer encoder","tf","DL_tf"),
          ("Foundation model, 24-h equivalent yield: deviation of day t from forecast","chronos-bolt-base","FM_anom_my24"),("Foundation model, 24-h equivalent yield: forecast 10% quantile for day t + 1","chronos-bolt-base","FM_fc_my24"),("Foundation model, 24-h equivalent yield: forecast median for day t + 1","chronos-bolt-base","FM_fc50_my24"),
          ("Foundation model, daily sum: deviation of day t from forecast","chronos-bolt-base","FM_anom_y_sum"),("Foundation model, daily sum: forecast 10% quantile for day t + 1","chronos-bolt-base","FM_fc_y_sum"),("Foundation model, daily sum: forecast median for day t + 1","chronos-bolt-base","FM_fc50_y_sum")]
    rows=[]
    for lab,run,c in spec:
        r=[lab]
        for t in (1,2):
            z=pv[(t,run)]; r.append(mr(z[c])); r.append("" if c=="R_deviation" else f"{int((z[c]>z['R_deviation']).sum())}/{len(z)}")
        rows.append(r)
    return table(["Score","Task 1: onsets on 10% lists, % (range)","Above the deviation rule","Task 2: cases detected with 5% lists, % (range)","Above the deviation rule"],rows,widths=[40,17,11,20,12],caption="Table 6. Early warning with deep sequence models and a time-series foundation model (Chronos-Bolt, base, without training) in 16 farm-years of leave-one-farm-out validation.",
        note="Values are means over farm-years; the last column of each task gives the farm-years in which the score was above the deviation rule. The deep sequence models were trained on the same data as gradient boosting. Forecasts of the foundation model were made from days t − 28 to t − 1 (day t) or t − 27 to t (day t + 1); quantiles for day t + 1 are relative to the median of the context. Scores were ranked lowest first, except model probabilities.")
def nom():
    rows=[["D<sub>it</sub>","Session-aware deviation of cow i on day t, ln(Σ y / Σ e)","–"],["e","Expected yield of a milking","kg"],["y","Recorded yield of a milking","kg"],["r<sub>ref</sub>","Reference secretion rate of the cow in the shift","kg h<sup>−1</sup>"],
          ["Δ","Interval since the previous recorded milking","h"],["t","Day","d"],["H<sub>fd</sub>","Herd deviation, mean of D over the cows of farm f on day d","–"],["THI","Temperature-humidity index","–"],["T","Air temperature","°C"],["RH","Relative humidity","%"],
          ["THI<sub>fd</sub>","Mean daily maximum THI of the two preceding days on farm f and day d","–"],["τ","Threshold of THI in the slope model","–"],["β","Slope of H on THI above τ","THI<sup>−1</sup>"],["α<sub>fm</sub>","Fixed effect of farm f in calendar month m","–"],["ε<sub>fd</sub>","Residual","–"],
          ["k","List size, share of the cows in the risk set","%"],["DIM","Days in milk","d"],["ERA5","Fifth-generation ECMWF atmospheric reanalysis","–"],["ECMWF","European Centre for Medium-Range Weather Forecasts","–"],["GRU","Gated recurrent unit","–"],["ICAR","International Committee for Animal Recording","–"],["ROC","Receiver operating characteristic","–"]]
    return table(["Symbol or abbreviation","Definition","Unit"],rows,align=["left","left","left"],caption="Nomenclature")
TABLES={"NOM":nom,"T1":t1,"T2":t2,"T3":t3,"T4":t4,"T5":t5,"T6":t6}
def build_ms():
    src=open(MS+"manuscript.md").read(); res=open(MS+"results_a.md").read().rstrip()+"\n\n"+open(MS+"results_b.md").read().lstrip()+"\n\n"+open(MS+"results_c.md").read().lstrip()
    for k,txt in (("{{ABSTRACT}}",open(MS+"abstract.md").read().strip()),("{{RESULTS}}",res),("{{DISCUSSION}}",open(MS+"discussion.md").read()),("{{CONCLUSIONS}}",open(MS+"conclusions.md").read()),("{{BACK}}",open(MS+"back.md").read())): src=src.replace(k,txt)
    open(MS+"manuscript_full.md","w").write(src); body=md2html(src)
    for k,fn in TABLES.items():
        if "{{"+k+"}}" in body: body=body.replace("{{"+k+"}}",fn())
    assert "{{" not in body, re.findall(r"\{\{.*?\}\}",body)
    to_docx("manuscript",body,lines=True); print("manuscript built")
if __name__=="__main__":
    what=sys.argv[1] if len(sys.argv)>1 else "ms"
    if what in ("ms","all"): build_ms()
    if what in ("supp","all"):
        import build2_supp; to_docx("supplementary",build2_supp.supp(),css=CSSL); print("supplementary built")
    if what.startswith("cover_") or what.startswith("note_"):
        s=open(MS+what+".md").read(); to_docx(what,md2html(s),css=CSS2); print(what,"built")
    if what in ("response","all"):
        import build2_supp; s=open(MS+"response.md").read(); body=md2html(s).replace("{{TR1}}",build2_supp.tr1()).replace("{{TR2}}",build2_supp.tr2()); to_docx("response",body,css=CSS2); print("response built")
    if what in ("highlights","all"):
        hl=[l[2:].strip() for l in open(MS+"highlights.md").read().splitlines() if l.startswith("- ")]; assert 3<=len(hl)<=5 and max(map(len,hl))<=85, [len(x) for x in hl]
        to_docx("highlights","<p><b>Highlights</b></p>\n"+"\n".join(f"<p>• {html.escape(x,quote=False)}</p>" for x in hl),css=CSS2,pages=False); print("highlights built")
