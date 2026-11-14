#!/usr/bin/env python3
"""投稿清单.md -> docx (宋体, 三线表)."""
import re, html, subprocess
MS='/work/cea_sat/rescue/ms/'; src=open(MS+"投稿清单.md").read().strip().split("\n"); out=[]; i=0
def inline(t): t=html.escape(t,quote=False); t=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",t); t=re.sub(r"`(.+?)`",r"\1",t); return t
while i<len(src):
    l=src[i]
    if l.startswith("# "): out.append(f"<h1>{inline(l[2:])}</h1>")
    elif l.startswith("## "): out.append(f"<h2>{inline(l[3:])}</h2>")
    elif l.startswith("|"):
        rows=[]
        while i<len(src) and src[i].startswith("|"):
            if not re.match(r"^\|[-| ]+\|$",src[i]): rows.append([c.strip() for c in src[i].strip("|").split("|")])
            i+=1
        t=['<table width="100%" cellspacing="0" cellpadding="4" style="border-collapse:collapse">']
        for k,r in enumerate(rows):
            st="padding:3pt 5pt;font-size:10.5pt;"+("border-top:1.5pt solid #000;border-bottom:0.75pt solid #000;font-weight:bold;" if k==0 else "")+("border-bottom:1.5pt solid #000;" if k==len(rows)-1 else "")
            t.append("<tr>"+"".join(f'<td style="{st}">{inline(c)}</td>' for c in r)+"</tr>")
        out.append("\n".join(t)+"</table>"); continue
    elif l.strip(): out.append(f"<p>{inline(re.sub(r'^- ','',l))}</p>")
    i+=1
CSS='@page{size:21cm 29.7cm;margin:2.2cm}body{font-family:"SimSun","宋体","Noto Serif CJK SC",serif;font-size:11pt;line-height:1.6;margin:0}h1{font-size:15pt}h2{font-size:12.5pt;margin-top:14pt}p{margin-top:5pt;margin-bottom:5pt}'
open(MS+"投稿清单.html","w").write(f'<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>'+"\n".join(out)+"</body></html>")
subprocess.run(["soffice","--headless","--convert-to",'docx:MS Word 2007 XML',"--outdir",MS,MS+"投稿清单.html"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
import sys; sys.path.insert(0,MS); from docxfix import fix_docx; fix_docx(MS+"投稿清单.docx",latin="Times New Roman",cjk="SimSun"); print("checklist built")
