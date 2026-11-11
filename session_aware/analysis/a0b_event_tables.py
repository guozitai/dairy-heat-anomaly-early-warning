#!/usr/bin/env python3
"""A0b. Event tables of the herd-management system (one SQL dump per table) to parquet; records flagged as deleted are dropped."""
import os,re,sys,json
import pandas as pd
from multiprocessing import Pool
D='/home/guozitai/20260513-ningxianewdata/event_data/extracted_rar/事件数据'
OUT='/home/guozitai/20261014_event_layer/parquet'
os.makedirs(OUT,exist_ok=True)
Q=chr(39)
BT=chr(96)
def get_cols(path):
    cols=[]; intab=False
    pat=re.compile(r'\s*'+BT+r'([^'+BT+r']+)'+BT)
    with open(path,encoding='utf-8',errors='ignore') as f:
        for ln in f:
            if ln.startswith('CREATE TABLE'): intab=True; continue
            if intab:
                m=pat.match(ln)
                if m: cols.append(m.group(1))
                elif cols: break
    return cols
def parse_tuple(s):
    vals=[]; i=0; n=len(s); cur=''; inq=False
    while i<n:
        c=s[i]
        if inq:
            if c=='\\' and i+1<n: cur+=s[i+1]; i+=2; continue
            if c==Q: inq=False; i+=1; continue
            cur+=c; i+=1; continue
        if c==Q: inq=True; i+=1; continue
        if c==',': vals.append(cur.strip()); cur=''; i+=1; continue
        if c==')': vals.append(cur.strip()); return vals
        cur+=c; i+=1
    return vals
def build(fn):
    path=os.path.join(D,fn)
    tab=fn[fn.find('e_'):].replace('.sql','') if 'e_' in fn else fn.replace('.sql','')
    cols=get_cols(path)
    rows=[]; chunks=[]
    for ln in open(path,encoding='utf-8',errors='ignore'):
        if not ln.startswith('INSERT INTO'): continue
        j=ln.find('VALUES (')
        v=parse_tuple(ln[j+8:])
        if len(v)>=len(cols): rows.append(v[:len(cols)])
        if len(rows)>=200000:
            chunks.append(pd.DataFrame(rows,columns=cols)); rows=[]
    if rows: chunks.append(pd.DataFrame(rows,columns=cols))
    if not chunks: return (fn,0)
    df=pd.concat(chunks,ignore_index=True)
    df=df.replace({'NULL':None})
    if 'del_flag' in df.columns:
        df=df[df.del_flag!='1']
    df.to_parquet(os.path.join(OUT,tab+'.parquet'))
    return (fn,len(df))
if __name__=='__main__':
    files=[f for f in sorted(os.listdir(D)) if f.endswith('.sql')]
    with Pool(10) as p:
        for fn,n in p.imap_unordered(build,files):
            print(fn,n,flush=True)
    print('ALL DONE',flush=True)
