"""
4 场反刍批量解析 (FARM_7/13/14/15), 用 FARM_1 验证过的软匹配 schema
"""
import pandas as pd
import os, glob, time
from multiprocessing import Pool

BASE = "${DATA_ROOT}/raw/jibuqi/计步器数据"
OUT_DIR = "${DATA_ROOT}/derived"

# 场名 → 子目录 + farm_id 映射
FARM_DIRS = {
    5:  ("3.奶五SCR历史数据20200101-2024.6.14",   "FARM_7"),
    13: ("(SCR data folder name redacted)", "FARM_4"),
    14: ("(SCR data folder name redacted)", "FARM_5"),
    15: ("(SCR data folder name redacted)",      "FARM_6"),
}

FIELD_MAP = {
    'allflex_id':       ['Allflex项圈号'],
    'bullNumber':       ['牛号'],
    'date':             ['日期'],
    'hour':             ['时间'],
    'pen':              ['当前组别'],
    'parity_state':     ['胎次状态'],
    'lactation_period': ['泌乳期'],
    'DIM_rumination':   ['泌乳天数/日龄', '当前泌乳天数'],
    'last_calving':     ['最近分娩日期'],
    'daily_rumination': ['日反刍量'],
    'daily_activity':   ['日活动量'],
    'birth_date':       ['出生日期'],
    'culled':           ['淘汰的'],
}

def parse_one(fp):
    try:
        df = pd.read_excel(fp, engine='openpyxl')
        out = {}
        for target, candidates in FIELD_MAP.items():
            for cand in candidates:
                if cand in df.columns:
                    out[target] = df[cand]; break
            if target not in out:
                out[target] = None
        result = pd.DataFrame(out)
        result = result[result['hour']=='0:00'].copy()
        result = result.drop(columns=['hour'])
        result['bullNumber'] = result['bullNumber'].astype(str)
        result['allflex_id'] = result['allflex_id'].astype(str)
        if 'pen' in result.columns:
            result['pen'] = result['pen'].astype(str)
        result['date'] = pd.to_datetime(result['date'], errors='coerce')
        result['last_calving'] = pd.to_datetime(result['last_calving'], errors='coerce')
        result['birth_date'] = pd.to_datetime(result['birth_date'], errors='coerce')
        result = result[result['date'].notna() & (result['date'] > '2010-01-01')]
        if 'culled' in result.columns and result['culled'].notna().any():
            result['culled_flag'] = result['culled']=='是'
        else:
            result['culled_flag'] = False
        if 'culled' in result.columns:
            result = result.drop(columns=['culled'])
        result['source_file'] = os.path.basename(fp)
        return result
    except Exception as e:
        return pd.DataFrame({'_error':[f'{os.path.basename(fp)}: {str(e)[:120]}']})


if __name__ == '__main__':
    for fid, (subdir, fname) in FARM_DIRS.items():
        fdir = os.path.join(BASE, subdir)
        files = sorted(glob.glob(os.path.join(fdir, '**/*.xlsx'), recursive=True))
        # 排除 ~$ 临时文件
        files = [f for f in files if not os.path.basename(f).startswith('~$')]
        out_path = os.path.join(OUT_DIR, f'{fname}_rumination_daily.parquet')
        
        print(f'\n{"="*65}\n=== {fname} ({len(files)} xlsx) ===\n{"="*65}')
        if len(files) == 0:
            print(f'  目录无 xlsx,跳过: {fdir}'); continue
        
        t0 = time.time()
        with Pool(processes=85) as pool:
            results = []
            for i, df in enumerate(pool.imap_unordered(parse_one, files), 1):
                results.append(df)
                if i % 30 == 0 or i == len(files):
                    elapsed = time.time() - t0
                    eta = elapsed / i * (len(files) - i)
                    print(f'  {i}/{len(files)} done, {elapsed/60:.1f}min used, ~{eta/60:.1f}min left')
        
        valid = [d for d in results if '_error' not in d.columns]
        errors = [d for d in results if '_error' in d.columns]
        print(f'\n  成功: {len(valid)}/{len(files)}, 失败: {len(errors)}')
        if errors:
            for e in errors[:3]: print(f'    {e["_error"].iloc[0][:100]}')
        
        if not valid:
            print('  全部失败,跳过'); continue
        
        big = pd.concat(valid, ignore_index=True)
        n_before = len(big)
        big = big.sort_values('source_file').drop_duplicates(['bullNumber','date'], keep='last').reset_index(drop=True)
        
        print(f'\n  raw: {n_before:,} → dedupe: {len(big):,}')
        print(f'  独立 cows: {big["bullNumber"].nunique():,}')
        print(f'  日期: {big["date"].min()} → {big["date"].max()}')
        print(f'  rum NA: {big["daily_rumination"].isna().mean()*100:.1f}%, act NA: {big["daily_activity"].isna().mean()*100:.1f}%')
        
        big['year'] = big['date'].dt.year
        print(f'  按年:')
        print(big.groupby('year').agg(rows=('bullNumber','size'),cows=('bullNumber','nunique')).to_string())
        
        big.to_parquet(out_path, index=False)
        print(f'  保存: {out_path} ({(time.time()-t0)/60:.1f}min)')
    
    print(f'\n\n{"="*65}\n4 场全部完成\n{"="*65}')
