"""
SQL → Parquet 解析器
解析 8 个 mp_milkProductDetailInfo_*.sql,输出到 sql_parquet/
"""
import re
import pandas as pd
import os
from glob import glob

COLS = [
    'id', 'bullNumber', 'idenNumber', 'milkDate', 'milkingParlor',
    'shift', 'batch', 'platformNumber', 'idenDate',
    'milkingStart', 'milkingEnd', 'milkYield',
    'flowVelocity', 'flowRate', 'peakTime',
    'Slips', 'KickOffs', 'Blocks', 'NoOfReattaches',
    'AverageBlood', 'MaxBlood',
    'MaxConductivity', 'AvgConductivity', 'RelativeConductivity',
    'ManualMode', 'ManualDetach', 'ForcedRetract',
    'BloodAlarm', 'ConductivityAlarm',
    'TakeOffFlow', 'UpDateTime', 'extra1', 'extra2',
]

FARM_ID_MAP = {
    'FARM_8_schema': 3,    # FARM_8 (farm name redacted)
    'FARM_7_schema': 5,    # FARM_7 (farm name redacted)
    'FARM_1_schema': 6,    # FARM_1 (farm name redacted)
    'FARM_4_schema':  13,   # FARM_4 (farm name redacted)
    'FARM_5_schema': 14,   # FARM_5 (farm name redacted)
    'FARM_6_schema':  15,   # FARM_6 (farm name redacted)
    'FARM_3_schema':  16,   # FARM_3 (farm name redacted)
    'FARM_2_schema': 17,   # FARM_2 (farm name redacted)
}

def parse_value(v):
    v = v.strip()
    if v == 'NULL':
        return None
    if v.startswith("'") and v.endswith("'"):
        return v[1:-1]
    return v

def parse_sql_file(sql_path):
    fname = os.path.basename(sql_path)
    farm_id = None
    for key, fid in FARM_ID_MAP.items():
        if key in fname:
            farm_id = fid
            break
    if farm_id is None:
        print(f"  无法识别 farm_id: {fname}", flush=True)
        return None
    
    rows = []
    insert_pattern = re.compile(r"^INSERT INTO `[^`]+` VALUES \((.*)\);$")
    n_lines = 0
    
    with open(sql_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('INSERT'):
                continue
            n_lines += 1
            if n_lines % 200000 == 0:
                print(f"  已解析 {n_lines:,} 行...", flush=True)
            
            m = insert_pattern.match(line.rstrip('\n'))
            if not m:
                continue
            values_str = m.group(1)
            parts = re.findall(r"NULL|'[^']*'|[^,]+", values_str)
            if len(parts) != len(COLS):
                continue
            row = [parse_value(p) for p in parts]
            rows.append(row)
    
    df = pd.DataFrame(rows, columns=COLS)
    df['farm_id'] = farm_id
    
    # 类型转换
    for c in ['milkYield', 'flowVelocity', 'flowRate', 'RelativeConductivity', 'TakeOffFlow']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['Slips', 'KickOffs', 'Blocks', 'NoOfReattaches',
              'AverageBlood', 'MaxBlood', 'MaxConductivity', 'AvgConductivity']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['ManualMode', 'ManualDetach', 'ForcedRetract',
              'BloodAlarm', 'ConductivityAlarm']:
        df[c] = df[c].map({'True': True, 'False': False})
    
    df['milkDate'] = pd.to_datetime(df['milkDate'], errors='coerce')
    
    return df

if __name__ == '__main__':
    sql_dir = '/home/guozitai/20260513-ningxianewdata/sql_raw/'
    out_dir = '/home/guozitai/20260513-ningxianewdata/sql_parquet/'
    os.makedirs(out_dir, exist_ok=True)
    
    sql_files = sorted(glob(os.path.join(sql_dir, '*.sql')))
    print(f"找到 {len(sql_files)} 个 SQL 文件", flush=True)
    print(f"输出目录: {out_dir}\n", flush=True)
    
    for sql_path in sql_files:
        out_name = os.path.basename(sql_path).replace('.sql', '.parquet')
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            print(f"跳过(已存在): {out_name}\n", flush=True)
            continue
        
        print(f"=== {os.path.basename(sql_path)} ===", flush=True)
        df = parse_sql_file(sql_path)
        if df is not None:
            df.to_parquet(out_path, index=False)
            print(f"  → {out_name} ({len(df):,} 行)\n", flush=True)

    print("\n所有 SQL 解析完成!", flush=True)
