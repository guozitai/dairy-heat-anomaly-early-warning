"""
parse_e_calving.py
==================
解析产犊e_calving.sql → e_calving.parquet
然后:
  1. 为每个 (farm_id, bullNumber, milkDate) 算出 days_in_milk (DIM)
  2. 加进 unified_daily_enriched.parquet (v2)
"""
import pandas as pd
import re
import time
from pathlib import Path

t = time.time()

SQL_PATH = '${DATA_ROOT}/raw/event_data/事件数据/产犊e_calving.sql'
OUT_PARQUET = '${DATA_ROOT}/derived/e_calving.parquet'

print(f"[{time.time()-t:.0f}s] 读 SQL...", flush=True)
with open(SQL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
print(f"  文件大小: {len(content)/1024/1024:.1f} MB chars", flush=True)

# 找所有 INSERT INTO `e_calving` VALUES (...) 元组
# 用一个稳健的正则,匹配每个 VALUES 后面的括号组
# 形式: INSERT INTO `e_calving` VALUES (X, Y, 'Z', ...);
print(f"[{time.time()-t:.0f}s] 解析 INSERT...", flush=True)

# 抓所有 VALUES (xxx); 的内容
# 注意: VALUES 后可能多行 (Navicat 可能批量),但前 3 条看起来每个 INSERT 是单独一行
pattern = re.compile(r"VALUES\s*\(([^;]+?)\);", re.DOTALL)
matches = pattern.findall(content)
print(f"  匹配到 {len(matches):,} 条 INSERT", flush=True)

if len(matches) == 0:
    print("WARN: 没找到 INSERT, 检查正则", flush=True)
    raise SystemExit(1)

# 解析每行 VALUES 元组
# 字段顺序 (从 INSERT 推断):
#   id, farm_id, bullNumber, pen, parity_decimal, parity, calf_sex, 
#   calvingDate, midwife, operator, created_at, ..., note, source, deleted, other
print(f"[{time.time()-t:.0f}s] 解析每条记录...", flush=True)

records = []
fail = 0
for i, m in enumerate(matches):
    # 简单 split by comma, 但要保留 '...' 内部的逗号
    # 用 csv module 解析每行
    import csv
    import io
    reader = csv.reader(io.StringIO(m), quotechar="'", skipinitialspace=True)
    try:
        fields = next(reader)
        if len(fields) < 8:
            fail += 1
            continue
        records.append({
            'id': fields[0].strip(),
            'farm_id': fields[1].strip(),
            'bullNumber': fields[2].strip().strip("'"),
            'pen': fields[3].strip().strip("'"),
            'parity_decimal': fields[4].strip(),
            'parity': fields[5].strip(),
            'calf_sex': fields[6].strip().strip("'"),
            'calvingDate': fields[7].strip().strip("'"),
        })
    except Exception as e:
        fail += 1
        if fail < 5:
            print(f"  解析失败 row {i}: {e}", flush=True)
    
    if (i+1) % 100000 == 0:
        print(f"  已解析 {i+1:,}", flush=True)

print(f"\n[{time.time()-t:.0f}s] 解析完成", flush=True)
print(f"  成功: {len(records):,}", flush=True)
print(f"  失败: {fail:,}", flush=True)

df = pd.DataFrame(records)
df['farm_id'] = pd.to_numeric(df['farm_id'], errors='coerce').astype('Int64')
df['bullNumber'] = df['bullNumber'].astype(str)
df['calvingDate'] = pd.to_datetime(df['calvingDate'], errors='coerce')
df['parity'] = pd.to_numeric(df['parity'], errors='coerce').astype('Int64')

# 过滤
df = df.dropna(subset=['farm_id', 'calvingDate'])

print(f"\n[{time.time()-t:.0f}s] 数据概况:", flush=True)
print(f"  总产犊记录: {len(df):,}")
print(f"  涉及 farm: {sorted(df['farm_id'].unique())}")
print(f"  时间范围: {df['calvingDate'].min()} → {df['calvingDate'].max()}")
print(f"  涉及独立牛: {df['bullNumber'].nunique():,}")

print(f"\n  按 farm 分布:")
g = df.groupby('farm_id').agg(
    n_records=('id', 'count'),
    n_cows=('bullNumber', 'nunique'),
    first_calving=('calvingDate', 'min'),
    last_calving=('calvingDate', 'max'),
)
print(g.to_string())

# 保存 e_calving.parquet
df.to_parquet(OUT_PARQUET, index=False, compression='snappy')
print(f"\n[{time.time()-t:.0f}s] ✅ 保存: {OUT_PARQUET}", flush=True)
print(f"⏱  用时: {time.time()-t:.0f} 秒")
