import pyarrow.parquet as pq, pandas as pd, sys
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
parts = []
for i in range(30):
    p = f"datasets/sajilck/malayalam-asr-corpus/data/train-{i:05d}-of-00030.parquet"
    with fs.open(p, block_size=1<<20) as f:
        t = pq.ParquetFile(f).read(columns=["sentence", "source"])
    parts.append(t.to_pandas()); print(i, len(parts[-1]), flush=True)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA
pd.concat(parts).to_csv(f"{DATA}/train_text.csv", index=False)
