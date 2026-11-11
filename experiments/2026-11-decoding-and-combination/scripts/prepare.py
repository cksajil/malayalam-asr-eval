"""Build test231 (identical to malayalam-asr-eval balanced231) and a disjoint, leakage-free dev set
(30 clips/source from the remaining clean pool) used ONLY for tuning hyperparameters."""
import sys, gc
import pandas as pd
from huggingface_hub import hf_hub_download
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

paths = [hf_hub_download(DATASET_ID, f"data/test-0000{i}-of-00002.parquet", repo_type="dataset") for i in (0, 1)]
test = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
clean = pd.read_csv(f"{REPO}/own_corpus_test_clean.csv")
keys = set(clean["text"].map(match_key))
pool = test[test["sentence"].map(match_key).isin(keys)].copy()
pool["hf_row"] = pool.index
print("clean pool", len(pool))
del test; gc.collect()

t231 = pd.concat([g.sample(min(len(g), 50), random_state=42) for _, g in pool.groupby("source", sort=True)]).reset_index(drop=True)
man = pd.read_csv(f"{REPO}/results/balanced231/manifest.csv")
assert list(t231.hf_row) == list(man.hf_row), "subset mismatch with published manifest"
rest = pool[~pool.hf_row.isin(t231.hf_row)]
# dev must not share sentences with test231 either
rest = rest[~rest["sentence"].map(match_key).isin(set(t231.sentence.map(match_key)))]
dev = pd.concat([g.sample(min(len(g), 30), random_state=7) for _, g in rest.groupby("source", sort=True)]).reset_index(drop=True)
for name, d in (("test231", t231), ("dev", dev)):
    d["wav"] = d["audio"].map(decode_audio)
    d["dur"] = d["wav"].map(lambda w: len(w) / SR)
    d[["hf_row", "source", "sentence", "dur", "wav"]].to_pickle(f"{DATA}/{name}.pkl")
    print(name, len(d), f"{d.dur.sum()/60:.1f} min", d.source.value_counts().to_dict())
