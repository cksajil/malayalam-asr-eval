"""Transformers decoding (FP32) with configurable generation settings. Batched for CPU throughput.
usage: python hf_decode.py <set> <run_name> '<json generate kwargs>' [model_id]"""
import sys, json, time, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration

torch.set_num_threads(4)
setname, name = sys.argv[1], sys.argv[2]
gk = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mid = sys.argv[4] if len(sys.argv) > 4 else MODEL_ID
bs = gk.pop("batch", 8)
df = load_set(setname)
proc = WhisperProcessor.from_pretrained(mid)
model = WhisperForConditionalGeneration.from_pretrained(mid).eval()
kw = dict(language="malayalam", task="transcribe", max_new_tokens=444)
kw.update(gk)
order = np.argsort(df.dur.values)  # sort by length so batches finish together
hyps = [None] * len(df)
t0 = time.time()
for i in range(0, len(df), bs):
    ids = order[i:i + bs]
    feats = proc.feature_extractor([df.wav.iloc[j] for j in ids], sampling_rate=SR, return_tensors="pt").input_features
    with torch.inference_mode():
        out = model.generate(feats, **kw)
    for j, t in zip(ids, proc.batch_decode(out, skip_special_tokens=True)):
        hyps[j] = t.strip()
    print(f"{i+len(ids)}/{len(df)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
save_run(name, df, hyps, dict(gen=kw, secs=time.time() - t0))
