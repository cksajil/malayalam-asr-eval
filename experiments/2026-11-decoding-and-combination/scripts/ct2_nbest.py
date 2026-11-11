"""CTranslate2 decoding producing an N-best list per clip (text, normalised score, #tokens).
usage: python ct2_nbest.py <set> <model_dir> <beam> <nbest> <out_name> [preproc]"""
import sys, time, pickle
import ctranslate2
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from transformers import WhisperProcessor

setname, mdir, beam, nb, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
pre = sys.argv[6] if len(sys.argv) > 6 else "none"
df = load_set(setname)
proc = WhisperProcessor.from_pretrained(MODEL_ID)
tok = proc.tokenizer
prompt = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ml|>", "<|transcribe|>", "<|notimestamps|>"])
model = ctranslate2.models.Whisper(f"{ROOT}/models/{mdir}", device="cpu", compute_type="default", intra_threads=4)


def preprocess(w):
    if pre == "none":
        return w
    import librosa
    if "trim" in pre:
        w, _ = librosa.effects.trim(w, top_db=35)
        w = np.concatenate([np.zeros(1600, np.float32), w, np.zeros(1600, np.float32)])
    if "loud" in pre:
        w = w / (np.sqrt(np.mean(w ** 2)) + 1e-8) * 0.05  # -26 dBFS RMS
        w = np.clip(w, -1, 1)
    return w.astype(np.float32)


order = np.argsort(df.dur.values)
res = [None] * len(df)
bs = 4
t0 = time.time()
for i in range(0, len(df), bs):
    ids = order[i:i + bs]
    feats = proc.feature_extractor([preprocess(df.wav.iloc[j]) for j in ids], sampling_rate=SR, return_tensors="np").input_features
    out = model.generate(ctranslate2.StorageView.from_array(np.ascontiguousarray(feats.astype(np.float32))),
                         [prompt] * len(ids), beam_size=beam, num_hypotheses=nb, return_scores=True,
                         max_length=888, length_penalty=1.0, repetition_penalty=1.0, suppress_blank=True)
    for j, r in zip(ids, out):
        res[j] = [(tok.decode(s, skip_special_tokens=True).strip(), sc, len(s)) for s, sc in zip(r.sequences_ids, r.scores)]
    print(f"{i+len(ids)}/{len(df)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
secs = time.time() - t0
pickle.dump(res, open(f"{RES}/{name}.nbest.pkl", "wb"))
ntok = [r[0][2] for r in res]
print("max tokens", max(ntok), "clips >= 440 tokens:", sum(t >= 440 for t in ntok))
save_run(name, df, [r[0][0] for r in res], dict(secs=secs, rtf=secs / df.dur.sum()))
