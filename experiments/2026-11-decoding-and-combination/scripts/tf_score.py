"""Teacher-forced Whisper log-prob for every candidate (N-best + optional extra system hypotheses).
Gives one acoustic score defined the same way for hypotheses from any system.
usage: python tf_score.py <set> <nbest_name> <out_name> [extra_csv ...]"""
import sys, pickle, time, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from transformers import WhisperProcessor, WhisperForConditionalGeneration

torch.set_num_threads(int(os.environ.get("THREADS", "4")))
setname, nbname, out = sys.argv[1:4]
extras = [pd.read_csv(f"{RES}/{x}.csv").hyp.fillna("").tolist() for x in sys.argv[4:]]
df = load_set(setname)
nb = pickle.load(open(f"{RES}/{nbname}.nbest.pkl", "rb"))
proc = WhisperProcessor.from_pretrained(MODEL_ID)
tok = proc.tokenizer
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).eval()
prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", "<|ml|>", "<|transcribe|>", "<|notimestamps|>"])
eot = tok.eos_token_id
res = []
t0 = time.time()
for i in range(len(df)):
    cands, src = [], []
    for h, _, _ in nb[i]:
        cands.append(h); src.append("whisper")
    for k, e in enumerate(extras):
        cands.append(e[i]); src.append(f"extra{k}")
    seqs = [prefix + tok.encode(c, add_special_tokens=False)[:440] + [eot] for c in cands]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), eot)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
    feats = proc.feature_extractor(df.wav.iloc[i], sampling_rate=SR, return_tensors="pt").input_features
    with torch.inference_mode():
        enc = model.model.encoder(feats).last_hidden_state.expand(len(seqs), -1, -1)
        logits = model(encoder_outputs=(enc,), decoder_input_ids=ids[:, :-1]).logits
        lp = torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
    row = []
    for j, s in enumerate(seqs):
        n = len(s) - len(prefix)  # text tokens + eot
        row.append(dict(text=cands[j], src=src[j], am=float(lp[j, len(prefix) - 1: len(s) - 1].sum()), ntok=n))
    res.append(row)
    if i % 20 == 0:
        print(i, f"{time.time()-t0:.0f}s", file=sys.stderr, flush=True)
pickle.dump(res, open(f"{RES}/{out}.tf.pkl", "wb"))
print("done", out, f"{time.time()-t0:.0f}s")
