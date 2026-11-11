"""Which character LM picks between Whisper and IndicConformer best, and what the deployed configuration
(Whisper int8 + IndicConformer + the pruned 5-gram read by asr_select.py) scores. No tuned parameters:
keep the transcript with the higher per-character log-probability."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
import kenlm, asr_select

t = load_set("test231"); ref = t.sentence
C = pd.read_csv(f"{RES}/pub_indicconformer-ctc_test.csv").hyp.fillna("").tolist()
LM = {k: kenlm.Model(f"{ROOT}/lm/{k}.bin") for k in ["train_chr8", "wiki_chr8", "both_chr5p"]}
pc = lambda m, h: m.score(" ".join(norm_ml(h).replace(" ", "|"))) / max(len(norm_ml(h)), 1)
W = pd.read_csv(f"{RES}/test231_ct2_greedy.csv").hyp.fillna("").tolist()
for name, ks in {"train8+wiki8": ["train_chr8", "wiki_chr8"], "train8": ["train_chr8"], "wiki8": ["wiki_chr8"],
                 "both5 pruned": ["both_chr5p"]}.items():
    h = [c if sum(pc(LM[k], c) for k in ks) > sum(pc(LM[k], w) for k in ks) else w for w, c in zip(W, C)]
    s = score(ref, h, boot=False)
    print(f"{name:14s} WER {s['wer']:.2f} CER {s['cer']:.2f}  delta vs Whisper {paired_delta(ref, W, h)}")
lm = asr_select.ArpaLM(f"{ROOT}/lm/ml_char5_pruned.arpa.gz")           # pure Python, as on the Space
for wname in ["test231_ct2_greedy", "pub_fw-v2-int8_test"]:
    Wn = pd.read_csv(f"{RES}/{wname}.csv").hyp.fillna("").tolist()
    h = [[w, c][asr_select.select(lm, [w, c])] for w, c in zip(Wn, C)]
    save_run(f"space_{wname}_sel5", t, h)
    print("   delta vs Whisper alone:", paired_delta(ref, Wn, h))
