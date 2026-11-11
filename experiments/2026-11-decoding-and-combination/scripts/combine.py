"""Hypothesis selection between systems (Whisper vs IndicConformer) using LM scores.
Selection rules:
  A) parameter-free: pick the hypothesis with the best per-character LM log-prob (char 8-gram, train+wiki avg)
  B) 2-fold cross-validated linear selector (weights fitted on one half of test, applied to the other):
     reported separately because no dev outputs exist for IndicConformer (running it here was not permitted).
usage: python combine.py <whisper_run_csv> <conformer_run_csv> <out_name>"""
import sys, itertools
import kenlm
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

wcsv, ccsv, out = sys.argv[1:4]
t = load_set("test231")
W = pd.read_csv(f"{RES}/{wcsv}.csv").hyp.fillna("").tolist()
C = pd.read_csv(f"{RES}/{ccsv}.csv").hyp.fillna("").tolist()
LM = {k: kenlm.Model(f"{ROOT}/lm/{k}.bin") for k in ("train_chr8", "wiki_chr8", "train_word4", "wiki_word4")}


def f(h):
    t_ = norm_ml(h)
    ch = " ".join(t_.replace(" ", "|"))
    nc, nw = max(len(t_), 1), max(len(t_.split()), 1)
    v = {k: m.score(ch if "chr" in k else t_) for k, m in LM.items()}
    return np.array([v["train_chr8"] / nc, v["wiki_chr8"] / nc, v["train_word4"] / nw, v["wiki_word4"] / nw, nw])


FW, FC = np.array([f(h) for h in W]), np.array([f(h) for h in C])
# A) parameter-free
pickA = (FC[:, 0] + FC[:, 1]) > (FW[:, 0] + FW[:, 1])
hypA = [c if p else w for w, c, p in zip(W, C, pickA)]
sA = save_run(f"{out}_charLM_paramfree", t, hypA, dict(n_conformer=int(pickA.sum())))
print("   delta vs whisper:", paired_delta(t.sentence, W, hypA), " vs conformer:", paired_delta(t.sentence, C, hypA))

# B) 2-fold CV: decision = bias + w . (FC - FW) > 0; grid search on fold, apply to other fold
D = FC - FW
grid = [0, 0.5, 1, 2]
rng = np.random.default_rng(0)
folds = rng.permutation(len(t)) % 2
hypB = list(W)
from common import _counts
errW = np.array([sum(_counts(r, h)[:3]) for r, h in zip(t.sentence, W)])
errC = np.array([sum(_counts(r, h)[:3]) for r, h in zip(t.sentence, C)])
for k in (0, 1):
    tr, te = folds != k, folds == k
    best = (1e9, None)
    for ws in itertools.product(grid, repeat=4):
        for b in (-0.5, -0.2, 0, 0.2, 0.5):
            sel = D[tr, :4] @ np.array(ws) + b > 0
            e = np.where(sel, errC[tr], errW[tr]).sum()
            if e < best[0]:
                best = (e, (ws, b))
    ws, b = best[1]
    sel = D[:, :4] @ np.array(ws) + b > 0
    for i in np.where(te)[0]:
        hypB[i] = C[i] if sel[i] else W[i]
    print(f"   fold {k}: weights {ws} bias {b}")
save_run(f"{out}_cv2", t, hypB)
print("   delta vs whisper:", paired_delta(t.sentence, W, hypB))
print("   ortho-normalised:", {k: round(v, 2) for k, v in score_ortho(t.sentence, hypA).items()}, "(param-free)")
