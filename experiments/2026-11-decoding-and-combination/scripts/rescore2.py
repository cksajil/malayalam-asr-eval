"""Log-linear rescoring of candidate pools: score = AM(teacher-forced Whisper logprob) + sum_k w_k * LM_k + b * #words.
Weights are tuned on dev (Whisper N-best only) and applied unchanged to test.
usage: python rescore2.py <dev_tf> <test_tf> <out_prefix> [--drop-extra]"""
import sys, pickle, itertools, json
import kenlm
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from common import _counts

dev_tf, test_tf, out = sys.argv[1:4]
drop_extra = "--drop-extra" in sys.argv
LMK = ("train_word4", "wiki_word4", "train_chr8", "wiki_chr8")
LMS = {k: kenlm.Model(f"{ROOT}/lm/{k}.bin") for k in LMK}
LN10 = np.log(10)


def build(setname, tfname):
    df = load_set(setname)
    pools = pickle.load(open(f"{RES}/{tfname}.tf.pkl", "rb"))
    P = []
    for r, pool in zip(df.sentence, pools):
        if drop_extra:
            pool = [c for c in pool if c["src"] == "whisper"]
        rows = []
        for c in pool:
            t = norm_ml(c["text"])
            ch = " ".join(t.replace(" ", "|"))
            f = [c["am"]] + [LN10 * LMS[k].score(ch if "chr" in k else t) for k in LMK] + [len(t.split())]
            rows.append((c["text"], c["src"], np.array(f), sum(_counts(r, c["text"])[:3])))
        P.append(rows)
    N = sum(_counts(r, "")[3] for r in df.sentence)
    return df, P, N


def choose(P, w):
    return [max(rows, key=lambda x: x[2] @ w) for rows in P]


def err(P, w, N):
    return 100 * sum(x[3] for x in choose(P, w)) / N


ddf, dP, dN = build("dev", dev_tf)
tdf, tP, tN = build("test231", test_tf)
base = np.array([1, 0, 0, 0, 0, 0.0])
print(f"dev  AM-only {err(dP, base, dN):.2f}  oracle {100*sum(min(x[3] for x in r) for r in dP)/dN:.2f}")
print(f"test AM-only {err(tP, base, tN):.2f}  oracle {100*sum(min(x[3] for x in r) for r in tP)/tN:.2f}")
grid = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
configs = {"word_train": [1], "word_wiki": [2], "chr_train": [3], "chr_wiki": [4], "chr_both": [3, 4],
           "word+chr_train": [1, 3], "all4": [1, 2, 3, 4]}
summary = {}
for name, idx in configs.items():
    best = (1e9, None)
    for ws in itertools.product(grid, repeat=len(idx)):
        for b in (-2, -1, 0, 1, 2, 4):
            w = base.copy(); w[5] = b
            for i, v in zip(idx, ws):
                w[i] = v
            e = err(dP, w, dN)
            if e < best[0] - 1e-9:
                best = (e, w)
    hyps = [x[0] for x in choose(tP, best[1])]
    s = save_run(f"{out}_{name}", tdf, hyps, dict(weights=best[1].tolist(), dev_wer=best[0]))
    nsrc = pd.Series([x[1] for x in choose(tP, best[1])]).value_counts().to_dict()
    summary[name] = dict(dev=round(best[0], 2), test=round(s["wer"], 2), cer=round(s["cer"], 2),
                         w=[round(v, 2) for v in best[1]], picked=nsrc)
    print("  ", name, summary[name], flush=True)
json.dump(summary, open(f"{RES}/{out}_summary.json", "w"), indent=1)
