"""Shared helpers: data prep (same selection as malayalam-asr-eval) and scoring (same normalisation)."""
import io, os, re, sys, unicodedata, json
import numpy as np, pandas as pd, jiwer

from pathlib import Path

# Working files (decoded audio, KenLM models, converted models, run outputs) go under EXP_DIR, default the
# experiment folder; data/, lm/, models/ and *.pkl are git-ignored there. REPO is the malayalam-asr-eval root.
HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("EXP_DIR", str(HERE.parent))
REPO = str(HERE.parents[2])
DATA = f"{ROOT}/data"
RES = f"{ROOT}/runs"
os.makedirs(DATA, exist_ok=True)
os.makedirs(RES, exist_ok=True)
DATASET_ID = "sajilck/malayalam-asr-corpus"
MODEL_ID = "sajilck/whisper-small-malayalam-v2"
SR = 16000

CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ", "ല്‍": "ൽ", "ള്‍": "ൾ", "ക്‍": "ൿ"}


def norm_raw(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def norm_ml(s):
    s = unicodedata.normalize("NFC", str(s))
    for k, v in CHILLU.items():
        s = s.replace(k, v)
    s = s.replace("‍", "").replace("‌", "")
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


match_key = norm_raw


def decode_audio(a):
    import soundfile as sf, librosa
    b = a["bytes"] if isinstance(a, dict) else a
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    except Exception:
        wav, sr = librosa.load(io.BytesIO(b), sr=None, mono=True)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_set(name):
    return pd.read_pickle(f"{DATA}/{name}.pkl")


# ---------------- scoring ----------------
def _counts(ref, hyp):
    r, h = norm_ml(ref), norm_ml(hyp)
    o = jiwer.process_words(r, h) if r else None
    S, D, I, N = (o.substitutions, o.deletions, o.insertions, o.substitutions + o.deletions + o.hits) if o else (0, 0, len(h.split()), 0)
    rc, hc = r.replace(" ", ""), h.replace(" ", "")
    oc = jiwer.process_characters(rc, hc) if rc else None
    cE = (oc.substitutions + oc.deletions + oc.insertions) if oc else len(hc)
    return S, D, I, N, cE, len(rc)


def score(refs, hyps, sources=None, boot=True):
    c = np.array([_counts(r, h) for r, h in zip(refs, hyps)], dtype=float)
    e, n = c[:, :3].sum(1), c[:, 3]
    out = dict(wer=100 * e.sum() / n.sum(), cer=100 * c[:, 4].sum() / c[:, 5].sum(),
               sub=100 * c[:, 0].sum() / n.sum(), dele=100 * c[:, 1].sum() / n.sum(), ins=100 * c[:, 2].sum() / n.sum(),
               exact=100 * float(np.mean(e == 0)))
    if boot:
        r = np.random.default_rng(42)
        idx = r.integers(0, len(c), (1000, len(c)))
        w = e[idx].sum(1) / n[idx].sum(1)
        out["ci"] = [round(100 * x, 1) for x in np.percentile(w, [2.5, 97.5])]
    if sources is not None:
        src = np.asarray(sources)
        out["per_source"] = {s: [round(100 * e[src == s].sum() / n[src == s].sum(), 1),
                                 round(100 * c[src == s, 4].sum() / c[src == s, 5].sum(), 1)] for s in sorted(set(src))}
    return out


def paired_delta(refs, hyp_a, hyp_b, B=2000):
    """Bootstrap 95% CI of WER(b) - WER(a), paired over utterances."""
    a = np.array([_counts(r, h) for r, h in zip(refs, hyp_a)], dtype=float)
    b = np.array([_counts(r, h) for r, h in zip(refs, hyp_b)], dtype=float)
    ea, eb, n = a[:, :3].sum(1), b[:, :3].sum(1), a[:, 3]
    idx = np.random.default_rng(0).integers(0, len(a), (B, len(a)))
    d = 100 * (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
    return round(100 * (eb.sum() - ea.sum()) / n.sum(), 2), [round(x, 2) for x in np.percentile(d, [2.5, 97.5])]


def fmt(name, s):
    ci = f" ({s['ci'][0]}–{s['ci'][1]})" if "ci" in s else ""
    return f"{name:40s} WER {s['wer']:5.2f}{ci}  CER {s['cer']:4.2f}  S/D/I {s['sub']:.1f}/{s['dele']:.1f}/{s['ins']:.1f}  exact {s['exact']:.1f}%"


def save_run(name, df, hyps, extra=None):
    out = df[["source", "sentence"]].copy()
    out["hyp"] = hyps
    out.to_csv(f"{RES}/{name}.csv", index=False)
    s = score(out.sentence, out.hyp, out.source)
    if extra:
        s.update(extra)
    json.dump(s, open(f"{RES}/{name}.json", "w"), ensure_ascii=False, indent=1)
    print(fmt(name, s), flush=True)
    print("   per source:", s["per_source"], flush=True)
    return s


# ---- secondary metric: orthographic-variant normalisation (reported separately, never as the headline) ----
# word-final consonant+virama for ണ ര ല ള -> atomic chillu (pre-Unicode-5.1 spelling of the same word).
# ന് is deliberately NOT mapped: word-final ന് is often the dative suffix (അവന് "to him" vs അവൻ "he").
_WF = {"ണ്": "ൺ", "ര്": "ർ", "ല്": "ൽ", "ള്": "ൾ"}


def norm_ortho(s):
    s = norm_ml(s).replace("ൌ", "ൗ")  # ൌ (AU sign) -> ൗ (AU length mark): same vowel
    return " ".join(next((w[:-2] + v for k, v in _WF.items() if w.endswith(k)), w) for w in s.split())


def score_ortho(refs, hyps):
    r = [norm_ortho(x) for x in refs]; h = [norm_ortho(x) for x in hyps]
    o = jiwer.process_words(r, h)
    oc = jiwer.process_characters([x.replace(" ", "") for x in r], [x.replace(" ", "") for x in h])
    return dict(wer=100 * o.wer, cer=100 * oc.cer)
