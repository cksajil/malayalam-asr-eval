"""WER / CER scoring — identical to the published v1/v2 clean-eval notebooks.

Corpus WER = sum(S + D + I) / sum(N) over utterances, after `norm_ml`.
CER is computed on normalised text with spaces removed.
95% CI: utterance-level bootstrap, B=1000, fixed seed (reproducible).
"""
from __future__ import annotations

import jiwer
import numpy as np
import pandas as pd

from .normalize import norm_ml, norm_raw

SEED = 42


def utt_counts(ref: str, hyp: str) -> dict:
    if not ref.strip():
        return dict(S=0, D=0, I=len(hyp.split()), N=0)
    o = jiwer.process_words(ref, hyp if hyp.strip() else "")
    return dict(S=o.substitutions, D=o.deletions, I=o.insertions, N=o.substitutions + o.deletions + o.hits)


def char_counts(ref: str, hyp: str) -> dict:
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r:
        return dict(cE=len(h), cN=0)
    o = jiwer.process_characters(r, h if h else "")
    return dict(cE=o.substitutions + o.deletions + o.insertions, cN=len(r))


def is_loop(text: str, min_rep: int = 4) -> bool:
    """True if any 1-4 word n-gram repeats `min_rep` times in a row (hallucination loop)."""
    w = text.split()
    for n in range(1, 5):
        for i in range(0, max(0, len(w) - n * min_rep + 1)):
            g = w[i:i + n]
            if all(w[i + k * n: i + (k + 1) * n] == g for k in range(min_rep)):
                return True
    return False


def score(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-utterance error counts to a frame with `ref` and `hyp` columns."""
    df = df.copy().reset_index(drop=True)
    df["hyp"] = df["hyp"].fillna("").astype(str)
    for tag, f in (("raw", norm_raw), ("norm", norm_ml)):
        c = pd.DataFrame([utt_counts(f(r), f(h)) for r, h in zip(df.ref, df.hyp)]).add_prefix(f"{tag}_")
        df = pd.concat([df, c], axis=1)
    cc = pd.DataFrame([char_counts(norm_ml(r), norm_ml(h)) for r, h in zip(df.ref, df.hyp)])
    df = pd.concat([df, cc], axis=1)
    df["wer_utt"] = (df.norm_S + df.norm_D + df.norm_I) / df.norm_N.clip(lower=1)
    df["is_empty"] = df.hyp.map(lambda s: not norm_ml(s))
    df["loop"] = df.hyp.map(lambda s: is_loop(norm_ml(s)))
    df["len_ratio"] = (df.hyp.map(lambda s: len(norm_ml(s).split()))
                       / df.ref.map(lambda s: len(norm_ml(s).split())).clip(lower=1))
    return df


def corpus_wer(df: pd.DataFrame, tag: str = "norm") -> float:
    return float((df[f"{tag}_S"] + df[f"{tag}_D"] + df[f"{tag}_I"]).sum() / df[f"{tag}_N"].sum())


def corpus_cer(df: pd.DataFrame) -> float:
    return float(df.cE.sum() / df.cN.sum())


def boot_ci(df: pd.DataFrame, B: int = 1000, seed: int = SEED) -> tuple[float, float]:
    """95% bootstrap CI of corpus WER. Fresh generator per call, so every table agrees."""
    r = np.random.default_rng(seed)
    e = (df.norm_S + df.norm_D + df.norm_I).values
    n = df.norm_N.values
    idx = r.integers(0, len(df), (B, len(df)))
    w = e[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(w, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(df: pd.DataFrame) -> dict:
    """Accuracy + speed summary for one scored run (expects total_s and dur columns for speed)."""
    lo, hi = boot_ci(df)
    sdi = df[["norm_S", "norm_D", "norm_I"]].sum() / df.norm_N.sum()
    out = {
        "n_clips": int(len(df)),
        "audio_min": round(float(df.dur.sum()) / 60, 2),
        "WER": corpus_wer(df), "WER_CI95": [lo, hi], "WER_raw": corpus_wer(df, "raw"), "CER": corpus_cer(df),
        "sub": float(sdi.norm_S), "del": float(sdi.norm_D), "ins": float(sdi.norm_I),
        "exact_match": float((df.wer_utt == 0).mean()),
        "empty": int(df.is_empty.sum()), "loops": int(df.loop.sum()),
    }
    if "total_s" in df:
        rtf = float(df.total_s.sum() / df.dur.sum())
        out.update({"RTF": rtf, "RTFx": 1 / rtf if rtf else None,
                    "latency_p50_s": float(df.total_s.quantile(.5)),
                    "latency_p90_s": float(df.total_s.quantile(.9)),
                    "latency_max_s": float(df.total_s.max())})
    if "near_cap" in df:
        out["near_cap"] = int(df.near_cap.sum())
    return out


def per_source(df: pd.DataFrame) -> dict:
    res = {}
    for src, g in df.groupby("source"):
        d = {"n": int(len(g)), "WER": corpus_wer(g), "CER": corpus_cer(g)}
        if len(g) >= 20:
            d["WER_CI95"] = list(boot_ci(g))
        if "total_s" in g:
            d["RTF"] = float(g.total_s.sum() / g.dur.sum())
        res[src] = d
    return res
