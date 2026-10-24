"""Build the evaluation set: HF test split -> leakage-free rows -> subset -> decoded 16 kHz audio.

The 231-clip headline subset is 50 clips per source (all clips for sources with fewer),
sampled with seed 42 from the clean pool. This reproduces the clips behind the published
v1 (43.5% WER) and v2 (31.2% WER) numbers.
"""
from __future__ import annotations

import gc
import glob
import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .normalize import match_key

DATASET_ID = "sajilck/malayalam-asr-corpus"
FALLBACK_TEST_FILES = ["data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
CLEAN_CSV_NAME = "own_corpus_test_clean.csv"
EXPECTED_CLEAN = 2684
N_PER_SOURCE = 50
SEED = 42
SR = 16000
REPO_ROOT = Path(__file__).resolve().parent.parent

SUBSETS = {
    "balanced231": dict(n_per_source=N_PER_SOURCE, expected=231),
    "full": dict(n_per_source=None, expected=None),
}


def find_clean_csv(path: str | None = None) -> str:
    candidates = [path] if path else []
    candidates += [str(REPO_ROOT / CLEAN_CSV_NAME), CLEAN_CSV_NAME]
    candidates += glob.glob(f"/kaggle/input/**/{CLEAN_CSV_NAME}", recursive=True)
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(f"{CLEAN_CSV_NAME} not found (looked in {candidates})")


def test_files(token: str | None = None) -> list[str]:
    from huggingface_hub import list_repo_files
    try:
        files = sorted(f for f in list_repo_files(DATASET_ID, repo_type="dataset", token=token)
                       if f.startswith("data/test-") and f.endswith(".parquet"))
        return files or FALLBACK_TEST_FILES
    except Exception:
        return FALLBACK_TEST_FILES


def load_test_split(token: str | None = None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(DATASET_ID, f, repo_type="dataset", token=token) for f in test_files(token)]
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def clean_pool(test: pd.DataFrame, clean: pd.DataFrame, expected: int = EXPECTED_CLEAN) -> pd.DataFrame:
    """Rows of the HF test split whose transcript is in the clean CSV (same rule as the notebooks)."""
    tcol = next((c for c in ("sentence", "text", "transcript") if c in clean.columns), None)
    if tcol is None:
        raise ValueError(f"clean CSV has no text column: {list(clean.columns)}")
    keys = set(clean[tcol].map(match_key))
    is_clean = test["sentence"].map(match_key).isin(keys)
    n = int(is_clean.sum())
    if expected and abs(n - expected) > 0.02 * expected:
        raise AssertionError(f"matched {n} clean rows, expected ~{expected}")
    pool = test[is_clean].copy()
    pool["hf_row"] = pool.index  # row number in the concatenated HF test split
    return pool


def balanced_subset(pool: pd.DataFrame, n_per_source: int = N_PER_SOURCE, seed: int = SEED) -> pd.DataFrame:
    """Same rows, same order as the notebooks' `pool.groupby("source").apply(g.sample(...))`.

    Written as an explicit loop over sorted sources so it doesn't depend on pandas'
    groupby.apply semantics, which change in pandas 3.
    """
    parts = [g.sample(min(len(g), n_per_source), random_state=seed)
             for _, g in pool.groupby("source", sort=True)]
    return pd.concat(parts).reset_index(drop=True)


def decode_audio(a, target_sr: int = SR) -> np.ndarray:
    import librosa
    import soundfile as sf
    b = a["bytes"] if isinstance(a, dict) else a
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    except Exception:
        wav, sr = librosa.load(io.BytesIO(b), sr=None, mono=True)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def prepare(subset: str = "balanced231", out_dir: str = "eval_data", clean_csv: str | None = None,
            token: str | None = None) -> str:
    """Download, filter, subsample and decode once; every model run then reads the same pickle."""
    if subset not in SUBSETS:
        raise ValueError(f"unknown subset {subset}; choose from {list(SUBSETS)}")
    token = token or os.environ.get("HF_TOKEN")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, f"{subset}.pkl")
    if os.path.exists(out_pkl):
        print(f"reusing {out_pkl}")
        return out_pkl

    t0 = time.time()
    test = load_test_split(token)
    clean = pd.read_csv(find_clean_csv(clean_csv))
    pool = clean_pool(test, clean)
    del test
    gc.collect()
    print(f"clean pool: {len(pool)} rows ({time.time() - t0:.0f}s)")

    cfg = SUBSETS[subset]
    sample = balanced_subset(pool, cfg["n_per_source"]) if cfg["n_per_source"] else pool.reset_index(drop=True)
    if cfg["expected"] and len(sample) != cfg["expected"]:
        raise AssertionError(f"{subset}: got {len(sample)} clips, expected {cfg['expected']}")

    sample["wav"] = sample["audio"].map(decode_audio)
    sample["dur"] = sample["wav"].map(lambda w: len(w) / SR)
    sample = sample.drop(columns=["audio"])
    keep = ["hf_row", "source", "sentence", "dur", "wav"]
    sample[keep].to_pickle(out_pkl)
    sample[["hf_row", "source", "dur", "sentence"]].to_csv(os.path.join(out_dir, f"{subset}_manifest.csv"),
                                                           index=False)
    print(sample.source.value_counts().sort_index().to_string())
    print(f"{len(sample)} clips, {sample.dur.sum() / 60:.1f} min, longest {sample.dur.max():.1f}s -> {out_pkl}")
    return out_pkl


def load_prepared(path: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_pickle(path)
    return df.head(limit).copy() if limit else df
