"""Run one model over a prepared evaluation set and write per-utterance results + a summary.

Timing covers only the model call on audio that is already decoded and resampled to 16 kHz,
after a cold call and `warmup` warm calls on the first clip. RTF = total compute / total audio.
Partial results are checkpointed every 25 clips, so a crashed Kaggle run can resume.
"""
from __future__ import annotations

import json
import os
import platform
import time

import numpy as np
import pandas as pd

from . import metrics
from .cpu import isa_flags, peak_rss_mb
from .data import load_prepared
from .models import MODELS, build

NEAR_CAP_MARGIN = 4   # output within this many tokens of the cap is flagged as possibly cut off


def _clean_json(o):
    if isinstance(o, dict):
        return {str(k): _clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_model(name: str, data_pkl: str, out_root: str = "results", threads: int = 2, cpu_info: dict | None = None,
              warmup: int = 3, limit: int | None = None, resume: bool = True, model_id: str | None = None,
              tag: str | None = None, **overrides) -> dict:
    subset = os.path.splitext(os.path.basename(data_pkl))[0]
    run_name = tag or name
    out_dir = os.path.join(out_root, subset, run_name)
    os.makedirs(out_dir, exist_ok=True)
    partial = os.path.join(out_dir, "partial.csv")

    df = load_prepared(data_pkl, limit)
    done = pd.read_csv(partial, keep_default_na=False) if resume and os.path.exists(partial) else None
    rows = done.to_dict("records") if done is not None else []
    todo = df.index[len(rows):]
    if rows:
        print(f"[{run_name}] resuming at clip {len(rows)}/{len(df)}")

    adapter = build(name, threads=threads, model_id=model_id, **overrides)
    t0 = time.time()
    adapter.load()
    load_s = time.time() - t0
    print(f"[{run_name}] loaded in {load_s:.1f}s", flush=True)

    first = df.wav.iloc[0]
    t0 = time.perf_counter()
    adapter.transcribe(first)
    cold_s = time.perf_counter() - t0
    for _ in range(warmup):
        adapter.transcribe(first)

    cap = adapter.token_cap
    t_run = time.time()
    for k, i in enumerate(todo, start=len(rows)):
        wav = df.wav.iloc[i]
        t0 = time.perf_counter()
        tr = adapter.transcribe(wav)
        dt = time.perf_counter() - t0
        rows.append({"i": int(i), "hyp": tr.text, "total_s": dt,
                     "n_tokens": tr.n_tokens if tr.n_tokens is not None else -1})
        if (k + 1) % 25 == 0 or k + 1 == len(df):
            pd.DataFrame(rows).to_csv(partial, index=False)
            sofar = pd.DataFrame(rows)
            rtf = sofar.total_s.sum() / df.dur.iloc[:len(sofar)].sum()
            print(f"[{run_name}] {k + 1}/{len(df)}  RTF so far {rtf:.2f}  ({(time.time() - t_run) / 60:.1f} min)",
                  flush=True)

    raw = pd.DataFrame(rows)
    res = df.loc[raw.i, ["hf_row", "source", "sentence", "dur"]].reset_index(drop=True).rename(columns={"sentence": "ref"})
    res["hyp"] = raw.hyp.values
    res["total_s"] = raw.total_s.values
    res["n_tokens"] = raw.n_tokens.values
    res["rtf"] = res.total_s / res.dur
    res["near_cap"] = (res.n_tokens >= cap - NEAR_CAP_MARGIN) if cap else False
    res = metrics.score(res)

    spec = MODELS[name]
    summary = {
        "model": run_name, "registry_entry": name, "model_id": adapter.model_id, "role": spec.get("role"), "params": spec.get("params"),
        "subset": subset, "overlap_note": None if model_id else spec.get("overlap"),
        "overrides": overrides or None,
        "overall": metrics.summarise(res), "per_source": metrics.per_source(res),
        "speed": {"load_s": load_s, "cold_first_clip_s": cold_s, "warmup_calls": warmup,
                  "peak_rss_MB": peak_rss_mb(),
                  "max_tokens_seen": int(res.n_tokens.max()) if (res.n_tokens >= 0).any() else None},
        "adapter": adapter.info(),
        "hardware": {**(cpu_info or {}), "isa": isa_flags(), "python": platform.python_version()},
        "date": time.strftime("%Y-%m-%d"),
    }
    exp = spec.get("expect")
    if exp and subset == "balanced231" and not limit and not model_id and not overrides:
        summary["sanity"] = {"expected_WER": exp["WER"], "expected_CER": exp["CER"],
                             "delta_WER_pts": 100 * (summary["overall"]["WER"] - exp["WER"]),
                             "delta_CER_pts": 100 * (summary["overall"]["CER"] - exp["CER"])}

    res.to_csv(os.path.join(out_dir, "per_utterance.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(_clean_json(summary), f, ensure_ascii=False, indent=2)
    if os.path.exists(partial):
        os.remove(partial)

    o = summary["overall"]
    print(f"[{run_name}] WER {100 * o['WER']:.1f}% (95% CI {100 * o['WER_CI95'][0]:.1f}–{100 * o['WER_CI95'][1]:.1f})"
          f"  CER {100 * o['CER']:.1f}%  RTF {o['RTF']:.2f}  near-cap {o.get('near_cap', 0)}"
          f"  peak RAM {summary['speed']['peak_rss_MB']:.0f} MB -> {out_dir}")
    return summary
