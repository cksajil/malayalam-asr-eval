"""Collect results/<subset>/<model>/summary.json into a leaderboard (Markdown + CSV)."""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

SOURCES = ["shrutilipi", "commonvoice", "indictts", "imasc", "openslr63", "smc"]


def load_summaries(results_dir: str, subset: str) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(results_dir, subset, "*", "summary.json"))):
        s = json.load(open(p))
        sp = s.setdefault("speed", {})
        utt = os.path.join(os.path.dirname(p), "per_utterance.csv")
        if "clips_at_max_tokens" not in sp and os.path.exists(utt):   # results from before this field existed
            cols = pd.read_csv(utt, nrows=0).columns
            if "n_tokens" in cols:
                tok = pd.read_csv(utt, usecols=["n_tokens"]).n_tokens
                sp["clips_at_max_tokens"] = int((tok == tok.max()).sum()) if (tok >= 0).any() else None
        out.append(s)
    return out


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}"


def v2_rtf_by_session(summaries: list[dict]) -> dict:
    """whisper-v2's RTF in each session: the speed reference measured on the same machine."""
    ref = {}
    for s in summaries:
        if s.get("registry_entry", s["model"]) == "whisper-v2" and not s.get("overrides") and s.get("session"):
            ref[s["session"]] = s["overall"]["RTF"]
    return ref


def speed_vs_v2(s: dict, ref: dict):
    """How many times faster than whisper-v2 on the same machine (>1 = faster). None if no reference."""
    v2 = ref.get(s.get("session")) or s.get("speed_ref", {}).get("v2_RTF")
    return round(v2 / s["overall"]["RTF"], 2) if v2 else None


def overlap_sources(s: dict) -> set:
    from .models import MODELS
    spec = MODELS.get(s.get("registry_entry", s["model"]), {})
    return set() if s.get("model_id") != spec.get("model_id") else set(spec.get("overlap_sources", []))


def unseen_wer(s: dict, results_dir: str, subset: str):
    """Corpus WER after dropping clips from sources in the model's own training data."""
    flagged = overlap_sources(s)
    if not flagged:
        return s["overall"]["WER"], s["overall"]["n_clips"]
    utt = os.path.join(results_dir, subset, s["model"], "per_utterance.csv")
    if not os.path.exists(utt):
        return None, None
    d = pd.read_csv(utt, keep_default_na=False)
    d = d[~d.source.isin(flagged)]
    from .metrics import corpus_wer
    return corpus_wer(d), len(d)


def tables(summaries: list[dict], results_dir: str = "results",
           subset: str = "balanced231") -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, src_rows = [], []
    ref = v2_rtf_by_session(summaries)
    for s in summaries:
        o, sp = s["overall"], s["speed"]
        rows.append({
            "Model": s["model"], "Checkpoint": s["model_id"], "Params": s.get("params"),
            "Engine": s["adapter"].get("engine"),
            "WER %": round(100 * o["WER"], 1),
            "95% CI": f"{pct(o['WER_CI95'][0])}–{pct(o['WER_CI95'][1])}",
            "CER %": round(100 * o["CER"], 1),
            "WER % unseen sources": (lambda w: f"{100 * w[0]:.1f} ({w[1]} clips)" if w[0] is not None else "–")(
                unseen_wer(s, results_dir, subset)),
            "Speed vs v2": speed_vs_v2(s, ref),
            "RTF": round(o["RTF"], 2), "RTFx": round(o["RTFx"], 2) if o.get("RTFx") else None,
            "p90 latency s": round(o["latency_p90_s"], 2),
            "Peak RAM GB": round(sp["peak_rss_MB"] / 1000, 2),
            "Near cap": o.get("near_cap", 0),
            "Max tokens": (f"{sp['max_tokens_seen']} ({sp.get('clips_at_max_tokens', '?')}×)"
                           if sp.get("max_tokens_seen") not in (None, -1) else "–"),
            "Empty": o["empty"], "Loops": o["loops"],
        })
        r = {"Model": s["model"]}
        flagged = overlap_sources(s)
        for src in SOURCES:
            d = s["per_source"].get(src)
            r[src] = (f"{pct(d['WER'])} / {pct(d['CER'])}" + (" *" if src in flagged else "")) if d else "–"
        src_rows.append(r)
    overall = pd.DataFrame(rows).sort_values("WER %").reset_index(drop=True)
    per_src = pd.DataFrame(src_rows).set_index("Model").loc[overall.Model].reset_index()
    return overall, per_src


def write_report(results_dir: str = "results", subset: str = "balanced231") -> str:
    summaries = load_summaries(results_dir, subset)
    if not summaries:
        raise FileNotFoundError(f"no summary.json under {results_dir}/{subset}")
    overall, per_src = tables(summaries, results_dir, subset)
    out_dir = os.path.join(results_dir, subset)
    overall.to_csv(os.path.join(out_dir, "leaderboard.csv"), index=False)
    per_src.to_csv(os.path.join(out_dir, "per_source.csv"), index=False)

    hw = summaries[0].get("hardware", {})
    first = summaries[0]["overall"]
    lines = [
        f"# Malayalam ASR leaderboard — `{subset}`",
        "",
        f"{first['n_clips']} leakage-free clips ({first['audio_min']} min). "
        f"CPU: {hw.get('cpu_model', '?')}, pinned to {hw.get('pinned_physical_cores', '?')} physical cores "
        f"({hw.get('threads', '?')} threads), batch 1, greedy decoding. RTF below 1 is faster than real time. "
        f"Speed vs v2 = whisper-v2's RTF on the same machine ÷ this model's RTF (2.0 = twice as fast); "
        f"use it rather than RTF to compare rows from different Kaggle sessions.",
        "",
        "**WER % unseen sources** drops the clips from corpora in that model's own training data "
        "(marked * below). It is the fair column for comparing third-party models.",
        "",
        "## Overall (sorted by WER)",
        "",
        overall.to_markdown(index=False),
        "",
        "## WER / CER % by source",
        "",
        per_src.to_markdown(index=False),
        "",
        "\\* The model's published training data includes this corpus, so these test sentences "
        "may have been seen in training.",
        "",
    ]
    sanity = [s for s in summaries if "sanity" in s]
    if sanity:
        lines += ["## Sanity check against published numbers", ""]
        for s in sanity:
            c = s["sanity"]
            ok = abs(c["delta_WER_pts"]) <= 1.0 and abs(c["delta_CER_pts"]) <= 0.5
            lines.append(f"- **{s['model']}**: WER {pct(s['overall']['WER'])}% vs {pct(c['expected_WER'])}% "
                         f"({c['delta_WER_pts']:+.1f} pts), CER {pct(s['overall']['CER'])}% vs "
                         f"{pct(c['expected_CER'])}% ({c['delta_CER_pts']:+.1f} pts) — "
                         f"{'matches' if ok else 'DOES NOT MATCH: check the setup before trusting other rows'}")
        lines.append("")
    flagged = [s for s in summaries if s.get("warnings")]
    if flagged:
        lines += ["## Warnings", ""]
        for s in flagged:
            for w in s["warnings"]:
                lines.append(f"- **{s['model']}**: {w}")
        lines.append("")
    lines += ["## Training-data overlap notes", ""]
    for s in summaries:
        if s.get("overlap_note"):
            lines.append(f"- **{s['model']}**: {s['overlap_note']}")
    path = os.path.join(out_dir, "LEADERBOARD.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
