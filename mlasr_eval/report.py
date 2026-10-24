"""Collect results/<subset>/<model>/summary.json into a leaderboard (Markdown + CSV)."""
from __future__ import annotations

import glob
import json
import os

import pandas as pd

SOURCES = ["shrutilipi", "commonvoice", "indictts", "imasc", "openslr63", "smc"]


def load_summaries(results_dir: str, subset: str) -> list[dict]:
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(results_dir, subset, "*", "summary.json")))]


def pct(x, nd=1):
    return "–" if x is None else f"{100 * x:.{nd}f}"


def tables(summaries: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, src_rows = [], []
    for s in summaries:
        o, sp = s["overall"], s["speed"]
        rows.append({
            "Model": s["model"], "Checkpoint": s["model_id"], "Params": s.get("params"),
            "Engine": s["adapter"].get("engine"),
            "WER %": round(100 * o["WER"], 1),
            "95% CI": f"{pct(o['WER_CI95'][0])}–{pct(o['WER_CI95'][1])}",
            "CER %": round(100 * o["CER"], 1),
            "RTF": round(o["RTF"], 2), "RTFx": round(o["RTFx"], 2) if o.get("RTFx") else None,
            "p90 latency s": round(o["latency_p90_s"], 2),
            "Peak RAM GB": round(sp["peak_rss_MB"] / 1000, 2),
            "Near cap": o.get("near_cap", 0), "Empty": o["empty"], "Loops": o["loops"],
        })
        r = {"Model": s["model"]}
        for src in SOURCES:
            d = s["per_source"].get(src)
            r[src] = f"{pct(d['WER'])} / {pct(d['CER'])}" if d else "–"
        src_rows.append(r)
    overall = pd.DataFrame(rows).sort_values("WER %").reset_index(drop=True)
    per_src = pd.DataFrame(src_rows).set_index("Model").loc[overall.Model].reset_index()
    return overall, per_src


def write_report(results_dir: str = "results", subset: str = "balanced231") -> str:
    summaries = load_summaries(results_dir, subset)
    if not summaries:
        raise FileNotFoundError(f"no summary.json under {results_dir}/{subset}")
    overall, per_src = tables(summaries)
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
        f"({hw.get('threads', '?')} threads), batch 1, greedy decoding. RTF below 1 is faster than real time.",
        "",
        "## Overall (sorted by WER)",
        "",
        overall.to_markdown(index=False),
        "",
        "## WER / CER % by source",
        "",
        per_src.to_markdown(index=False),
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
    lines += ["## Training-data overlap notes", ""]
    for s in summaries:
        if s.get("overlap_note"):
            lines.append(f"- **{s['model']}**: {s['overlap_note']}")
    path = os.path.join(out_dir, "LEADERBOARD.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
