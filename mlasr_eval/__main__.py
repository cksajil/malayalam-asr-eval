"""CLI.

    python -m mlasr_eval prepare --subset balanced231
    python -m mlasr_eval run whisper-v2 --cores 2
    python -m mlasr_eval report
    python -m mlasr_eval list

Run one model per process: CPU pinning and thread limits are applied before torch,
onnxruntime or ctranslate2 are imported, and peak RAM is then per model.
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mlasr_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download, filter, subsample and decode the test audio")
    p.add_argument("--subset", default="balanced231", choices=["balanced231", "full"])
    p.add_argument("--out", default="eval_data")
    p.add_argument("--clean-csv", default=None)

    r = sub.add_parser("run", help="score one model")
    r.add_argument("model")
    r.add_argument("--data", default="eval_data/balanced231.pkl")
    r.add_argument("--results", default="results")
    r.add_argument("--cores", type=int, default=2, help="physical cores to pin to")
    r.add_argument("--threads", type=int, default=None, help="math threads (default = cores)")
    r.add_argument("--warmup", type=int, default=3)
    r.add_argument("--limit", type=int, default=None, help="first N clips only (smoke test)")
    r.add_argument("--no-resume", action="store_true")
    r.add_argument("--max-new-tokens", type=int, default=None)
    r.add_argument("--model-id", default=None, help="score another checkpoint with this entry's engine")
    r.add_argument("--tag", default=None, help="results folder name (default: the model name)")

    rep = sub.add_parser("report", help="build the leaderboard from results/")
    rep.add_argument("--results", default="results")
    rep.add_argument("--subset", default="balanced231")

    sub.add_parser("list", help="list registered models")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        # Must happen before numpy/torch/onnxruntime/ctranslate2 are imported.
        from .cpu import configure
        cpu_info = configure(a.cores, a.threads)
        from .runner import run_model
        overrides = {"max_new_tokens": a.max_new_tokens} if a.max_new_tokens else {}
        run_model(a.model, a.data, a.results, threads=cpu_info["threads"], cpu_info=cpu_info,
                  warmup=a.warmup, limit=a.limit, resume=not a.no_resume, model_id=a.model_id,
                  tag=a.tag, **overrides)
    elif a.cmd == "prepare":
        from .data import prepare
        prepare(a.subset, a.out, a.clean_csv)
    elif a.cmd == "report":
        from .report import write_report
        path = write_report(a.results, a.subset)
        print(open(path).read())
    elif a.cmd == "list":
        from .models import DEFAULT_PLAN, MODELS
        for name, s in MODELS.items():
            flag = "default" if name in DEFAULT_PLAN else ("optional" if s.get("optional") else "")
            print(f"{name:22s} {s['model_id']:48s} {s.get('role', ''):34s} {flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
