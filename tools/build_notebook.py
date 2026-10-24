"""Generate notebooks/kaggle_cpu_benchmark.ipynb. Edit here, then: python tools/build_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# Malayalam ASR eval — accuracy per source + CPU speed on 2 cores

Runs every model in [malayalam-asr-eval](https://github.com/cksajil/malayalam-asr-eval) on the
**231-clip leakage-free subset** (50 per source, seed 42 — the same clips as the published v1/v2 numbers),
one model per process, pinned to **2 physical cores**, batch 1, greedy decoding.

**Order:** `whisper-v2` runs first as a sanity check. It must land near **31.2% WER / 4.0% CER**; if it
doesn't, the notebook stops, because nothing after it would be comparable.

**Kaggle setup**
- Accelerator → **None** (CPU). Internet → **On**.
- Add-ons → Secrets → `HF_TOKEN` (needed for IndicConformer, which is gated: accept its conditions on
  huggingface.co/ai4bharat/indic-conformer-600m-multilingual first). Without it that one model is skipped.
- Runtime: roughly 4–5 h for the default plan (vrclc medium is the slowest, so it runs last).
  Finished models are skipped on re-run, and an interrupted model resumes from its last checkpoint.""")

code(r'''# ---------------- CONFIG ----------------
REPO_URL   = "https://github.com/cksajil/malayalam-asr-eval"
BRANCH     = "main"         # use the PR branch name here until it is merged
CORES      = 2              # physical cores to pin each run to
PLAN       = ["whisper-v2",          # sanity check, must be first
              "fw-v2-int8",          # CTranslate2 int8 of v2
              "indicconformer-ctc",  # CTC, non-autoregressive
              "mms-1b-all",          # wav2vec2 CTC (MMS, Malayalam adapter)
              "kavya-small",         # competitor
              "vrclc-medium"]        # competitor, slowest
OPTIONAL   = []             # e.g. ["indicconformer-rnnt", "xlsr-ml-shawn", "whisper-v1"]
RUN_FULL   = False          # also score the full 2,684-clip clean set (~88% Shrutilipi; ~10x longer)
SANITY_TOL = {"WER": 1.0, "CER": 0.5}   # percentage points vs published v2
WORK       = "/kaggle/working"''')

code(r'''import os, sys, json, subprocess, shutil, time
os.chdir(WORK)
if not os.path.isdir("malayalam-asr-eval"):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", BRANCH, REPO_URL], check=True)
REPO = f"{WORK}/malayalam-asr-eval"
print(subprocess.run(["git", "-C", REPO, "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
# Every model runs in a fresh subprocess, so packages installed here are picked up without a kernel restart.
subprocess.run(f"pip install -q -e '{REPO}[all]'", shell=True, check=True)''')

code(r'''try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception as e:
    print("No HF_TOKEN secret - gated models (IndicConformer) will be skipped:", type(e).__name__)
HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))

print(subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core|Socket|Flags' | cut -c1-160",
                     shell=True, capture_output=True, text=True).stdout)
import importlib.metadata as md_
for p in ["torch", "transformers", "ctranslate2", "faster-whisper", "onnxruntime", "jiwer", "pandas"]:
    try: print(f"{p:16s} {md_.version(p)}")
    except Exception: print(f"{p:16s} -")''')

md("## 1. Evaluation data (download, filter to clean rows, sample, decode once)")

code(r'''def cli(*args, log=None):
    cmd = [sys.executable, "-u", "-m", "mlasr_eval", *args]
    print("$", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if not any(k in line for k in ("Warning", "warn(", "it/s]")):
            print(line, end="", flush=True)
    p.wait()
    if log:
        open(log, "w").write("".join(lines))
    return p.returncode, "".join(lines)

rc, _ = cli("prepare", "--subset", "balanced231", "--out", f"{WORK}/eval_data")
assert rc == 0, "data preparation failed"
DATA = f"{WORK}/eval_data/balanced231.pkl"
RESULTS = f"{WORK}/results"''')

md("## 2. Sanity check: v2 must reproduce 31.2% WER / 4.0% CER")

code(r'''def summary_path(name, subset="balanced231"):
    return f"{RESULTS}/{subset}/{name}/summary.json"

def run(name, data=DATA):
    if os.path.exists(summary_path(name, os.path.splitext(os.path.basename(data))[0])):
        print(f"[{name}] already done, skipping"); return 0
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    rc, out = cli("run", name, "--data", data, "--results", RESULTS, "--cores", str(CORES),
                  log=f"{RESULTS}/{name}.log")
    print(f"[{name}] exit {rc} after {(time.time()-t0)/60:.1f} min\n")
    return rc

assert run("whisper-v2") == 0, "whisper-v2 run failed - see results/whisper-v2.log"
s = json.load(open(summary_path("whisper-v2")))
c = s["sanity"]
print(f"v2 WER {100*s['overall']['WER']:.1f}% (expected 31.2, {c['delta_WER_pts']:+.1f} pts)  "
      f"CER {100*s['overall']['CER']:.1f}% (expected 4.0, {c['delta_CER_pts']:+.1f} pts)  "
      f"near-cap clips {s['overall']['near_cap']}  RTF {s['overall']['RTF']:.2f}")
if abs(c["delta_WER_pts"]) > SANITY_TOL["WER"] or abs(c["delta_CER_pts"]) > SANITY_TOL["CER"]:
    raise SystemExit("Sanity check FAILED - the setup does not match the published eval. "
                     "Check the transformers version (4.46.3), max_new_tokens (444) and the clip manifest.")
print("Sanity check passed - setup matches the published numbers.")''')

md("## 3. All other models")

code(r'''sys.path.insert(0, REPO)   # the editable install isn't visible to this kernel until a restart
from mlasr_eval.models import MODELS
status = {}
for name in PLAN[1:] + OPTIONAL:
    if MODELS[name].get("gated") and not HAS_TOKEN:
        status[name] = "skipped (needs HF_TOKEN)"; print(f"[{name}] {status[name]}"); continue
    rc = run(name)
    status[name] = "ok" if rc == 0 else f"FAILED (exit {rc}) - see results/{name}.log"
status''')

md("## 4. Leaderboard")

code(r'''from IPython.display import Markdown, display
from mlasr_eval.report import write_report
display(Markdown(open(write_report(RESULTS, "balanced231")).read()))''')

md("## 5. Optional: full 2,684-clip clean set")

code(r'''if RUN_FULL:
    rc, _ = cli("prepare", "--subset", "full", "--out", f"{WORK}/eval_data")
    for name in PLAN + OPTIONAL:
        if MODELS[name].get("gated") and not HAS_TOKEN:
            continue
        run(name, data=f"{WORK}/eval_data/full.pkl")
    display(Markdown(open(write_report(RESULTS, "full")).read()))''')

md("## 6. Package results (download `results.zip`, or commit `results/` to the repo)")

code(r'''shutil.copy(f"{WORK}/eval_data/balanced231_manifest.csv", f"{RESULTS}/balanced231/manifest.csv")
shutil.make_archive(f"{WORK}/results", "zip", RESULTS)
print(sorted(os.listdir(f"{RESULTS}/balanced231")))''')

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"},
                  "kaggle": {"accelerator": "none", "isInternetEnabled": True, "language": "python",
                             "sourceType": "notebook", "isGpuEnabled": False}}
nbf.write(nb, "notebooks/kaggle_cpu_benchmark.ipynb")
print("wrote notebooks/kaggle_cpu_benchmark.ipynb")
