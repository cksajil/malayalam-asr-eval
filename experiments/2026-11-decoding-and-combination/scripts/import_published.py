"""Score the published balanced231 outputs (whisper-v2, fw-v2-int8, indicconformer-ctc) with this folder's scorer
and save them as runs/pub_<model>_test.csv, the inputs combine.py and error_analysis.py read."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *

t = load_set("test231")
for m in ["whisper-v2", "fw-v2-int8", "indicconformer-ctc"]:
    d = pd.read_csv(f"{REPO}/results/balanced231/{m}/per_utterance.csv")
    assert list(d.hf_row) == list(t.hf_row), m
    save_run(f"pub_{m}_test", t, d.hyp.fillna("").tolist())
