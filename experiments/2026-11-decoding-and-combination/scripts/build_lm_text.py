"""LM training text: the corpus train transcripts and Malayalam Wikipedia (20231101.ml), normalised with norm_ml,
minus every sentence that is also in test231 or dev (so the LMs cannot have seen a test sentence).
Writes lm/train.txt, lm/wiki.txt and character versions lm/*.chr ("a b c | d e", | = word boundary).
Run get_text.py first (train transcripts only, no audio)."""
import re, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from huggingface_hub import hf_hub_download

out = f"{ROOT}/lm"; os.makedirs(out, exist_ok=True)
excl = {norm_ml(s) for n in ("test231", "dev") for s in load_set(n).sentence}
tr = pd.read_csv(f"{DATA}/train_text.csv")
train = [l for l in (norm_ml(s) for s in tr.sentence) if l and l not in excl]
print("train lines", len(train), "removed", len(tr) - len(train))
w = pd.read_parquet(hf_hub_download("wikimedia/wikipedia", "20231101.ml/train-00000-of-00001.parquet", repo_type="dataset"),
                    columns=["text"])
ml = re.compile(r"^[ഀ-ൿ\s]+$")
wiki = [s for t in w.text for s in map(norm_ml, re.split(r"[.!?\n।]+", t))
        if 3 <= len(s.split()) <= 40 and ml.match(s) and s not in excl]
print("wiki sentences", len(wiki))
for name, lines in (("train", train), ("wiki", wiki)):
    open(f"{out}/{name}.txt", "w").write("\n".join(lines) + "\n")
    open(f"{out}/{name}.chr", "w").write("\n".join(" ".join(l.replace(" ", "|")) for l in lines) + "\n")
