"""Where v2's errors come from on test231: spacing-only errors, substituted word pairs, orthographic-variant
normalisation (secondary metric), and how complementary Whisper and IndicConformer are (oracle choice)."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from common import _counts

t = load_set("test231"); ref = t.sentence
w = pd.read_csv(f"{RES}/pub_whisper-v2_test.csv").hyp.fillna("")
c = pd.read_csv(f"{RES}/pub_indicconformer-ctc_test.csv").hyp.fillna("")
N = sum(_counts(r, "")[3] for r in ref)
ew = np.array([sum(_counts(r, h)[:3]) for r, h in zip(ref, w)])
ec = np.array([sum(_counts(r, h)[:3]) for r, h in zip(ref, c)])
ns = lambda s: norm_ml(s).replace(" ", "")
seg = np.array([ns(r) == ns(h) and norm_ml(r) != norm_ml(h) for r, h in zip(ref, w)])
print(f"clips wrong only by word spacing: {seg.sum()} -> {100 * ew[seg].sum() / N:.1f} of {100 * ew.sum() / N:.1f} WER points")
o = jiwer.process_words([norm_ml(x) for x in ref], [norm_ml(x) for x in w])
subs = [p for a, rw, hw in zip(o.alignments, o.references, o.hypotheses) for ch in a if ch.type == "substitute"
        for p in zip(rw[ch.ref_start_idx:ch.ref_end_idx], hw[ch.hyp_start_idx:ch.hyp_end_idx])]
def char_edits(a, b):
    o = jiwer.process_characters(a, b)
    return o.substitutions + o.deletions + o.insertions
one = sum(1 for a, b in subs if char_edits(a, b) == 1)
print(f"substituted word pairs: {len(subs)}, one character apart: {one}")
for name, h in (("whisper-v2", w), ("indicconformer", c)):
    print(f"ortho-normalised {name}:", {k: round(v, 2) for k, v in score_ortho(ref, h).items()})
print(f"oracle(whisper, conformer) WER {100 * np.minimum(ew, ec).sum() / N:.1f}; "
      f"whisper better {(ew < ec).sum()}, conformer better {(ec < ew).sum()}, tie {(ew == ec).sum()}")
