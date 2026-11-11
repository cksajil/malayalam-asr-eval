"""Transcribe test231 with the Space's exact faster-whisper settings; compare with published fw-v2-int8 outputs."""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
from faster_whisper import WhisperModel
d = sys.argv[1]
m = WhisperModel(d, device="cpu", compute_type="int8", cpu_threads=4, num_workers=1)
m.max_length = 888
t = load_set("test231")
pub = pd.read_csv(f"{REPO}/results/balanced231/fw-v2-int8/per_utterance.csv").hyp.fillna("")
hyps, t0 = [], time.time()
for w in t.wav:
    seg, _ = m.transcribe(w, language="ml", task="transcribe", beam_size=1, best_of=1, temperature=0.0,
                          condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
                          word_timestamps=False, max_new_tokens=884)
    hyps.append(" ".join(s.text.strip() for s in seg).strip())
same = sum(a.strip() == b.strip() for a, b in zip(hyps, pub))
print(f"identical to published fw-v2-int8: {same}/{len(t)}  ({time.time()-t0:.0f}s)")
s = save_run("fw_int8_preconv_test", t, hyps)
print("published:", fmt("fw-v2-int8 (published)", score(t.sentence, pub)))
