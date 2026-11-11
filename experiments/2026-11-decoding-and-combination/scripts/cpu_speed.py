"""CPU cost of whisper-small (v2) vs whisper-large-v3-turbo, CTranslate2 int8, greedy: the encoder (fixed, one
30 s window) and the decoder per token, at 2 and 4 threads; then the time for a question of 3, 5 and 10 s at the
measured 28.4 tokens per second of Malayalam (same for both tokenizers). Convert first, e.g.
  ct2-transformers-converter --model openai/whisper-large-v3-turbo --output_dir models/ct2-turbo-int8 --quantization int8
  ct2-transformers-converter --model sajilck/whisper-small-malayalam-v2 --output_dir models/ct2-small-int8 --quantization int8"""
import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from common import *
import ctranslate2
from transformers import WhisperFeatureExtractor

t = load_set("test231"); wav = t.wav.iloc[int(np.argmin(abs(t.dur - 4.8)))]
TOK_PER_S, EOT = 28.4, 50257
for name, nmel, prompt in [("small", 80, [50258, 50303, 50359, 50363]), ("turbo", 128, [50258, 50304, 50360, 50364])]:
    f = ctranslate2.StorageView.from_array(np.ascontiguousarray(
        WhisperFeatureExtractor(feature_size=nmel)(wav, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)))
    for th in (2, 4):
        m = ctranslate2.models.Whisper(f"{ROOT}/models/ct2-{name}-int8", device="cpu", compute_type="int8", intra_threads=th)
        def run(n):                     # EOT suppressed so exactly n tokens are generated; median of 3
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); m.generate(f, [prompt], max_length=len(prompt) + n, suppress_tokens=[EOT], beam_size=1)
                ts.append(time.perf_counter() - t0)
            return sorted(ts)[1]
        run(1); enc = run(1); per = (run(141) - enc) / 140
        for secs in (3, 5, 10):
            tot = enc + per * TOK_PER_S * secs
            print(f"{name} threads={th} encoder {enc:.2f}s per-token {1000 * per:.1f}ms -> {secs}s question {tot:.1f}s RTF {tot / secs:.2f}")
