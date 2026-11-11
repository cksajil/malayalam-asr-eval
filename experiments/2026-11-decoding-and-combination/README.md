# Decoding, rescoring and system combination for whisper-small-malayalam-v2 (Nov 2026)

What can be gained for v2 **without retraining**, on CPU, measured on the 231-clip leakage-free set
(`balanced231`) with this repo's normalisation and scoring. All weights were tuned on a separate
90-clip leakage-free dev set (`results/dev_manifest.csv`: 30 clips each from CommonVoice, IndicTTS and
Shrutilipi, drawn from the clean pool minus balanced231, sharing no sentence with it) and applied to the
test set once. Changes are reported with a paired bootstrap 95% CI (2,000 resamples) against v2 greedy.

**Short version:** decoding tricks alone do nothing; picking between v2 and IndicConformer per clip with a
character LM gives −3.5 to −4.8 WER points, significant, with no tuned parameters. It was deployed to the
voice assistant Space and **reverted**: running both recognisers on the Space's shared CPU slowed the LLM
reply stage from ~2–4 s to ~8 s per turn.

## Results (balanced231)

| System | WER (95% CI) | CER | ΔWER vs v2 greedy [95% CI] |
|---|---|---|---|
| v2 greedy, CTranslate2 FP32 (identical to the published FP32 run on 231/231 clips) | 31.2 (27.9–34.6) | 4.04 | – |
| v2 beam 5 | 31.4 (28.3–34.9) | 3.96 | +0.3 [−1.0, +1.5] |
| v2 beam 5 + word 4-gram N-best rescoring (tuned on dev) | 30.4 (27.2–34.1) | 3.96 | −0.8 [−2.5, +0.9] |
| IndicConformer-600M CTC alone (published run) | 30.1 (27.2–33.3) | 4.00 | −1.0 [−3.6, +1.7] |
| v2 greedy + IndicConformer, AM+LM pick (tuned on dev) | 29.9 | 3.74 | |
| **v2 greedy + IndicConformer, char-LM pick** (two 8-gram LMs, no tuned parameters) | 27.6 (24.5–30.8) | 3.32 | **−3.5 [−5.4, −1.6]** |
| v2 beam 5 + IndicConformer, char-LM pick | 27.0 (24.0–30.2) | 3.16 | −4.2 [−6.3, −2.1] |
| **v2 beam 5 + rescoring + IndicConformer, char-LM pick** | **26.4 (23.3–29.7)** | **3.16** | **−4.8 [−6.9, −2.5]** |
| v2 greedy + IndicConformer, pruned 5-gram pick (10 MB, `asr_select.py`) | 28.5 (25.3–31.7) | 3.35 | −2.7 [−4.6, −0.7] |
| v2 CTranslate2 int8 + IndicConformer, pruned 5-gram (the Space configuration) | 29.1 (26.0–32.4) | 3.45 | −3.5 vs int8 alone [−5.6, −1.5] |

Without Shrutilipi rows (IndicConformer may have trained on Shrutilipi) the best system scores 26.6 against
31.1, so the gain is not from data overlap. Per-source numbers for every run are in `results/summaries.json`;
per-clip outputs of the runs above are in `results/per_clip/`.

## Findings

- **Beam search and N-best LM rescoring don't transfer.** Beam 5 changes nothing. N-best rescoring with word and
  character n-grams (train transcripts and Malayalam Wikipedia, every test/dev sentence removed) gained 3 points
  on dev but 0.8 (not significant) on test: the weights overfit 90 clips. The headroom is real (test 5-best
  oracle 21.5% WER), but an n-gram LM cannot find the right hypothesis.
- **Audio preprocessing doesn't help** (dev): loudness normalisation +0.5, silence trim + loudness −1.1, CIs cross 0.
- **v2 and IndicConformer make different mistakes.** Picking the better of the two per clip with hindsight gives
  23.9% WER. A selector with no tuned parameters (keep the transcript with the higher average per-character
  log-probability) gets most of the way. Using Whisper's own teacher-forced score to choose is worse (29.9): it
  favours Whisper's outputs.
- **Most errors are real, not spelling.** 24 clips are wrong only by word spacing (3.7 WER points); normalising
  orthographic variants (word-final ണ്/ര്/ല്/ള് → chillu, ൌ → ൗ; not ന്, often the dative) moves v2 from
  31.2 to 30.3 only; 81 of 338 substituted word pairs are one character apart (`error_analysis.py`).
- **Truncation trap:** CTranslate2 also emits at most `max_length // 2` tokens; the first dev run stopped at 224
  tokens and scored 47% WER. `max_length=888` fixes it (as in the faster-whisper adapter).
- **whisper-large-v3-turbo is not a CPU option for short questions** (`cpu_speed.py`, CTranslate2 int8, 4-vCPU box):
  its decoder costs the same per token as small's (5.5 vs 5.6 ms at 2 threads; both tokenizers need 28.4 tokens
  per second of Malayalam), but its 32-layer encoder takes 7.0 s per 30 s window against 1.4 s. A 3 s question
  takes 7.5 s (RTF 2.5) against 1.9 s; a 10 s one 8.6 s against 3.0 s.
- **Pre-converted int8 is byte-identical** to converting on the fly (SHA-256 of `model.bin`; `ct2-int8/` in the model
  repo). Its test231 run scores 31.8 / 4.06 against the published int8 32.7 / 4.22: 46 clips differ, int8
  arithmetic varying between CPUs, not the file (`fw_check.py`).

## On the voice assistant Space

The int8 + IndicConformer + pruned 5-gram configuration went live on `sajilck/malayalam-voice-assistant` and was
reverted the same day. From the Space's turn logs: ASR went from ~0.6 s to ~1.5 s, but the Gemma reply stage went
from ~2–4 s to ~7.5–8 s on every turn (IndicConformer's ONNX threads competing with llama.cpp for the same cores),
and the first turn after start took 66 s. ASR was only ~20% of a turn there (p50 0.7 s, RTF 0.23), so on that
hardware an accuracy gain must not cost CPU time. IndicConformer alone (CTC, ~7× real time on 2 cores) is the
speed option if accuracy parity is enough.

## Reproduce

`scripts/run_all.sh` lists every step in order. Working files go under `EXP_DIR` (default this folder; `data/`,
`lm/`, `models/`, `runs/` are git-ignored): decoded audio from `sajilck/malayalam-asr-corpus`, CTranslate2 copies of
v2, KenLM models (`build_lm_text.py`, `build_lms.sh`, ~1.3 GB; KenLM built with `KENLM_MAX_ORDER=8`). Needs
`ctranslate2`, `faster-whisper`, `transformers`, `kenlm`, `jiwer`, `librosa`. About 4 hours on a 4-vCPU machine,
mostly beam-5 decoding.

| Script | Does |
|---|---|
| `prepare.py` | test231 (asserts it equals `results/balanced231/manifest.csv`) and the dev set, decoded to 16 kHz |
| `import_published.py`, `error_analysis.py` | score the published runs; error breakdown and oracle |
| `ct2_nbest.py` | CTranslate2 decoding with an N-best list; optional preprocessing |
| `tf_score.py` | teacher-forced Whisper log-prob for any candidate (N-best and IndicConformer) |
| `rescore2.py` | AM + LM log-linear rescoring, weights tuned on dev, applied to test |
| `combine.py` | v2 vs IndicConformer pick: no-parameter char-LM rule and a 2-fold cross-validated selector |
| `selector_eval.py`, `asr_select.py` | selector LM comparison; the pure-Python ARPA reader shipped in the model repo |
| `cpu_speed.py` | encoder and per-token decoder cost, small vs turbo |
| `fw_check.py` | pre-converted int8 vs the published int8 run |
| `hf_decode.py`, `get_text.py`, `build_lm_text.py`, `build_lms.sh` | Transformers decoding; LM text and models |
| `train_v3_gpu.py` | **untested draft** of a GPU fine-tuning recipe (turbo base, normalised targets, augmentation) |

`results/nbest/` keeps the beam-5 N-best lists (5 hypotheses per clip, CTranslate2's length-normalised score) for
anyone who wants to try a better rescorer: the oracle says there are 10 WER points in them.
IndicConformer outputs for the dev set could not be produced in this run, so the combination rule was evaluated
without tuned parameters only (plus a 2-fold cross-validation on test, reported in `summaries.json` as `*_cv2`).
