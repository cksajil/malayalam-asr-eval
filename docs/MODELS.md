# Models: what's in, what's out, and why

Checked October 2026. "Overlap" means a third-party model may have been trained on the
same public corpora our test rows come from. `own_corpus_test_clean.csv` is only filtered
against `sajilck/malayalam-asr-corpus`'s own train split, so it can't rule this out.

## Benchmarked

| Registry name | Checkpoint | Engine | Why it's here | Overlap risk |
|---|---|---|---|---|
| `whisper-v2` | sajilck/whisper-small-malayalam-v2 | Transformers FP32, greedy, 444 tokens | Reference; sanity check (31.2% / 4.0%) | None (filtered) |
| `fw-v2-int8` | same weights, converted | CTranslate2 int8 (faster-whisper) | Does int8 + CTranslate2 make autoregressive Whisper usable on 2 cores? | None |
| `indicconformer-ctc` | ai4bharat/indic-conformer-600m-multilingual | HF remote code (ONNX Runtime), CTC | Non-autoregressive: cost scales with audio length, not token count | Training data not stated; AI4Bharat also built Shrutilipi |
| `mms-1b-all` | facebook/mms-1b-all, `target_lang="mal"` | wav2vec2 CTC, FP32 | The strongest maintained wav2vec2 option with a Malayalam head | Fine-tuned on data that includes Common Voice and FLEURS |
| `kavya-small` | kavyamanohar/whisper-small-malayalam | Transformers FP32 | Competitor, same size as v2 | Card lists Common Voice 13 |
| `vrclc-medium` | vrclc/Whisper-medium-Malayalam | Transformers FP32 | Competitor, 3× larger | Not checked yet: read its card before quoting per-source numbers |

## Optional entries

| Registry name | Notes |
|---|---|
| `whisper-v1` | Re-score v1 (published 43.5% / 7.3%) under the same harness. |
| `indicconformer-rnnt` | Same model with RNNT decoding, to see what CTC gives up. |
| `xlsr-ml-shawn` | shawn-nyk/wav2vec-large-xlsr-malayalam-with-lm (XLS-R 300M, greedy, LM not used). Its card says it was trained on Indic TTS, OpenSLR, SMC and IIIT-H Malayalam, so `indictts`, `openslr63` and `smc` rows likely overlap; it is off by default. |

Other wav2vec2 Malayalam checkpoints on the Hub (e.g. gvs/wav2vec2-large-xlsr-malayalam) are small
XLS-R fine-tunes from 2021 on Common Voice; MMS was preferred as the default because it is
maintained, larger and trained on more data.

## Checked and excluded: no Malayalam support

| Model | Supported languages (from the official model pages) | Decision |
|---|---|---|
| NVIDIA Nemotron 3.5 ASR Streaming 0.6B | 40 locales across three tiers; Indic coverage is Hindi only | Skip |
| NVIDIA Parakeet 1.1B RNNT multilingual (NGC) | Multilingual set + an Indic variant with Bengali, Hindi and Tamil | Skip |
| NVIDIA Parakeet TDT 0.6B v3 | 25 European languages | Skip |

None of these list Malayalam, so they would need fine-tuning before they could be compared,
which is a separate project. Revisit if NVIDIA adds Malayalam or a community Malayalam
fine-tune appears.

## Notes on fairness

- Every model gets the same audio (16 kHz mono float32), the same normalisation and the
  same scorer. No language model is used for any CTC model.
- Autoregressive models are capped at 444 new tokens (4 prompt + 444 = 448, Whisper's
  limit). The `Near cap` column counts clips whose output reached the cap; it should be 0.
- CPU speed: each model runs in its own process pinned to 2 physical cores. IndicConformer's
  ONNX Runtime sessions pick their own thread count, but affinity keeps them on the same
  2 cores. Speeds depend on the CPU Kaggle assigns (the CPU model is recorded in every
  result), so compare rows from the same run.
