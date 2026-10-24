# malayalam-asr-eval

A reproducible benchmark for Malayalam speech recognition: **WER/CER per source** on a
leakage-free test set, plus **CPU speed on 2 cores** — because a model that needs a GPU
won't reach most of the people Malayalam ASR is for.

- **Leakage-free test set.** `own_corpus_test_clean.csv` keeps only rows of
  [sajilck/malayalam-asr-corpus](https://huggingface.co/datasets/sajilck/malayalam-asr-corpus)'s
  test split whose transcript doesn't appear in its train split (2,684 of 4,828 rows; see
  [Part 2](https://intuitivetutorial.com/2026/10/09/i-re-tested-my-malayalam-asr-model-on-clean-data-part-2/)).
- **Headline subset: 231 clips**, 50 per source (all of IMaSC, OpenSLR 63 and SMC, which have
  fewer), sampled with seed 42. The full clean set is 88% Shrutilipi, so it mostly measures
  one source; the balanced subset is what the published v1/v2 numbers use.
- **One interface for every engine:** Transformers Whisper, CTranslate2/faster-whisper,
  IndicConformer (CTC/RNNT) and wav2vec2-CTC (MMS, XLS-R) all sit behind
  `transcribe(wav) -> text`.

## Models

| Name | Checkpoint | Engine |
|---|---|---|
| `whisper-v2` | sajilck/whisper-small-malayalam-v2 | Transformers FP32 (sanity check: 31.2% WER / 4.0% CER) |
| `fw-v2-int8` | same weights | CTranslate2 int8 |
| `indicconformer-ctc` | ai4bharat/indic-conformer-600m-multilingual | CTC (gated: needs `HF_TOKEN`) |
| `mms-1b-all` | facebook/mms-1b-all (`mal` adapter) | wav2vec2 CTC |
| `kavya-small` | kavyamanohar/whisper-small-malayalam | Transformers FP32 |
| `vrclc-medium` | vrclc/Whisper-medium-Malayalam | Transformers FP32 |

Optional: `whisper-v1`, `indicconformer-rnnt`, `xlsr-ml-shawn`. Parakeet and Nemotron were
checked and don't support Malayalam. Details, and which models may have trained on the same
public corpora as the test rows, are in [docs/MODELS.md](docs/MODELS.md).

## Results (interim, 24 Oct 2026)

231 leakage-free clips (18.6 min of audio), Kaggle CPU (Intel Xeon 2.20 GHz) pinned to 2 physical
cores, batch 1, greedy decoding. RTF below 1 is faster than real time. Full tables, per-source
numbers and per-clip outputs: [results/balanced231/](results/balanced231/LEADERBOARD.md).

| Model | WER % (95% CI) | CER % | RTF | Peak RAM |
|---|---|---|---|---|
| IndicConformer 600M, CTC | **30.1** (27.2–33.3) | **4.0** | **0.14** | 3.9 GB |
| whisper-small-malayalam-v2 | 31.2 (27.9–34.6) | **4.0** | 1.93 | 2.4 GB |
| MMS-1b-all (wav2vec2 CTC) | 52.5 (49.0–56.1) | 9.2 | 0.86 | 8.8 GB |
| kavyamanohar/whisper-small-malayalam | 87.0 (82.4–93.7) | 29.2 | 2.30 | 2.5 GB |

Still to come: v2 on CTranslate2 int8 (the first run hit a hidden 224-token limit in CTranslate2
and was discarded) and vrclc/Whisper-medium-Malayalam (gated; access requested). See
[docs/MODELS.md](docs/MODELS.md) for possible training-data overlap before comparing third-party
models per source.

## Run it

**Kaggle (recommended, CPU, ~4–5 h):** open
[`notebooks/kaggle_cpu_benchmark.ipynb`](notebooks/kaggle_cpu_benchmark.ipynb), set Accelerator
to None and Internet on, add an `HF_TOKEN` secret, and run all. It scores `whisper-v2` first and
stops if it doesn't reproduce 31.2% / 4.0% (±1.0 / ±0.5 points).

**Anywhere else:**

```bash
pip install -e ".[all]"                       # or .[whisper], .[faster-whisper], .[conformer], .[wav2vec2]
python -m mlasr_eval prepare --subset balanced231
python -m mlasr_eval run whisper-v2 --cores 2  # one model per process
python -m mlasr_eval run fw-v2-int8 --cores 2
python -m mlasr_eval report                    # results/balanced231/LEADERBOARD.md
```

Useful flags: `--limit 10` (smoke test), `--model-id <repo>` (score another checkpoint with an
entry's engine, e.g. a new Whisper fine-tune via `whisper-v2`), `--tag <name>`, `--threads`.
Runs checkpoint every 25 clips and resume automatically.

Each run writes `results/<subset>/<model>/per_utterance.csv` (reference, hypothesis, errors,
timing, token count per clip) and `summary.json` (overall and per-source WER with 95% CI, CER,
RTF, latency, peak RAM, hardware and library versions).

## Method

- **Normalisation** (identical to the published notebooks): NFC; old-style chillu (consonant
  + virama + ZWJ) → atomic chillu; drop ZWJ/ZWNJ; punctuation and symbols → space; collapse
  whitespace. `WER_raw` is also reported (NFC + whitespace only).
- **Scoring:** corpus WER = Σ(S+D+I) / Σ(reference words) with jiwer; CER on normalised text with
  spaces removed; 95% CI from a 1,000-sample utterance bootstrap with a fixed seed.
- **Decoding:** greedy everywhere, batch 1, no language model. Whisper models use
  `max_new_tokens=444`: Malayalam needs ~27 Whisper tokens per second of audio, and the usual
  225 cap silently truncated clips longer than ~9 s
  ([Part 3](https://intuitivetutorial.com/2026/10/11/part-3-retraining-my-malayalam-asr-model-and-a-bug-in-my-own-evaluation/)).
  `Near cap` counts clips whose output reached the cap; it should be 0.
- **Speed:** the process is pinned to 2 physical cores (`sched_setaffinity`) and math libraries
  get 2 threads. Timing covers only the model call on already-decoded audio, after one cold and
  three warm calls. RTF = compute time / audio time (below 1 is faster than real time);
  RTFx = 1/RTF. Peak RAM is the process's max RSS.
- **Versions:** Transformers is pinned to 4.46.3, the version behind the published numbers.

## Add a model

Add an entry to `MODELS` in [`mlasr_eval/models.py`](mlasr_eval/models.py). If it needs a new
engine, add an adapter in `mlasr_eval/adapters/` implementing `load()` and
`transcribe(wav) -> Transcript`, and register it in `ADAPTERS`. Run `pytest` before opening a PR:
the tests build tiny random models offline to exercise every adapter.

## Related

- Models: [v2](https://huggingface.co/sajilck/whisper-small-malayalam-v2) ·
  [v1](https://huggingface.co/sajilck/whisper-small-malayalam)
- Blog: [Part 2 — leakage filtering](https://intuitivetutorial.com/2026/10/09/i-re-tested-my-malayalam-asr-model-on-clean-data-part-2/) ·
  [Part 3 — the token-cap bug](https://intuitivetutorial.com/2026/10/11/part-3-retraining-my-malayalam-asr-model-and-a-bug-in-my-own-evaluation/)
