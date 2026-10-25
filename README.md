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
| `vrclc-medium` | vrclc/Whisper-medium-Malayalam | Transformers FP32 (gated: request access) |

Optional: `whisper-v1`, `indicconformer-rnnt`, `xlsr-ml-shawn`. Parakeet and Nemotron were
checked and don't support Malayalam. Details, and which models may have trained on the same
public corpora as the test rows, are in [docs/MODELS.md](docs/MODELS.md).

## Results (October 2026)

231 leakage-free clips (18.6 min of audio), Kaggle CPU (Intel Xeon 2.20 GHz) pinned to 2 physical
cores, batch 1, greedy decoding. Full tables, per-source numbers and per-clip outputs:
[results/balanced231/LEADERBOARD.md](results/balanced231/LEADERBOARD.md).

| Model | WER % (95% CI) | CER % | WER % on unseen sources¹ | Speed vs v2² | RTF³ | Peak RAM |
|---|---|---|---|---|---|---|
| IndicConformer 600M, CTC | **30.1** (27.2–33.3) | **4.0** | 31.3 | **13.8×** | **0.14** | 3.9 GB |
| whisper-small-malayalam-v2 | 31.2 (27.9–34.6) | **4.0** | 31.2 | 1.0× | 2.28 | 2.4 GB |
| v2, CTranslate2 int8 | 32.7 (29.6–36.1) | 4.2 | 32.7 | 1.9× | 1.22 | 2.6 GB⁴ |
| vrclc/Whisper-medium-Malayalam | 39.7 (33.1–47.3) | 10.1 | 52.9 | 0.35× | 6.58 | 5.2 GB |
| MMS-1b-all (wav2vec2 CTC) | 52.5 (49.0–56.1) | 9.2 | 51.5 | 2.2× | 0.86 | 8.8 GB |
| kavyamanohar/whisper-small-malayalam | 87.0 (82.4–93.7) | 29.2 | 88.8 | 0.84× | 2.30 | 2.5 GB |

¹ WER after dropping clips from corpora in that model's own published training data: the fair
column for third-party models (see [docs/MODELS.md](docs/MODELS.md)).
² How many times faster than v2 measured on the same machine; speeds on the "same" Kaggle CPU varied
by ~20% between sessions, so this is the column to compare.
³ Compute time ÷ audio time; below 1 is faster than real time.
⁴ Overstated: this run converted the model to int8 in the same process, so the peak includes the FP32 model used for conversion. The int8 model itself is 255 MB (FP32: 967 MB). Conversion now runs in a separate process.

### Findings

- **A CTC model matches v2's accuracy at ~14× the speed.** IndicConformer is statistically tied with
  v2 (confidence intervals overlap almost completely) and runs 7× faster than real time on 2 cores.
  It is the only model here that is both accurate and real-time on this hardware.
- **Optimising autoregressive Whisper isn't enough.** CTranslate2 int8 makes v2 1.9× faster for 1.5
  WER points, but it is still slower than real time (RTF 1.22). Whisper's cost grows with the number
  of tokens, and its tokenizer needs ~27 tokens per second of Malayalam speech.
- **Overlap with training data distorts rankings.** vrclc medium scores 0.0% WER on SMC, 8.4% on
  OpenSLR 63 and 16.1% on IMaSC, the corpora it was trained on, but 62–70% on Shrutilipi and Common
  Voice. Its headline 39.7% becomes 52.9% on sources it hasn't seen.
- **Hidden output limits are easy to miss.** The first CTranslate2 run stopped every transcript at
  224 tokens (CTranslate2 generates at most half of `max_length`) and scored 38.0% WER. The runner
  now flags any run where several outputs end at exactly the same token count.

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
