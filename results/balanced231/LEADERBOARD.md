# Malayalam ASR leaderboard — `balanced231`

231 leakage-free clips (18.64 min). CPU: Intel(R) Xeon(R) CPU @ 2.20GHz, pinned to 2 physical cores (2 threads), batch 1, greedy decoding. RTF below 1 is faster than real time. Speed vs v2 = whisper-v2's RTF on the same machine ÷ this model's RTF (2.0 = twice as fast); use it rather than RTF to compare rows from different Kaggle sessions.

**WER % unseen sources** drops the clips from corpora in that model's own training data (marked * below). It is the fair column for comparing third-party models.

## Overall (sorted by WER)

| Model              | Checkpoint                                  | Params   | Engine                        |   WER % | 95% CI    |   CER % | WER % unseen sources   |   Speed vs v2 |   RTF |   RTFx |   p90 latency s |   Peak RAM GB |   Near cap | Max tokens   |   Empty |   Loops |
|:-------------------|:--------------------------------------------|:---------|:------------------------------|--------:|:----------|--------:|:-----------------------|--------------:|------:|-------:|----------------:|--------------:|-----------:|:-------------|--------:|--------:|
| indicconformer-ctc | ai4bharat/indic-conformer-600m-multilingual | 600M     | HF remote code (ONNX Runtime) |    30.1 | 27.2–33.3 |     4   | 31.3 (181 clips)       |         13.77 |  0.14 |   7.13 |            1.07 |          3.88 |          0 | –            |       0 |       0 |
| whisper-v2         | sajilck/whisper-small-malayalam-v2          | 244M     | transformers FP32             |    31.2 | 27.9–34.6 |     4   | 31.2 (231 clips)       |          1    |  2.28 |   0.44 |           17.24 |          2.41 |          0 | 439 (1×)     |       0 |       0 |
| fw-v2-int8         | sajilck/whisper-small-malayalam-v2          | 244M     | CTranslate2 int8              |    32.7 | 29.6–36.1 |     4.2 | 32.7 (231 clips)       |          1.87 |  1.22 |   0.82 |            7.93 |          2.55 |          0 | 439 (1×)     |       0 |       0 |
| vrclc-medium       | vrclc/Whisper-medium-Malayalam              | 769M     | transformers FP32             |    39.7 | 33.1–47.3 |    10.1 | 52.9 (150 clips)       |          0.35 |  6.58 |   0.15 |           47.52 |          5.16 |          3 | 444 (3×)     |       0 |       2 |
| mms-1b-all         | facebook/mms-1b-all                         | 965M     | transformers FP32             |    52.5 | 49.0–56.1 |     9.2 | 51.5 (181 clips)       |          2.24 |  0.86 |   1.16 |            6.6  |          8.77 |          0 | –            |       1 |       0 |
| kavya-small        | kavyamanohar/whisper-small-malayalam        | 244M     | transformers FP32             |    87   | 82.4–93.7 |    29.2 | 88.8 (181 clips)       |          0.84 |  2.3  |   0.44 |           17    |          2.46 |          1 | 444 (1×)     |       0 |       1 |

## WER / CER % by source

| Model              | shrutilipi   | commonvoice   | indictts    | imasc        | openslr63   | smc         |
|:-------------------|:-------------|:--------------|:------------|:-------------|:------------|:------------|
| indicconformer-ctc | 27.5 / 4.2 * | 38.0 / 6.5    | 25.4 / 2.7  | 34.8 / 3.4   | 24.7 / 3.6  | 45.0 / 8.5  |
| whisper-v2         | 31.4 / 4.7   | 38.0 / 7.8    | 26.7 / 2.6  | 31.8 / 2.2   | 28.9 / 3.7  | 25.0 / 3.1  |
| fw-v2-int8         | 32.7 / 4.8   | 40.2 / 7.9    | 25.4 / 2.7  | 35.8 / 2.6   | 30.1 / 3.9  | 30.0 / 3.9  |
| vrclc-medium       | 62.3 / 20.8  | 70.5 / 20.7   | 26.4 / 3.1  | 16.1 / 1.2 * | 8.4 / 0.7 * | 0.0 / 0.0 * |
| mms-1b-all         | 66.6 / 14.3  | 57.7 / 10.5 * | 43.7 / 5.1  | 47.5 / 7.3   | 35.5 / 6.8  | 35.0 / 6.2  |
| kavya-small        | 100.5 / 45.6 | 77.8 / 29.6 * | 82.3 / 19.8 | 82.3 / 20.3  | 82.5 / 23.1 | 80.0 / 22.5 |

\* The model's published training data includes this corpus, so these test sentences may have been seen in training.

## Sanity check against published numbers

- **whisper-v2**: WER 31.2% vs 31.2% (-0.0 pts), CER 4.0% vs 4.0% (+0.0 pts) — matches

## Warnings

- **fw-v2-int8**: Peak RAM includes the one-time FP32-to-int8 conversion, which ran in the same process; the int8 runtime footprint is lower (model file 255 MB vs 967 MB). Conversion now runs in a separate process.
- **vrclc-medium**: 3 clips reached the 444-token cap (2 repetition loops, 1 repeated sentence); these are model failures, not truncation of real speech

## Training-data overlap notes

- **fw-v2-int8**: Same weights as whisper-v2.
- **indicconformer-ctc**: Training data not stated on the card; AI4Bharat also published Shrutilipi, so the shrutilipi rows may overlap.
- **kavya-small**: Card lists Common Voice 13; commonvoice test sentences may overlap its training data.
- **mms-1b-all**: MMS fine-tuning data includes Common Voice and FLEURS; commonvoice rows may overlap.
- **vrclc-medium**: Gated: request access on its Hugging Face page (the owners approve it) before running. Training data not checked yet; read the model card before quoting per-source numbers.
- **whisper-v2**: Test set filtered against this model's training corpus.
