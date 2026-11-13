# Malayalam ASR leaderboard — `balanced231`

231 leakage-free clips (18.64 min). CPU: Intel(R) Xeon(R) CPU @ 2.20GHz, thread count is recorded in each run's summary and is not the same for every row, batch 1, greedy decoding. RTF below 1 is faster than real time. Speed vs v2 = whisper-v2's RTF on the same machine ÷ this model's RTF (2.0 = twice as fast); use it rather than RTF to compare rows from different Kaggle sessions. Rows with no recorded session have no Speed vs v2 value.

**WER % / CER % no-digit** drop every clip whose reference or that model's hypothesis contains a digit or a Malayalam number word (`No-digit excluded` = how many). References spell numbers out, so a model that writes digits (Koyal) loses points on formatting, not recognition. Because the excluded set depends on each model's own output, compare rows with the excluded count in mind.

**WER % unseen sources** drops the clips from corpora in that model's own training data (marked * below). It is the fair column for comparing third-party models.

## Overall (sorted by WER)

| Model                 | Checkpoint                                  | Params   | Engine                        |   WER % | 95% CI    |   CER % |   WER % no-digit |   CER % no-digit |   No-digit excluded | WER % unseen sources   |   Speed vs v2 |   RTF |   RTFx |   p90 latency s |   Peak RAM GB |   Near cap | Max tokens   |   Empty |   Loops |
|:----------------------|:--------------------------------------------|:---------|:------------------------------|--------:|:----------|--------:|-----------------:|-----------------:|--------------------:|:-----------------------|--------------:|------:|-------:|----------------:|--------------:|-----------:|:-------------|--------:|--------:|
| indicconformer-ctc    | ai4bharat/indic-conformer-600m-multilingual | 600M     | HF remote code (ONNX Runtime) |    30.1 | 27.2–33.3 |     4   |             29.7 |              3.9 |                  33 | 31.3 (181 clips)       |         13.77 |  0.14 |   7.13 |            1.07 |          3.88 |          0 | –            |       0 |       0 |
| koyal                 | adalat-ai/koyal-ml-120m-1.0                 | 120M     | NeMo FP32                     |    30.3 | 27.1–33.7 |     6   |             28.3 |              4   |                  34 | 25.0 (5 clips)         |         12.58 |  0.17 |   5.93 |            1.33 |          2.09 |          0 | –            |       0 |       0 |
| whisper-v2-2026-11-13 | sajilck/whisper-small-malayalam-v2          | 244M     | transformers FP32             |    31.2 | 27.9–34.6 |     4   |             31.1 |              4   |                  32 | 31.2 (231 clips)       |          1    |  2.12 |   0.47 |           15.88 |          2.45 |          0 | 439 (1×)     |       0 |       0 |
| whisper-v2-4threads   | sajilck/whisper-small-malayalam-v2          | 244M     | transformers FP32             |    31.2 | 27.9–34.6 |     4   |             31.1 |              4   |                  32 | 31.2 (231 clips)       |               |  2.03 |   0.49 |           14.65 |          4.51 |          0 | 439 (1×)     |       0 |       0 |
| whisper-v2            | sajilck/whisper-small-malayalam-v2          | 244M     | transformers FP32             |    31.2 | 27.9–34.6 |     4   |             31.1 |              4   |                  32 | 31.2 (231 clips)       |          1    |  2.28 |   0.44 |           17.24 |          2.41 |          0 | 439 (1×)     |       0 |       0 |
| fw-v2-int8            | sajilck/whisper-small-malayalam-v2          | 244M     | CTranslate2 int8              |    32.7 | 29.6–36.1 |     4.2 |             32.7 |              4.1 |                  32 | 32.7 (231 clips)       |          1.87 |  1.22 |   0.82 |            7.93 |          2.55 |          0 | 439 (1×)     |       0 |       0 |
| vrclc-medium          | vrclc/Whisper-medium-Malayalam              | 769M     | transformers FP32             |    39.7 | 33.1–47.3 |    10.1 |             40.9 |             10.8 |                  31 | 52.9 (150 clips)       |          0.35 |  6.58 |   0.15 |           47.52 |          5.16 |          3 | 444 (3×)     |       0 |       2 |
| vrclc-medium-4threads | vrclc/Whisper-medium-Malayalam              | 769M     | transformers FP32             |    39.7 | 33.1–47.3 |    10.1 |             40.9 |             10.8 |                  31 | 52.9 (150 clips)       |               |  5.69 |   0.18 |           39.72 |          6.77 |          3 | 444 (3×)     |       0 |       2 |
| mms-1b-all            | facebook/mms-1b-all                         | 965M     | transformers FP32             |    52.5 | 49.0–56.1 |     9.2 |             51.7 |              8.2 |                  34 | 51.5 (181 clips)       |          2.24 |  0.86 |   1.16 |            6.6  |          8.77 |          0 | –            |       1 |       0 |
| kavya-small           | kavyamanohar/whisper-small-malayalam        | 244M     | transformers FP32             |    87   | 82.4–93.7 |    29.2 |             88.1 |             29.3 |                  32 | 88.8 (181 clips)       |          0.84 |  2.3  |   0.44 |           17    |          2.46 |          1 | 444 (1×)     |       0 |       1 |

## WER / CER % by source

| Model                 | shrutilipi   | commonvoice   | indictts     | imasc        | openslr63    | smc         |
|:----------------------|:-------------|:--------------|:-------------|:-------------|:-------------|:------------|
| indicconformer-ctc    | 27.5 / 4.2 * | 38.0 / 6.5    | 25.4 / 2.7   | 34.8 / 3.4   | 24.7 / 3.6   | 45.0 / 8.5  |
| koyal                 | 30.0 / 5.6 * | 37.2 / 8.5 *  | 24.4 / 4.9 * | 34.1 / 6.9 * | 26.5 / 4.6 * | 25.0 / 2.3  |
| whisper-v2-2026-11-13 | 31.4 / 4.7   | 38.0 / 7.8    | 26.7 / 2.6   | 31.8 / 2.2   | 28.9 / 3.7   | 25.0 / 3.1  |
| whisper-v2-4threads   | 31.4 / 4.7   | 38.0 / 7.8    | 26.7 / 2.6   | 31.8 / 2.2   | 28.9 / 3.7   | 25.0 / 3.1  |
| whisper-v2            | 31.4 / 4.7   | 38.0 / 7.8    | 26.7 / 2.6   | 31.8 / 2.2   | 28.9 / 3.7   | 25.0 / 3.1  |
| fw-v2-int8            | 32.7 / 4.8   | 40.2 / 7.9    | 25.4 / 2.7   | 35.8 / 2.6   | 30.1 / 3.9   | 30.0 / 3.9  |
| vrclc-medium          | 62.3 / 20.8  | 70.5 / 20.7   | 26.4 / 3.1   | 16.1 / 1.2 * | 8.4 / 0.7 *  | 0.0 / 0.0 * |
| vrclc-medium-4threads | 62.3 / 20.8  | 70.5 / 20.7   | 26.4 / 3.1   | 16.1 / 1.2 * | 8.4 / 0.7 *  | 0.0 / 0.0 * |
| mms-1b-all            | 66.6 / 14.3  | 57.7 / 10.5 * | 43.7 / 5.1   | 47.5 / 7.3   | 35.5 / 6.8   | 35.0 / 6.2  |
| kavya-small           | 100.5 / 45.6 | 77.8 / 29.6 * | 82.3 / 19.8  | 82.3 / 20.3  | 82.5 / 23.1  | 80.0 / 22.5 |

\* The model's published training data includes this corpus, so these test sentences may have been seen in training.

## Sanity check against published numbers

- **whisper-v2-2026-11-13**: WER 31.2% vs 31.2% (-0.0 pts), CER 4.0% vs 4.0% (+0.0 pts) — matches
- **whisper-v2-4threads**: WER 31.2% vs 31.2% (-0.0 pts), CER 4.0% vs 4.0% (+0.0 pts) — matches
- **whisper-v2**: WER 31.2% vs 31.2% (-0.0 pts), CER 4.0% vs 4.0% (+0.0 pts) — matches

## Warnings

- **fw-v2-int8**: Peak RAM includes the one-time FP32-to-int8 conversion, which ran in the same process; the int8 runtime footprint is lower (model file 255 MB vs 967 MB). Conversion now runs in a separate process.
- **vrclc-medium-4threads**: 3 clips reached the 444-token cap (2 repetition loops, 1 repeated sentence); these are model failures, not truncation of real speech
- **vrclc-medium**: 3 clips reached the 444-token cap (2 repetition loops, 1 repeated sentence); these are model failures, not truncation of real speech

## Training-data overlap notes

- **fw-v2-int8**: Same weights as whisper-v2.
- **indicconformer-ctc**: Training data not stated on the card; AI4Bharat also published Shrutilipi, so the shrutilipi rows may overlap.
- **kavya-small**: Card lists Common Voice 13; commonvoice test sentences may overlap its training data.
- **koyal**: Adalat AI, NeMo Conformer hybrid TDT-CTC, greedy TDT. Card lists ~1,350 h from IndicVoices, Shrutilipi, SPRING-INX, Kathbath, Vaani, IMaSC, IndicTTS, FLEURS, ULCA, OpenSLR 63, Common Voice and Festvox IIITH, so shrutilipi, commonvoice, indictts, imasc and openslr63 rows may have been seen in training; only smc is unseen. Writes punctuation and digits (rich orthography): punctuation is removed by the normaliser, digits are not (see the no-digit columns).
- **mms-1b-all**: MMS fine-tuning data includes Common Voice and FLEURS; commonvoice rows may overlap.
- **vrclc-medium-4threads**: Gated (request access on its Hugging Face page). Card lists IMaSC, OpenSLR 63, SMC's Malayalam Speech Corpus and Festvox-IIITH as training data, so its imasc, openslr63 and smc rows may have been seen in training; shrutilipi, commonvoice and indictts are fair.
- **vrclc-medium**: Gated: request access on its Hugging Face page (the owners approve it) before running. Training data not checked yet; read the model card before quoting per-source numbers.
- **whisper-v2-2026-11-13**: Test set filtered against this model's training corpus.
- **whisper-v2-4threads**: Test set filtered against this model's training corpus.
- **whisper-v2**: Test set filtered against this model's training corpus.
