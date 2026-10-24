# Malayalam ASR leaderboard — `balanced231`

231 leakage-free clips (18.64 min). CPU: Intel(R) Xeon(R) CPU @ 2.20GHz, pinned to 2 physical cores (2 threads), batch 1, greedy decoding. RTF below 1 is faster than real time.

## Overall (sorted by WER)

| Model              | Checkpoint                                  | Params   | Engine                        |   WER % | 95% CI    |   CER % |   RTF |   RTFx |   p90 latency s |   Peak RAM GB |   Near cap | Max tokens   |   Empty |   Loops |
|:-------------------|:--------------------------------------------|:---------|:------------------------------|--------:|:----------|--------:|------:|-------:|----------------:|--------------:|-----------:|:-------------|--------:|--------:|
| indicconformer-ctc | ai4bharat/indic-conformer-600m-multilingual | 600M     | HF remote code (ONNX Runtime) |    30.1 | 27.2–33.3 |     4   |  0.14 |   7.13 |            1.07 |          3.88 |          0 | –            |       0 |       0 |
| whisper-v2         | sajilck/whisper-small-malayalam-v2          | 244M     | transformers FP32             |    31.2 | 27.9–34.6 |     4   |  1.93 |   0.52 |           14.73 |          2.42 |          0 | 439 (1×)     |       0 |       0 |
| mms-1b-all         | facebook/mms-1b-all                         | 965M     | transformers FP32             |    52.5 | 49.0–56.1 |     9.2 |  0.86 |   1.16 |            6.6  |          8.77 |          0 | –            |       1 |       0 |
| kavya-small        | kavyamanohar/whisper-small-malayalam        | 244M     | transformers FP32             |    87   | 82.4–93.7 |    29.2 |  2.3  |   0.44 |           17    |          2.46 |          1 | 444 (1×)     |       0 |       1 |

## WER / CER % by source

| Model              | shrutilipi   | commonvoice   | indictts    | imasc       | openslr63   | smc         |
|:-------------------|:-------------|:--------------|:------------|:------------|:------------|:------------|
| indicconformer-ctc | 27.5 / 4.2   | 38.0 / 6.5    | 25.4 / 2.7  | 34.8 / 3.4  | 24.7 / 3.6  | 45.0 / 8.5  |
| whisper-v2         | 31.4 / 4.7   | 38.0 / 7.8    | 26.7 / 2.6  | 31.8 / 2.2  | 28.9 / 3.7  | 25.0 / 3.1  |
| mms-1b-all         | 66.6 / 14.3  | 57.7 / 10.5   | 43.7 / 5.1  | 47.5 / 7.3  | 35.5 / 6.8  | 35.0 / 6.2  |
| kavya-small        | 100.5 / 45.6 | 77.8 / 29.6   | 82.3 / 19.8 | 82.3 / 20.3 | 82.5 / 23.1 | 80.0 / 22.5 |

## Sanity check against published numbers

- **whisper-v2**: WER 31.2% vs 31.2% (-0.0 pts), CER 4.0% vs 4.0% (+0.0 pts) — matches

## Training-data overlap notes

- **indicconformer-ctc**: Training data not stated on the card; AI4Bharat also published Shrutilipi, so the shrutilipi rows may overlap.
- **kavya-small**: Card lists Common Voice 13; commonvoice test sentences may overlap its training data.
- **mms-1b-all**: MMS fine-tuning data includes Common Voice and FLEURS; commonvoice rows may overlap.
- **whisper-v2**: Test set filtered against this model's training corpus.
