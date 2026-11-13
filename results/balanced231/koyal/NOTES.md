# Koyal 120M on the 231-clip benchmark (November 2026)

`adalat-ai/koyal-ml-120m-1.0` (CC-BY-4.0): NeMo Conformer hybrid TDT-CTC with greedy TDT decoding,
FP32, run on Kaggle CPU pinned to 2 physical cores. It ran in the same session as a whisper-v2 re-run
(`whisper-v2-2026-11-13`), which reproduced the published 31.2% WER / 4.0% CER (Δ −0.04 / +0.04 pts) with
identical outputs on all 231 clips. NeMo 3.0.0, torch 2.11.0+cpu. On clip 1, the adapter's direct
tensor decoding gave the same text as `model.transcribe()`.

## Koyal vs v2 vs IndicConformer

| | Koyal 120M | whisper-v2 (244M) | IndicConformer CTC (600M) |
|---|---|---|---|
| WER % (95% CI) | 30.3 (27.1–33.7) | 31.2 (27.9–34.6) | 30.1 (27.2–33.3) |
| CER % | 6.0 | 4.0 | 4.0 |
| WER / CER % no-digit | 28.3 / 4.0 (34 clips dropped) | 31.1 / 4.0 (32 dropped) | 29.7 / 3.9 (33 dropped) |
| Exact-match clips | 26% | 28% | 27% |
| Speed vs v2 (same session) | 12.6× | 1.0× | 13.8× (Oct session) |
| RTF (2 cores) | 0.17 | 2.12 | 0.14 (Oct session) |
| Peak RAM (own process) | 2.1 GB | 2.5 GB | 3.9 GB |

Paired bootstrap of the WER difference (10,000 resamples over clips): Koyal − v2 = −0.8 pts (95% CI −3.7
to +2.1), Koyal − IndicConformer = +0.2 pts (−2.7 to +3.0). Neither difference is significant.

- **Digits explain Koyal's higher CER.** Koyal writes numbers as digits in 28 of 231 outputs (v2 and
  IndicConformer in none). On the 34 clips where its reference or output contains a number, its CER is
  17.9% (v2: 4.6%). On the rest it is 4.0%, the same as v2.
- **Punctuation costs nothing.** Koyal punctuates 137 of 231 outputs, and `norm_ml` removes punctuation
  before scoring.
- **Word splits and joins hit all three models about equally.** The number of clips whose output differs
  from the reference only in word boundaries: Koyal 21, v2 24, IndicConformer 20. Koyal's card reports a
  sandhi-tolerant WER that scores these as correct; plain WER, used here, counts them as errors.
- **Training-data overlap.** The card lists Shrutilipi, Common Voice, IndicTTS, IMaSC and OpenSLR 63, so
  five of the six test sources may have been seen in training. Only the 5 SMC clips are unseen (25.0% WER),
  too few to mean anything. The card names corpora, not utterances, so whether these exact test sentences
  were in Koyal's training set is unknown.

## The 5 clips where Koyal and v2 disagree most

Ranked by word edit distance between the two outputs (after `norm_ml`). "Errors" = S+D+I against the
reference.

| # | Source | Koyal / v2 errors (ref words) | Main error type |
|---|---|---|---|
| 20 | commonvoice | 10 / 7 (11) | Koyal: digits + word splits; v2: real misrecognition |
| 84 | imasc | 5 / 3 (12) | Koyal: digits + spelling variant |
| 222 | shrutilipi | 0 / 6 (19) | v2: word joins + real misrecognition |
| 78 | imasc | 4 / 1 (7) | Koyal: word split + spelling variant |
| 202 | shrutilipi | 5 / 8 (12) | Both: real misrecognition (place names) |

**#20 (commonvoice).** Koyal: digits and splits. v2: real misrecognition.
- REF: തീരദേശത്ത് വേലിയേറ്റരേഖയില്നിന്നും അൻപത് മീറ്ററിനുള്ളില് താമസിക്കുന്ന പതിനെണ്ണായിരത്തോളം മത്സ്യത്തൊഴിലാളി കുടുംബങ്ങളെ മാറ്റി സുരക്ഷിതമേഖലയില് പുനഃരധിവസിപ്പിക്കും.
- Koyal: തീരദേശത്ത് വേലിയേറ്റ രേഖയിൽ നിന്നും **50** മീറ്ററിനുള്ളിൽ താമസിക്കുന്ന **18000-ത്തോളം** മത്സ്യത്തൊഴിലാളി കുടുംബങ്ങളെ മാറ്റി സുരക്ഷിത മേഖലയിൽ പുനരധിവസിപ്പിക്കും.
- v2: തീരദേശത്ത് വേലിയേറ്റ രേഖയായിൽ നിന്നും അമ്പതുമീറ്റന്നൂളിൽ താമസിക്കുന്ന പതിനെണ്ണായിരത്തോളം മത്സ്യത്തൊഴിലാളി കുടുംബങ്ങളെ മാറ്റിസുരഹിതമേയിൽ പുനര്ദിവസിപ്പിക്കും.
- Koyal's transcript is essentially correct but scores worse (10 vs 7). The errors come from two digit forms,
  three word splits (വേലിയേറ്റ രേഖയിൽ, സുരക്ഷിത മേഖലയിൽ) and the reference's own misspelling
  (പുനഃരധിവസിപ്പിക്കും). v2 has real errors (സുരഹിതമേയിൽ, മീറ്റന്നൂളിൽ).

**#84 (imasc).** Koyal: digits plus a spelling variant.
- REF: സൌരയൂഥത്തിന്റെ ബാഹ്യ അതിർത്തിക്കു സമീപത്ത് സ്ഥിതി ചെയ്യുന്ന മൂന്ന് വസ്തുക്കളെ മൂന്നൂറ്റി പതിമൂന്ന് എന്നാണ് നിർവചിച്ചിട്ടുള്ളത്
- Koyal: സൗരയൂഥത്തിന്റെ ബാഹ്യ അതിർത്തിക്ക് സമീപത്ത് സ്ഥിതി ചെയ്യുന്ന **3** വസ്തുക്കളെ **313** എന്നാണ് നിർവചിച്ചിട്ടുള്ളത്.
- v2: സൌരയൂഥത്തിന്റെ ബാഹ്യ അതിർത്തിക്കു സമീപത്ത് സ്ഥിതിചെയ്യുന്ന മൂന്ന് വസ്തുക്കളെ മുന്നൂറ്റി പതിമൂന്ന് എന്നാണ് നിർവചിച്ചിട്ടുള്ളത്
- Koyal has no recognition errors. Its 5 "errors" are "3" and "313" (4 words) plus the vowel-sign variant
  സൗ/സൌ. Note that `norm_ml` does not unify സൗ and സൌ. v2 has one real error (മുന്നൂറ്റി) and one join.

**#222 (shrutilipi).** v2: word joins plus a real misrecognition.
- REF: രജിസ്ട്രേഷൻ നമ്പർ ഒറ്റ അക്കങ്ങളിൽ അവസാനിക്കുന്ന വാഹനങ്ങൾ … ഇരട്ട അക്കങ്ങളിൽ അവസാനിക്കുന്നവ ചൊവ്വ വ്യാഴം ശനി ദിവസങ്ങളിലും മാത്രം സഞ്ചരിക്കാം
- Koyal: identical to the reference (plus a final full stop).
- v2: … ഒറ്റാക്കങ്ങളിൽ … ഇരട്ടാക്കങ്ങളിൽ അവസാനിക്കുന്ന ചൊവ്വ **വേളം** ശനി …
- v2 joins two words by vowel elision (ഒറ്റ അക്കങ്ങളിൽ → ഒറ്റാക്കങ്ങളിൽ), which costs 2 errors each time. It
  also misrecognises വ്യാഴം as വേളം.

**#78 (imasc).** Koyal: a word split plus a spelling variant.
- REF: മുൻവർണവെറിയൻവാഴ്ചക്കെതിരെ സന്ധിയില്ലാതെ പോരാടിയ വെളുത്ത വർഗക്കാരിയായ നേതാവ് അന്തരിച്ചു
- Koyal: മുൻ വർണവെറിയൻ വാഴ്ചക്കെതിരെ സന്ധിയില്ലാതെ പോരാടിയ വെളുത്ത വർഗ്ഗക്കാരിയായ നേതാവ് അന്തരിച്ചു.
- v2: മുൻവർണവെറിയൻവാഴ്ചക്കെതിരെ സന്തിയില്ലാതെ പോരാടിയ വെളുത്ത വർഗക്കാരിയായ നേതാവ് അന്തരിച്ചു
- Koyal's transcript is arguably the better-written one. It is charged 4 errors for splitting a 3-word
  compound and for വർഗ്ഗ vs വർഗ, both accepted spellings. v2 has one real error (സന്തി for സന്ധി).

**#202 (shrutilipi).** Both: real misrecognition of place names.
- REF: മൽക്കൻഗിരി ജില്ലയിലെ കുർമാനൂർ സന്യാസിഗുഡ പാപ്പൂലൂർ തുട ങ്ങിയ മേഖയിൽ നിന്ന് …
- Koyal: **തൽക്കൽഗിരി** … **പാപ്പല്ലൂർ** തുടങ്ങിയ മേഖലയിൽ നിന്ന് …
- v2: **കൽക്കൻഗിനി** … **കുർമ്മാനൂർ** **സന്ന്യാസികുഡ** … ആയുധ ശേഖരവും …
- Both models garble the Odisha place names. Both are also charged for the reference's typos (തുട ങ്ങിയ,
  മേഖയിൽ).

Also worth noting: on #6 ("രണ്ടു ദിവസം" → "2 ദിവസം") Koyal's only error is the digit. v2 makes 6:
three are chillus written as consonant + virama without ZWJ (ആള്, കഴിഞ്ഞപ്പോള്, പരിപാടികള് for
ആൾ, കഴിഞ്ഞപ്പോൾ, പരിപാടികൾ), which `norm_ml` does not map to atomic chillus. The other three are real
errors (ഷട്ടൗണ്, തുടങ്ങും).

## Caveats

- **Koyal's per-source numbers may be optimistic.** Five of the six test sources are in its training
  corpora, and our leakage filter only covers v2's training data.
- **The no-digit WER drops a different clip set for each model** (32–34 clips each), so it is not a
  perfectly paired comparison. Its number-word list is hand-built (see `mlasr_eval/normalize.py`) and
  checked against the 231 references.
- **Plain WER penalises Koyal's style.** Koyal's rich, written-style output (digits, word splits,
  standard spellings such as സൗ and ഗ്ഗ) is charged as errors against these verbatim references. The
  normalised WER understates its usefulness for dictation-style output and overstates it for verbatim use.
- **One chillu spelling is not normalised.** `norm_ml` maps consonant + virama + ZWJ to atomic chillus, but
  not consonant + virama alone (ള്, ന്, ർ…), which v2 sometimes writes on Common Voice clips. That costs v2
  some errors that are only spelling. Changing it would change every published number, so it is noted
  here, not fixed.
- **Some references contain typos** (e.g. #20, #202), which charge every model equally.
- **IndicConformer's speed comes from the October session.** It is compared through "speed vs v2" in its
  own session, not measured alongside Koyal.
