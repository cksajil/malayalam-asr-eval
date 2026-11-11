#!/bin/bash
# The order the experiments were run in (Nov 2026, 4-vCPU CPU box, no GPU). Each step writes runs/<name>.csv/.json.
set -e
cd "$(dirname "$0")"
python prepare.py                                   # test231 (= balanced231) + 90-clip dev set, decoded audio -> data/
python import_published.py                          # published whisper-v2 / fw-v2-int8 / indicconformer outputs -> runs/pub_*
python error_analysis.py
# CTranslate2 FP32 copy of v2 (needs transformers >= 4.56 for the converter) -> models/ct2-f32
ct2-transformers-converter --model sajilck/whisper-small-malayalam-v2 --output_dir ../models/ct2-f32 --quantization float32
python ct2_nbest.py test231 ct2-f32 1 1 test231_ct2_greedy      # identical to the published FP32 run (231/231)
python ct2_nbest.py dev ct2-f32 1 1 dev_ct2_greedy
python ct2_nbest.py dev ct2-f32 1 1 dev_ct2_greedy_loud loud
python ct2_nbest.py dev ct2-f32 1 1 dev_ct2_greedy_trimloud trim+loud
python ct2_nbest.py dev ct2-f32 5 5 dev_ct2_beam5
python ct2_nbest.py test231 ct2-f32 5 5 test231_ct2_beam5
python get_text.py && python build_lm_text.py && ./build_lms.sh
python tf_score.py dev dev_ct2_beam5 dev_b5                                     # teacher-forced AM scores
python tf_score.py test231 test231_ct2_greedy test_g1c pub_indicconformer-ctc_test
python tf_score.py test231 test231_ct2_beam5 test_b5c pub_indicconformer-ctc_test
python rescore2.py dev_b5 test_b5c r2_b5 --drop-extra                           # N-best LM rescoring, tuned on dev
python rescore2.py dev_b5 test_g1c r2_g1c                                        # AM+LM pick incl. IndicConformer
python combine.py pub_whisper-v2_test pub_indicconformer-ctc_test comb_pubgreedy # LM-only pick, no tuned weights
python combine.py test231_ct2_beam5 pub_indicconformer-ctc_test comb_test231_ct2_beam5
python combine.py r2_b5_word_train pub_indicconformer-ctc_test comb_r2_b5_word_train
python selector_eval.py
python cpu_speed.py
python fw_check.py ../models/ct2-int8           # pre-converted int8 copy (model repo ct2-int8/) vs published int8 run
