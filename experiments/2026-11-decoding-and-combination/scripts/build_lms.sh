#!/bin/bash
# KenLM models used by rescore2.py, combine.py and selector_eval.py (about 1.3 GB in all; not committed).
# Needs KenLM's lmplz/build_binary built with -DKENLM_MAX_ORDER=8 (the PyPI kenlm module: MAX_ORDER=8 pip install ...;
# on Python 3.13 regenerate python/kenlm.cpp with cython first). Set KENLM_BIN to the folder with lmplz.
set -e
cd "${EXP_DIR:-$(dirname "$0")/..}/lm"
B=${KENLM_BIN:-.}
$B/lmplz -o 4 -S 25% --discount_fallback -T . < train.txt > train_word4.arpa && $B/build_binary train_word4.arpa train_word4.bin
$B/lmplz -o 4 -S 25% --discount_fallback -T . < wiki.txt  > wiki_word4.arpa  && $B/build_binary wiki_word4.arpa wiki_word4.bin
$B/lmplz -o 8 -S 25% --discount_fallback -T . < train.chr > train_chr8.arpa  && $B/build_binary train_chr8.arpa train_chr8.bin
$B/lmplz -o 8 -S 25% --discount_fallback -T . < wiki.chr  > wiki_chr8.arpa   && $B/build_binary wiki_chr8.arpa wiki_chr8.bin
# the small selector shipped in the model repo (selector/ml_char5_pruned.arpa.gz): train + wiki, 5-gram, pruned
cat train.chr wiki.chr > both.chr
$B/lmplz -o 5 -S 30% --discount_fallback --prune 0 1 2 3 -T . < both.chr > both_chr5p.arpa && $B/build_binary both_chr5p.arpa both_chr5p.bin
gzip -9 -c both_chr5p.arpa > ml_char5_pruned.arpa.gz
rm -f *.arpa.tmp
