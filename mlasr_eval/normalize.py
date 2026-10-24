"""Text normalisation — identical to the published v1/v2 clean-eval notebooks.

Do not change these functions without re-running every model: the leaderboard is only
comparable while every row uses exactly this normalisation.
"""
from __future__ import annotations

import re
import unicodedata

# Old-style chillu (consonant + virama + ZWJ) -> atomic chillu code points
CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ",
          "ല്‍": "ൽ", "ള്‍": "ൾ", "ക്‍": "ൿ"}


def norm_raw(s) -> str:
    """NFC + collapse whitespace. Used for the 'raw' WER column."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def norm_ml(s) -> str:
    """Headline normalisation: NFC, atomic chillus, drop ZWJ/ZWNJ, punctuation and symbols -> space."""
    s = unicodedata.normalize("NFC", str(s))
    for k, v in CHILLU.items():
        s = s.replace(k, v)
    s = s.replace("‍", "").replace("‌", "")
    s = "".join(" " if unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S") else c
                for c in s)
    return re.sub(r"\s+", " ", s).strip()


def match_key(s) -> str:
    """Key used to match own_corpus_test_clean.csv rows to the HF test split."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()
