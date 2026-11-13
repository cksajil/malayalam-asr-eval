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


# --- "no-digit" scoring variant -------------------------------------------------------------
# Some models (e.g. Koyal) write numbers as digits ("15") where the verbatim references spell
# them out ("പതിനഞ്ച്"). norm_ml can't reconcile the two, so the no-digit variant drops every
# clip whose reference OR hypothesis contains a digit or a Malayalam number word, for every model.
#
# Word-initial prefixes of cardinal/ordinal number words. Chosen against the 231 references to
# avoid look-alikes: ആയിരുന്നു ("was") is not ആയിരം, മൂന്നാർ (a place) is not മൂന്നാം,
# ലക്ഷ്യം ("goal") is not ലക്ഷം, പത്തനാപുരം is not പത്ത്, ഇരുപുറം is not ഇരുപത്.
# "ഒരു" (the indefinite article) and "ഒന്നും" ("nothing") are deliberately not matched.
NUMBER_WORD_PREFIXES = (
    "ഒന്ന്", "ഒന്നാം", "ഒന്നാമ",
    "രണ്ട",
    "മൂന്ന്", "മൂന്നു", "മൂന്നാം", "മൂന്നാമ", "മൂന്നില", "മൂന്നിന", "മൂന്നൂ",
    "നാല",
    "അഞ്ച", "അഞ്ഞൂ",
    "ആറാം", "ആറാമ", "ആറിന", "ആറില",
    "ഏഴ",
    "എട്ട",
    "ഒമ്പത", "ഒൻപത", "ഒൻപതാ",
    "പത്ത്", "പത്തു", "പത്താം", "പത്താമ", "പത്തിന", "പത്തില", "പത്തോ", "പത്തൊ", "പതിന",
    "ഇരുപത", "മുപ്പത", "അമ്പത", "അൻപത", "അറുപത", "എഴുപത", "എൺപത", "എണ്പത", "തൊണ്ണൂറ",
    "നൂറ", "ഇരുനൂറ", "ഇരുന്നൂറ", "മുന്നൂറ", "നാനൂറ", "അറുനൂറ", "അറുന്നൂറ", "എഴുനൂറ", "എഴുന്നൂറ",
    "എണ്ണൂറ", "തൊള്ളായിര",
    "ആയിരം", "ആയിരത്ത", "ലക്ഷം", "ലക്ഷത്ത", "ലക്ഷങ്ങ", "കോടി",
)
NUMBER_WORDS_EXACT = ("ആറ്", "ആറു")   # "ആറ്" alone; as a prefix it would also catch ആറ്റിൽ ("in the river")


def has_number(s) -> bool:
    """True if `s` (after norm_ml) has any numeric character (ASCII, Malayalam or other digits,
    Malayalam fraction signs) or a word starting with a Malayalam number-word prefix."""
    t = norm_ml(s)
    if any(unicodedata.category(c).startswith("N") for c in t):
        return True
    return any(w in NUMBER_WORDS_EXACT or w.startswith(NUMBER_WORD_PREFIXES) for w in t.split())
