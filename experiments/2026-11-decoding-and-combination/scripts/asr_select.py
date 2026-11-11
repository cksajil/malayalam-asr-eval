"""Pick the better of two Malayalam transcripts (e.g. Whisper and IndicConformer) with a character n-gram LM.

Rule (no tuned parameters): normalise each transcript, write it as space-separated characters with "|" for
word boundaries, and keep the one with the higher average log10-probability per character under the LM.
On the 231-clip leakage-free test set (malayalam-asr-eval balanced231), Whisper-small-ml-v2 greedy +
IndicConformer-600M CTC with this rule and `ml_char5_pruned.arpa.gz` scores 28.5% WER / 3.35% CER
(Whisper alone: 31.2% / 4.04%).

Pure Python (reads a standard ARPA file, backoff scoring identical to KenLM), so it needs no compiled package.
Memory: ~0.5 GB for the shipped 5-gram.
"""
from __future__ import annotations

import gzip
import re
import unicodedata

CHILLU = {"ണ്‍": "ൺ", "ന്‍": "ൻ", "ര്‍": "ർ", "ല്‍": "ൽ", "ള്‍": "ൾ", "ക്‍": "ൿ"}


def normalise(s: str) -> str:
    """Same normalisation as malayalam-asr-eval: NFC, atomic chillu, no ZWJ/ZWNJ, punctuation -> space."""
    s = unicodedata.normalize("NFC", str(s))
    for k, v in CHILLU.items():
        s = s.replace(k, v)
    s = s.replace("‍", "").replace("‌", "")
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    return re.sub(r"\s+", " ", s).strip()


class ArpaLM:
    """Backoff n-gram LM from an ARPA file (optionally .gz). score() matches kenlm.Model.score(bos=True, eos=True)."""

    def __init__(self, path: str):
        opener = gzip.open if path.endswith(".gz") else open
        self.p, self.b = {}, {}
        order = 0
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("ngram ") or line == "\\data\\" or line == "\\end\\":
                    continue
                if line.startswith("\\") and line.endswith("-grams:"):
                    order = int(line[1:line.index("-")])
                    continue
                parts = line.split("\t")
                key = parts[1]
                self.p[key] = float(parts[0])
                if len(parts) > 2 and float(parts[2]) != 0.0:
                    self.b[key] = float(parts[2])
        self.order = order
        self.unk = self.p.get("<unk>", -100.0)

    def _logp(self, ctx: list[str], w: str) -> float:
        backoff = 0.0
        for i in range(len(ctx) + 1):          # longest context first
            h = ctx[i:]
            key = " ".join(h + [w])
            if key in self.p:
                return self.p[key] + backoff
            if h:
                backoff += self.b.get(" ".join(h), 0.0)
        return self.unk + backoff

    def score(self, sentence: str) -> float:
        words = sentence.split()
        hist = ["<s>"]
        total = 0.0
        for w in words + ["</s>"]:
            total += self._logp(hist[-(self.order - 1):], w)
            hist.append(w)
        return total


def per_char(lm: ArpaLM, text: str) -> float:
    t = normalise(text)
    if not t:
        return float("-inf")
    return lm.score(" ".join(t.replace(" ", "|"))) / len(t)


def select(lm: ArpaLM, candidates: list[str]) -> int:
    """Index of the candidate with the best per-character LM score (ties -> the first, i.e. Whisper)."""
    scores = [per_char(lm, c) for c in candidates]
    return max(range(len(candidates)), key=lambda i: (scores[i], -i))
