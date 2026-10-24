import pandas as pd

from mlasr_eval import metrics
from mlasr_eval.normalize import CHILLU, norm_ml, norm_raw


def test_old_chillu_becomes_atomic():
    old = "അവന്‍"            # na + virama + ZWJ
    assert norm_ml(old) == "അവൻ"


def test_punctuation_symbols_zwnj_removed():
    assert norm_ml("കേരളം, ഇന്ത്യ! ₹100‌ ") == "കേരളം ഇന്ത്യ 100"


def test_raw_keeps_punctuation():
    assert norm_raw("  കേരളം,  ഇന്ത്യ ") == "കേരളം, ഇന്ത്യ"


def test_chillu_table_has_six_entries():
    assert len(CHILLU) == 6


def _scored(pairs, src="x"):
    df = pd.DataFrame([{"ref": r, "hyp": h, "source": src, "dur": 1.0, "total_s": 0.5} for r, h in pairs])
    return metrics.score(df)


def test_corpus_wer_is_error_sum_over_ref_words():
    df = _scored([("a b c d", "a b c d"), ("a b", "a x y")])   # 0 errors / 4, then 1 sub + 1 ins / 2
    assert metrics.corpus_wer(df) == 2 / 6


def test_cer_ignores_spaces():
    df = _scored([("ab cd", "abcd")])
    assert metrics.corpus_cer(df) == 0.0
    assert metrics.corpus_wer(df) == 1.0     # 1 sub + 1 del over 2 words


def test_empty_hypothesis_and_loop_flags():
    df = _scored([("a b", ""), ("a b", "x x x x x")])
    assert df.is_empty.tolist() == [True, False]
    assert df.loop.tolist() == [False, True]


def test_bootstrap_is_reproducible():
    df = _scored([("a b c", "a b d"), ("a b", "a b"), ("c d e f", "c d")] * 10)
    assert metrics.boot_ci(df) == metrics.boot_ci(df)


def test_summary_speed_fields():
    s = metrics.summarise(_scored([("a b", "a b")] * 4))
    assert s["RTF"] == 0.5 and s["RTFx"] == 2.0 and s["WER"] == 0.0
