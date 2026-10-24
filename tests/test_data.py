import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlasr_eval.data import balanced_subset, clean_pool

REPO = Path(__file__).resolve().parent.parent
REAL_COUNTS = {"shrutilipi": 2354, "indictts": 126, "commonvoice": 123, "imasc": 44, "openslr63": 32, "smc": 5}


def _pool(counts=REAL_COUNTS, seed=1):
    rng = np.random.default_rng(seed)
    rows = [{"sentence": f"{s} {i}", "source": s} for s, n in counts.items() for i in range(n)]
    order = rng.permutation(len(rows))           # interleave sources, like the real split
    df = pd.DataFrame([rows[i] for i in order])
    df.index = rng.choice(10_000, len(df), replace=False)   # non-contiguous, like a filtered frame
    return df


def test_balanced_subset_matches_notebook_formula():
    pool = _pool()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = (pool.groupby("source", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), 50), random_state=42))
                  .reset_index(drop=True))
    ours = balanced_subset(pool)
    pd.testing.assert_frame_equal(ours[legacy.columns], legacy)


def test_balanced_subset_has_231_clips_with_real_source_sizes():
    sub = balanced_subset(_pool())
    assert len(sub) == 231
    assert sub.source.value_counts().to_dict() == {"shrutilipi": 50, "indictts": 50, "commonvoice": 50,
                                                   "imasc": 44, "openslr63": 32, "smc": 5}


def test_clean_csv_in_repo_has_expected_rows():
    clean = pd.read_csv(REPO / "own_corpus_test_clean.csv")
    assert len(clean) == 2684
    assert clean.source.value_counts().to_dict() == REAL_COUNTS


def test_clean_pool_matches_on_normalised_text_and_drops_leaked_rows():
    clean = pd.read_csv(REPO / "own_corpus_test_clean.csv")
    test = clean.rename(columns={"text": "sentence"}).copy()
    test["sentence"] = "  " + test["sentence"].str.replace(" ", "  ") + " "   # whitespace noise
    leaked = pd.DataFrame({"sentence": ["leaked sentence %d" % i for i in range(40)], "source": "imasc"})
    test = pd.concat([test, leaked], ignore_index=True)
    pool = clean_pool(test, clean)
    assert len(pool) == 2684 and "hf_row" in pool


def test_clean_pool_rejects_wrong_csv():
    clean = pd.read_csv(REPO / "own_corpus_test_clean.csv")
    test = clean.rename(columns={"text": "sentence"}).head(100)
    with pytest.raises(AssertionError):
        clean_pool(test, clean)
