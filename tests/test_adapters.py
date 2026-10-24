import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from mlasr_eval import models
from mlasr_eval.cpu import configure
from mlasr_eval.report import write_report
from mlasr_eval.runner import run_model

WAV = (0.01 * np.random.default_rng(0).standard_normal(16000 * 2)).astype(np.float32)


def test_hf_whisper_greedy_and_token_count(tiny_whisper_dir):
    from mlasr_eval.adapters.hf_whisper import HFWhisper
    a = HFWhisper(tiny_whisper_dir, threads=1, max_new_tokens=20)
    a.load()
    t = a.transcribe(WAV)
    assert isinstance(t.text, str)
    assert 0 <= t.n_tokens <= 20
    assert a.info()["generation_config_borrowed_from"] is None


def test_hf_whisper_borrows_language_tables(tiny_whisper_no_langs, tiny_whisper_dir):
    from mlasr_eval.adapters.hf_whisper import HFWhisper
    a = HFWhisper(tiny_whisper_no_langs, threads=1, max_new_tokens=10, base_generation_config=tiny_whisper_dir)
    a.load()
    assert a.transcribe(WAV).n_tokens <= 10
    assert a.info()["generation_config_borrowed_from"] == tiny_whisper_dir


def test_hf_whisper_444_is_allowed(tiny_whisper_dir):
    """4 prompt tokens + 444 new tokens = 448, Whisper's decoder limit; generate must accept it."""
    from mlasr_eval.adapters.hf_whisper import HFWhisper
    a = HFWhisper(tiny_whisper_dir, threads=1)
    a.load()
    assert a.transcribe(WAV).n_tokens <= 444


def test_hf_whisper_long_form(tiny_whisper_dir):
    from mlasr_eval.adapters.hf_whisper import HFWhisper
    a = HFWhisper(tiny_whisper_dir, threads=1, max_new_tokens=8)
    a.load()
    long = np.tile(WAV, 18)        # 36 s
    t = a.transcribe(long)
    assert t.extra["long_form"] and a.long_form_clips == 1


def test_faster_whisper_converts_and_decodes(tiny_whisper_dir, tmp_path):
    pytest.importorskip("faster_whisper")
    from mlasr_eval.adapters.faster_whisper_ct2 import FasterWhisper
    a = FasterWhisper(tiny_whisper_dir, threads=1, ct2_dir=str(tmp_path / "ct2"), max_new_tokens=444)
    a.load()
    assert (tmp_path / "ct2" / "model.bin").exists() and (tmp_path / "ct2" / "tokenizer.json").exists()
    t = a.transcribe(WAV)
    assert isinstance(t.text, str) and t.n_tokens <= 444
    assert a.info()["ct2_size_MB"] > 0


def test_wav2vec2_ctc(tiny_w2v2_dir):
    from mlasr_eval.adapters.wav2vec2_ctc import Wav2Vec2CTC
    a = Wav2Vec2CTC(tiny_w2v2_dir, threads=1)
    a.load()
    t = a.transcribe(WAV)
    assert isinstance(t.text, str) and t.n_tokens is None


def test_indic_conformer_calls_remote_model(monkeypatch):
    import transformers
    from mlasr_eval.adapters.indic_conformer import IndicConformer
    calls = {}

    class Fake:
        def eval(self):
            return self

        def __call__(self, wav, lang, mode):
            calls.update(shape=tuple(wav.shape), lang=lang, mode=mode)
            return "കേരളം"

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: Fake())
    a = IndicConformer(threads=1)
    a.load()
    assert a.transcribe(WAV).text == "കേരളം"
    assert calls == {"shape": (1, len(WAV)), "lang": "ml", "mode": "ctc"}


def test_registry_builds_every_adapter_class():
    for name in models.MODELS:
        adapter = models.build(name, threads=2)
        assert adapter.model_id == models.MODELS[name]["model_id"]


def test_runner_end_to_end_and_report(tiny_whisper_dir, synthetic_eval, tmp_path, monkeypatch):
    monkeypatch.setitem(models.MODELS, "tiny", dict(adapter="hf_whisper", model_id=tiny_whisper_dir,
                                                    kwargs=dict(max_new_tokens=12), role="test",
                                                    expect={"WER": 0.3, "CER": 0.04}))
    results = tmp_path / "results"
    run_model("tiny", synthetic_eval, str(results), threads=1, warmup=1)
    out = results / "balanced231" / "tiny"
    assert (out / "per_utterance.csv").exists() and not (out / "partial.csv").exists()
    saved = json.load(open(out / "summary.json"))
    assert saved["overall"]["n_clips"] == 12 and saved["overall"]["RTF"] > 0
    assert set(saved["per_source"]) <= {"shrutilipi", "commonvoice", "indictts", "imasc", "openslr63", "smc"}
    assert saved["overall"]["near_cap"] == int((__import__("pandas").read_csv(out / "per_utterance.csv").n_tokens >= 8).sum())
    assert "sanity" in saved
    md = open(write_report(str(results), "balanced231")).read()
    assert "tiny" in md and "Sanity check" in md


def test_runner_resumes_from_partial(tiny_whisper_dir, synthetic_eval, tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setitem(models.MODELS, "tiny", dict(adapter="hf_whisper", model_id=tiny_whisper_dir,
                                                    kwargs=dict(max_new_tokens=4)))
    out = tmp_path / "results" / "balanced231" / "tiny"
    out.mkdir(parents=True)
    pd.DataFrame([{"i": i, "hyp": "കേരളം", "total_s": 0.1, "n_tokens": 1} for i in range(5)]).to_csv(
        out / "partial.csv", index=False)
    run_model("tiny", synthetic_eval, str(tmp_path / "results"), threads=1, warmup=0)
    res = pd.read_csv(out / "per_utterance.csv")
    assert len(res) == 12 and (res.hyp.iloc[:5] == "കേരളം").all()


def test_cpu_configure_pins_affinity():
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("no affinity API")
    before = os.sched_getaffinity(0)
    try:
        info = configure(cores=1, threads=1)
        assert os.sched_getaffinity(0) == set(info["pinned_logical_cpus"])
        assert info["pinned_physical_cores"] == 1 and os.environ["OMP_NUM_THREADS"] == "1"
    finally:
        os.sched_setaffinity(0, before)


def test_faster_whisper_is_not_capped_at_224_tokens(tiny_whisper_dir, tmp_path):
    """CTranslate2 emits at most max_length // 2 tokens; the adapter must double the budget.
    The random model never emits end-of-text, so it runs to whatever cap is in force."""
    pytest.importorskip("faster_whisper")
    from mlasr_eval.adapters.faster_whisper_ct2 import FasterWhisper
    a = FasterWhisper(tiny_whisper_dir, threads=1, ct2_dir=str(tmp_path / "ct2"), max_new_tokens=300)
    a.load()
    assert a.transcribe(WAV).n_tokens >= 299     # was 150 before the fix (304 // 2 - prompt)


def test_runner_warns_about_hidden_token_ceiling(tiny_whisper_dir, synthetic_eval, tmp_path, monkeypatch):
    # nominal cap 444 is never reached, but every clip stops at 10: the near-cap column can't see it
    from mlasr_eval.adapters import hf_whisper
    orig = hf_whisper.HFWhisper.transcribe

    def capped(self, wav):
        t = orig(self, wav)
        t.n_tokens = 10
        return t
    monkeypatch.setattr(hf_whisper.HFWhisper, "transcribe", capped)
    monkeypatch.setitem(models.MODELS, "tiny", dict(adapter="hf_whisper", model_id=tiny_whisper_dir,
                                                    kwargs=dict(max_new_tokens=4)))
    s = run_model("tiny", synthetic_eval, str(tmp_path / "r"), threads=1, warmup=0)
    assert s["overall"]["near_cap"] == 12 or s["warnings"]
    assert any("exactly 10 tokens" in w for w in s["warnings"])
