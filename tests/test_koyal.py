"""Koyal (NeMo) adapter registration and the no-digit scoring variant. NeMo is mocked."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mlasr_eval import metrics, models
from mlasr_eval.normalize import has_number


# --- registration -----------------------------------------------------------------------------

def test_koyal_is_registered():
    spec = models.MODELS["koyal"]
    assert spec["model_id"] == "adalat-ai/koyal-ml-120m-1.0"
    assert spec["adapter"] == "nemo_asr" and spec["python_env"] == "nemo"
    assert models.ADAPTERS["nemo_asr"] == ("mlasr_eval.adapters.nemo_asr", "NemoASR")
    assert "koyal" in models.DEFAULT_PLAN
    # Card's training data covers every test source except SMC.
    assert set(spec["overlap_sources"]) == {"shrutilipi", "commonvoice", "indictts", "imasc", "openslr63"}


def test_build_koyal_does_not_import_nemo():
    a = models.build("koyal", threads=2)
    assert type(a).__name__ == "NemoASR" and a.model_id == "adalat-ai/koyal-ml-120m-1.0"
    assert a.token_cap is None
    assert "nemo" not in sys.modules or isinstance(sys.modules["nemo"], types.ModuleType)


def test_koyal_unseen_wer_uses_only_smc(tmp_path):
    from mlasr_eval.report import unseen_wer
    srcs = ["shrutilipi", "commonvoice", "indictts", "imasc", "openslr63", "smc"]
    df = metrics.score(pd.DataFrame([{"ref": "a b", "hyp": "a b" if s == "smc" else "x y", "source": s,
                                      "dur": 1.0} for s in srcs]))
    d = tmp_path / "balanced231" / "koyal"
    d.mkdir(parents=True)
    df.to_csv(d / "per_utterance.csv", index=False)
    s = {"model": "koyal", "registry_entry": "koyal", "model_id": "adalat-ai/koyal-ml-120m-1.0",
         "overall": {"WER": metrics.corpus_wer(df), "n_clips": len(df)}}
    assert unseen_wer(s, str(tmp_path), "balanced231") == (0.0, 1)


# --- adapter with a fake NeMo -------------------------------------------------------------------

class _Hyp:
    def __init__(self, text):
        self.text = text


def _install_fake_nemo(monkeypatch, direct_text="കേരളം, 15.", transcribe_text=None, tuple_out=False):
    torch = pytest.importorskip("torch")
    calls = {}

    class FakeModel:
        def __init__(self):
            self.preprocessor = types.SimpleNamespace(featurizer=types.SimpleNamespace(dither=1e-5, pad_to=16))
            self.cfg = types.SimpleNamespace(decoding=types.SimpleNamespace(strategy="greedy_batch"))
            outer = self

            class Decoding:
                def rnnt_decoder_predictions_tensor(self, encoder_output, encoded_lengths):
                    calls["enc_shape"] = tuple(encoder_output.shape)
                    hyps = [_Hyp(direct_text)]
                    return (hyps, None) if tuple_out else hyps
            self.decoding = Decoding()

        def eval(self):
            calls["eval"] = True
            return self

        def forward(self, input_signal, input_signal_length):
            calls["signal"] = (tuple(input_signal.shape), input_signal.dtype, int(input_signal_length[0]))
            T = input_signal.shape[1] // 640 + 1
            return torch.zeros(1, 512, T), torch.tensor([T])

        def transcribe(self, audio, batch_size=1, verbose=False):
            calls["transcribe_calls"] = calls.get("transcribe_calls", 0) + 1
            assert isinstance(audio[0], np.ndarray)
            return [_Hyp(direct_text if transcribe_text is None else transcribe_text)]

    def from_pretrained(model_id, map_location=None):
        calls["model_id"], calls["map_location"] = model_id, map_location
        return FakeModel()

    asr = types.ModuleType("nemo.collections.asr")
    asr.models = types.SimpleNamespace(ASRModel=types.SimpleNamespace(from_pretrained=from_pretrained))
    collections = types.ModuleType("nemo.collections")
    collections.asr = asr
    nemo = types.ModuleType("nemo")
    nemo.collections = collections
    for name, mod in (("nemo", nemo), ("nemo.collections", collections), ("nemo.collections.asr", asr)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


WAV = (0.01 * np.random.default_rng(0).standard_normal(16000 * 2)).astype(np.float32)


def test_nemo_adapter_direct_decode(monkeypatch):
    calls = _install_fake_nemo(monkeypatch)
    a = models.build("koyal", threads=1)
    a.load()
    assert calls["model_id"] == "adalat-ai/koyal-ml-120m-1.0" and calls["eval"]
    assert a.model.preprocessor.featurizer.dither == 0.0 and a.model.preprocessor.featurizer.pad_to == 0
    t = a.transcribe(WAV)
    assert t.text == "കേരളം, 15." and t.n_tokens is None
    assert calls["signal"][0] == (1, len(WAV)) and calls["signal"][2] == len(WAV)
    a.transcribe(WAV)
    assert calls["transcribe_calls"] == 1           # cross-check runs on the first clip only
    info = a.info()
    assert info["matches_model_transcribe"] is True and info["decoding"].startswith("greedy (greedy_batch)")
    json.dumps(info)


def test_nemo_adapter_handles_tuple_output_and_flags_mismatch(monkeypatch):
    _install_fake_nemo(monkeypatch, direct_text="എ", transcribe_text="ബി", tuple_out=True)
    a = models.build("koyal", threads=1)
    a.load()
    assert a.transcribe(WAV).text == "എ"
    assert a.info()["matches_model_transcribe"] is False
    assert a.info()["mismatch_example"] == {"direct": "എ", "model.transcribe": "ബി"}


# --- no-digit variant ---------------------------------------------------------------------------

@pytest.mark.parametrize("s", ["15 പേർ", "൧൫", "പതിനഞ്ച് പേർ", "രണ്ടായിരത്തി ഇരുപത്", "ആറ്", "ലക്ഷം രൂപ",
                               "മൂന്നാം ദിവസം", "നൂറ്റാണ്ട്", "കോടിയാണ്"])
def test_has_number_true(s):
    assert has_number(s)


@pytest.mark.parametrize("s", ["അവൻ ആയിരുന്നു", "മൂന്നാർ", "ലക്ഷ്യം", "പത്തനാപുരത്ത്", "ഒരു കുട്ടി",
                               "ഒന്നും ഇല്ല", "ഇരുപുറങ്ങളിലും", "ആറ്റിൽ", "കോടതി", "കേരളം, ഇന്ത്യ."])
def test_has_number_false(s):
    assert not has_number(s)


def test_nodigit_excludes_ref_or_hyp_numbers():
    df = metrics.score(pd.DataFrame([
        {"ref": "പതിനഞ്ച് പേർ", "hyp": "15 പേർ", "source": "x", "dur": 1.0},    # number in both
        {"ref": "അവൻ വന്നു", "hyp": "അവൻ 2", "source": "x", "dur": 1.0},          # number in hyp only
        {"ref": "കേരളം ഇന്ത്യ", "hyp": "കേരളം, ഇന്ത്യ.", "source": "x", "dur": 1.0},  # punctuation only
        {"ref": "a b c", "hyp": "a x c", "source": "x", "dur": 1.0},
    ]))
    assert df.has_number.tolist() == [True, True, False, False]
    nd = metrics.nodigit(df)
    assert nd["nodigit_excluded"] == 2 and nd["nodigit_n_clips"] == 2
    assert nd["WER_nodigit"] == pytest.approx(1 / 5)
    s = metrics.summarise(df)
    assert s["WER_nodigit"] == nd["WER_nodigit"] and s["nodigit_excluded"] == 2
    assert s["WER"] > s["WER_nodigit"]


def test_nodigit_recomputed_for_old_results_without_column():
    df = metrics.score(pd.DataFrame([{"ref": "രണ്ട്", "hyp": "2", "source": "x", "dur": 1.0},
                                     {"ref": "a", "hyp": "a", "source": "x", "dur": 1.0}]))
    nd = metrics.nodigit(df.drop(columns=["has_number"]))
    assert nd["nodigit_excluded"] == 1 and nd["WER_nodigit"] == 0.0
