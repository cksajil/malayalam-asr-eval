"""Shared fixtures. Tiny random-weight models let the adapter code run offline in CI;
they say nothing about accuracy, only that loading, decoding and bookkeeping work."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

ML_WORDS = ["സംസ്ഥാനത്തെ", "വിദ്യാർത്ഥികളുടെയും", "നേടാനും", "മാറാനും", "ആഗോള", "കേരളം"]
SOURCES = ["shrutilipi", "commonvoice", "indictts", "imasc", "openslr63", "smc"]

WHISPER_SPECIALS = ["<|endoftext|>", "<|startoftranscript|>", "<|en|>", "<|ml|>", "<|translate|>",
                    "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nocaptions|>", "<|notimestamps|>"]


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


@pytest.fixture(scope="session")
def tiny_whisper_dir(tmp_path_factory):
    """A 2-layer random Whisper with a byte-level tokenizer and Whisper special tokens."""
    pytest.importorskip("torch")
    from transformers import (GenerationConfig, WhisperConfig, WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer)
    d = tmp_path_factory.mktemp("tiny_whisper")
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode())}
    for t in WHISPER_SPECIALS:
        vocab[t] = len(vocab)
    (d / "vocab.json").write_text(json.dumps(vocab))
    (d / "merges.txt").write_text("#version: 0.2\n")
    tok = WhisperTokenizer(str(d / "vocab.json"), str(d / "merges.txt"), unk_token="<|endoftext|>",
                           bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": WHISPER_SPECIALS[1:]})
    ids = {t: tok.convert_tokens_to_ids(t) for t in WHISPER_SPECIALS}
    cfg = WhisperConfig(vocab_size=len(vocab), d_model=64, encoder_layers=2, decoder_layers=2,
                        encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=128,
                        decoder_ffn_dim=128, max_source_positions=1500, max_target_positions=448,
                        decoder_start_token_id=ids["<|startoftranscript|>"], eos_token_id=ids["<|endoftext|>"],
                        pad_token_id=ids["<|endoftext|>"], bos_token_id=ids["<|endoftext|>"],
                        suppress_tokens=[], begin_suppress_tokens=[])
    import torch
    torch.manual_seed(0)
    model = WhisperForConditionalGeneration(cfg).eval()
    gen = GenerationConfig(decoder_start_token_id=ids["<|startoftranscript|>"], eos_token_id=ids["<|endoftext|>"],
                           pad_token_id=ids["<|endoftext|>"], bos_token_id=ids["<|endoftext|>"],
                           max_length=448, is_multilingual=True,
                           lang_to_id={"<|en|>": ids["<|en|>"], "<|ml|>": ids["<|ml|>"]},
                           task_to_id={"translate": ids["<|translate|>"], "transcribe": ids["<|transcribe|>"]},
                           no_timestamps_token_id=ids["<|notimestamps|>"],
                           prev_sot_token_id=ids["<|startofprev|>"], suppress_tokens=[],
                           begin_suppress_tokens=[])
    model.generation_config = gen
    model.save_pretrained(d)
    WhisperProcessor(WhisperFeatureExtractor(feature_size=80), tok).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="session")
def tiny_whisper_no_langs(tiny_whisper_dir, tmp_path_factory):
    """Same model, but its generation config lacks the language tables (like older fine-tunes)."""
    import shutil
    d = tmp_path_factory.mktemp("tiny_whisper_nolang")
    shutil.copytree(tiny_whisper_dir, d, dirs_exist_ok=True)
    gc = json.loads((d / "generation_config.json").read_text())
    for k in ("lang_to_id", "task_to_id", "no_timestamps_token_id"):
        gc.pop(k, None)
    (d / "generation_config.json").write_text(json.dumps(gc))
    return str(d)


@pytest.fixture(scope="session")
def tiny_w2v2_dir(tmp_path_factory):
    pytest.importorskip("torch")
    from transformers import (Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                              Wav2Vec2ForCTC, Wav2Vec2Processor)
    d = tmp_path_factory.mktemp("tiny_w2v2")
    chars = sorted(set("".join(ML_WORDS)))
    vocab = {"<pad>": 0, "<unk>": 1, "|": 2, **{c: i + 3 for i, c in enumerate(chars)}}
    (d / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    tok = Wav2Vec2CTCTokenizer(str(d / "vocab.json"), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                  return_attention_mask=False)
    Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok).save_pretrained(d)
    cfg = Wav2Vec2Config(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 4),
                         num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, pad_token_id=0)
    import torch
    torch.manual_seed(0)
    Wav2Vec2ForCTC(cfg).save_pretrained(d)
    return str(d)


@pytest.fixture()
def synthetic_eval(tmp_path):
    """A prepared-data pickle shaped like eval_data/balanced231.pkl (12 clips of noise)."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        dur = float(rng.uniform(1.0, 3.0))
        rows.append(dict(hf_row=i, source=SOURCES[i % len(SOURCES)],
                         sentence=" ".join(rng.choice(ML_WORDS, 4)), dur=dur,
                         wav=(0.01 * rng.standard_normal(int(16000 * dur))).astype(np.float32)))
    p = tmp_path / "balanced231.pkl"
    pd.DataFrame(rows).to_pickle(p)
    return str(p)
