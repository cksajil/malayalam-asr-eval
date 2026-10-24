"""Whisper via CTranslate2 / faster-whisper, int8 on CPU.

Converts the Hugging Face checkpoint once (cached), then decodes greedily with the same
444-token cap as the Transformers reference. On CPU use int8 — int8_float16 only helps on GPU.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from .base import Adapter, Transcript, versions

CACHE = Path(os.environ.get("MLASR_CACHE", Path.home() / ".cache" / "mlasr_eval"))


def _converter_class():
    """TransformersConverter that also works with transformers < 4.56.

    Recent ctranslate2 calls from_pretrained(..., dtype=...), but the pinned transformers
    (4.46.3, used for the published numbers) only knows `torch_dtype`.
    """
    from ctranslate2.converters import TransformersConverter

    class _Compat(TransformersConverter):
        def load_model(self, model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs:
                try:
                    return model_class.from_pretrained(model_name_or_path, **kwargs)
                except TypeError:
                    kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_name_or_path, **kwargs)

    return _Compat


class FasterWhisper(Adapter):
    family = "whisper-ctranslate2"

    def __init__(self, model_id: str, threads: int = 2, revision: str | None = None,
                 quantization: str = "int8", compute_type: str = "int8", max_new_tokens: int = 444,
                 beam_size: int = 1, language: str = "ml", ct2_dir: str | None = None, **kwargs):
        super().__init__(model_id, threads, revision, **kwargs)
        self.quantization = quantization
        self.compute_type = compute_type
        self.max_new_tokens = max_new_tokens
        self.beam_size = beam_size
        self.language = language
        self.token_cap = max_new_tokens
        self.ct2_dir = Path(ct2_dir) if ct2_dir else CACHE / "ct2" / f"{model_id.replace('/', '__')}-{quantization}"
        self.convert_s = None

    def convert(self) -> None:
        """HF checkpoint -> CTranslate2 directory, with the tokenizer and preprocessor config faster-whisper reads."""
        from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
        t0 = time.time()
        self.ct2_dir.parent.mkdir(parents=True, exist_ok=True)
        conv = _converter_class()(self.model_id, revision=self.revision)
        conv.convert(str(self.ct2_dir), quantization=self.quantization, force=True)
        # faster-whisper reads tokenizer.json (fast-tokenizer format) and preprocessor_config.json;
        # without them it silently falls back to the stock OpenAI tokenizer and feature settings.
        WhisperTokenizerFast.from_pretrained(self.model_id, revision=self.revision).save_pretrained(str(self.ct2_dir))
        WhisperFeatureExtractor.from_pretrained(self.model_id, revision=self.revision).save_pretrained(str(self.ct2_dir))
        if not (self.ct2_dir / "tokenizer.json").exists():
            raise RuntimeError(f"tokenizer.json was not written to {self.ct2_dir}")
        self.convert_s = time.time() - t0

    def load(self) -> None:
        from faster_whisper import WhisperModel
        if not (self.ct2_dir / "model.bin").exists() or not (self.ct2_dir / "tokenizer.json").exists():
            self.convert()
        self.model = WhisperModel(str(self.ct2_dir), device="cpu", compute_type=self.compute_type,
                                  cpu_threads=self.threads, num_workers=1)
        self.eot = self.model.hf_tokenizer.token_to_id("<|endoftext|>")

    def transcribe(self, wav: np.ndarray) -> Transcript:
        segments, _ = self.model.transcribe(
            wav, language=self.language, task="transcribe", beam_size=self.beam_size, best_of=1,
            temperature=0.0,                    # single temperature: no fallback, like HF greedy
            condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
            word_timestamps=False, max_new_tokens=self.max_new_tokens)
        segs = list(segments)                   # generator: decoding happens here, inside the timer
        text = " ".join(s.text.strip() for s in segs).strip()
        n_text = sum(sum(1 for t in s.tokens if t < self.eot) for s in segs)
        return Transcript(text=text, n_tokens=n_text, extra={"segments": len(segs)})

    def info(self) -> dict:
        d = super().info()
        d.update({"engine": f"CTranslate2 {self.compute_type}", "quantization": self.quantization,
                  "decoding": f"greedy (beam_size={self.beam_size}), no timestamps, no temperature fallback",
                  "max_new_tokens": self.max_new_tokens, "language": self.language,
                  "convert_s": self.convert_s, "ct2_dir": str(self.ct2_dir),
                  "ct2_size_MB": round(sum(f.stat().st_size for f in self.ct2_dir.glob("*")) / 1e6, 1)
                  if self.ct2_dir.exists() else None,
                  "versions": versions("ctranslate2", "faster-whisper", "transformers")})
        return d
