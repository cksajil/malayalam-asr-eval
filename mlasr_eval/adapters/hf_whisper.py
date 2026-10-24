"""Whisper via Hugging Face Transformers, FP32, greedy — the reference setup of the published numbers.

max_new_tokens defaults to 444: Malayalam needs ~27 Whisper tokens per second of audio, and
4 prompt tokens + 444 = 448, Whisper's decoder limit. The old 225 cap silently truncated
clips longer than ~9 s (Part 3 of the blog series).
"""
from __future__ import annotations

import numpy as np

from .base import Adapter, Transcript, versions
from ..cpu import torch_threads

SR = 16000
_SIZE_BY_DMODEL = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large"}


def guess_base_model(config) -> str:
    size = _SIZE_BY_DMODEL.get(getattr(config, "d_model", 0), "small")
    if size == "large":
        return "openai/whisper-large-v3" if getattr(config, "num_mel_bins", 80) == 128 else "openai/whisper-large-v2"
    return f"openai/whisper-{size}"


class HFWhisper(Adapter):
    family = "whisper-transformers"

    def __init__(self, model_id: str, threads: int = 2, revision: str | None = None,
                 max_new_tokens: int = 444, num_beams: int = 1, language: str = "ml",
                 base_generation_config: str | None = None, **kwargs):
        super().__init__(model_id, threads, revision, **kwargs)
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.language = language
        self.base_generation_config = base_generation_config
        self.token_cap = max_new_tokens
        self.patched_generation_config = None
        self.long_form_clips = 0

    def load(self) -> None:
        import torch
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor
        torch_threads(self.threads)
        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=torch.float32).eval()

        gc = self.model.generation_config
        if not getattr(gc, "lang_to_id", None):
            # Older fine-tunes ship a generation config without language tables, which makes
            # generate(language="ml") fail. Borrow the tables from the matching OpenAI checkpoint.
            base = self.base_generation_config or guess_base_model(self.model.config)
            ref = GenerationConfig.from_pretrained(base)
            for attr in ("lang_to_id", "task_to_id", "no_timestamps_token_id", "is_multilingual",
                         "decoder_start_token_id"):
                if getattr(ref, attr, None) is not None:
                    setattr(gc, attr, getattr(ref, attr))
            self.patched_generation_config = base
        # Language and task are passed explicitly; stale forced ids would conflict with them.
        gc.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None
        self.special_ids = set(self.processor.tokenizer.all_special_ids)

    def transcribe(self, wav: np.ndarray) -> Transcript:
        torch = self.torch
        long_form = len(wav) > 30 * SR
        if long_form:
            # >30 s needs sequential long-form decoding; plain feature extraction would drop the tail.
            self.long_form_clips += 1
            feats = self.processor(wav, sampling_rate=SR, return_tensors="pt", truncation=False,
                                   padding="longest", return_attention_mask=True)
            extra = dict(attention_mask=feats.attention_mask, return_timestamps=True)
        else:
            feats = self.processor(wav, sampling_rate=SR, return_tensors="pt")
            extra = {}
        with torch.inference_mode():
            out = self.model.generate(feats.input_features, language=self.language, task="transcribe",
                                      max_new_tokens=self.max_new_tokens, num_beams=self.num_beams,
                                      do_sample=False, **extra)
        seq = out.sequences if hasattr(out, "sequences") else out
        ids = seq[0].tolist()
        n_text = sum(1 for i in ids if i not in self.special_ids)
        text = self.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return Transcript(text=text, n_tokens=n_text, extra={"long_form": long_form})

    def info(self) -> dict:
        d = super().info()
        d.update({"engine": "transformers FP32", "decoding": f"greedy (num_beams={self.num_beams})",
                  "max_new_tokens": self.max_new_tokens, "language": self.language,
                  "generation_config_borrowed_from": self.patched_generation_config,
                  "long_form_clips": self.long_form_clips,
                  "versions": versions("torch", "transformers", "tokenizers")})
        return d
