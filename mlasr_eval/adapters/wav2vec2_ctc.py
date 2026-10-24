"""wav2vec2-style CTC models (MMS, XLS-R fine-tunes), FP32, greedy argmax decoding, no LM.

For MMS (facebook/mms-1b-all) pass target_lang="mal" to load the Malayalam adapter and vocabulary.
"""
from __future__ import annotations

import numpy as np

from .base import Adapter, Transcript, versions
from ..cpu import torch_threads

SR = 16000


class Wav2Vec2CTC(Adapter):
    family = "wav2vec2-ctc"

    def __init__(self, model_id: str, threads: int = 2, revision: str | None = None,
                 target_lang: str | None = None, **kwargs):
        super().__init__(model_id, threads, revision, **kwargs)
        self.target_lang = target_lang
        self.token_cap = None

    def load(self) -> None:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        torch_threads(self.threads)
        self.torch = torch
        if self.target_lang:
            # MMS: per-language adapter weights + vocabulary
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, revision=self.revision,
                                                               target_lang=self.target_lang)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, revision=self.revision,
                                                        target_lang=self.target_lang,
                                                        ignore_mismatched_sizes=True)
        else:
            # Wav2Vec2Processor (not AutoProcessor) so repos that ship an LM decoder still
            # load without pyctcdecode/kenlm; we score greedy CTC for every model.
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, revision=self.revision)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, revision=self.revision)
        self.model.eval()

    def transcribe(self, wav: np.ndarray) -> Transcript:
        inputs = self.processor(wav, sampling_rate=SR, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(inputs.input_values).logits
        ids = self.torch.argmax(logits, dim=-1)[0].tolist()
        text = self.processor.tokenizer.decode(ids)   # collapses repeats and drops blanks
        return Transcript(text=text.strip())

    def info(self) -> dict:
        d = super().info()
        d.update({"engine": "transformers FP32", "decoding": "greedy CTC, no LM", "target_lang": self.target_lang,
                  "versions": versions("torch", "transformers")})
        return d
