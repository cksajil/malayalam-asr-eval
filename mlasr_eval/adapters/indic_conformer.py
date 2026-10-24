"""AI4Bharat IndicConformer (600M, multilingual) through its Hugging Face remote code.

The model is gated: accept the conditions on its Hugging Face page and set HF_TOKEN.
Decoding is CTC by default (one non-autoregressive pass, so cost scales with audio length,
not token count); "rnnt" is available for comparison.

The remote code runs ONNX Runtime sessions whose thread pools we can't configure from
outside, so the 2-core limit is enforced by CPU affinity (see mlasr_eval.cpu).
"""
from __future__ import annotations

import numpy as np

from .base import Adapter, Transcript, versions
from ..cpu import torch_threads


class IndicConformer(Adapter):
    family = "conformer"

    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual", threads: int = 2,
                 revision: str | None = None, decoding: str = "ctc", language: str = "ml", **kwargs):
        super().__init__(model_id, threads, revision, **kwargs)
        if decoding not in ("ctc", "rnnt"):
            raise ValueError("decoding must be 'ctc' or 'rnnt'")
        self.decoding = decoding
        self.language = language
        self.token_cap = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel
        torch_threads(self.threads)
        self.torch = torch
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True, revision=self.revision)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(self, wav: np.ndarray) -> Transcript:
        x = self.torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)  # [1, T], 16 kHz
        with self.torch.inference_mode():
            out = self.model(x, self.language, self.decoding)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return Transcript(text=str(out).strip())

    def info(self) -> dict:
        d = super().info()
        d.update({"engine": "HF remote code (ONNX Runtime)", "decoding": self.decoding, "language": self.language,
                  "versions": versions("torch", "transformers", "onnxruntime")})
        return d
