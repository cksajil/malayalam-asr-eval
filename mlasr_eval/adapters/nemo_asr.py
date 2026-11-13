"""NVIDIA NeMo ASR models (Conformer/FastConformer, CTC / RNNT / TDT / hybrid), FP32 on CPU.

Used for Adalat AI's Koyal Malayalam model (adalat-ai/koyal-ml-120m-1.0, a Conformer hybrid
TDT-CTC). Loading follows the model card: `ASRModel.from_pretrained(model_id)`, decoding with
the checkpoint's own (greedy TDT) decoding config.

Each clip is fed to the model directly as a 16 kHz float32 tensor (preprocessor -> encoder ->
decoder), so the timer covers only the model, not the per-call dataloader that
`model.transcribe()` builds. On the first clip the adapter also runs `model.transcribe()` and
records whether both paths give the same text (`info()["matches_model_transcribe"]`).

NeMo needs a newer transformers than the one pinned for the published Whisper numbers, so on
Kaggle it is installed in a separate virtualenv (see the notebook); install with `pip install
"nemo_toolkit[asr]"`.

Koyal writes rich orthography (punctuation, numerals as digits). norm_ml strips punctuation;
see the no-digit WER variant in mlasr_eval.metrics for the digits.
"""
from __future__ import annotations

import numpy as np

from .base import Adapter, Transcript, versions
from ..cpu import torch_threads


def _hyp_text(h) -> str:
    """Text of one NeMo hypothesis, whatever shape this NeMo version returns."""
    if isinstance(h, (list, tuple)):
        h = h[0] if h else ""
    text = getattr(h, "text", h)
    return "" if text is None else str(text)


def _first(out):
    """NeMo returns a list of hypotheses, or (best, all) for some versions/model types."""
    if isinstance(out, tuple):
        out = out[0]
    return out[0] if isinstance(out, (list, tuple)) and out else out


class NemoASR(Adapter):
    family = "nemo"

    def __init__(self, model_id: str, threads: int = 2, revision: str | None = None,
                 decoder_type: str | None = None, check_transcribe: bool = True, **kwargs):
        super().__init__(model_id, threads, revision, **kwargs)
        self.decoder_type = decoder_type          # None = the checkpoint's default ("rnnt" for TDT hybrids)
        self.check_transcribe = check_transcribe
        self.token_cap = None                     # transducers/CTC have no global output-length cap
        self.matches_model_transcribe = None
        self.mismatch_example = None
        self.decoding_strategy = None

    def load(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr
        torch_threads(self.threads)
        self.torch = torch
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id, map_location="cpu")
        self.model.eval()
        if self.decoder_type and hasattr(self.model, "change_decoding_strategy"):
            self.model.change_decoding_strategy(decoder_type=self.decoder_type)
        # model.transcribe() turns these off for inference; do the same for the direct path.
        feat = getattr(getattr(self.model, "preprocessor", None), "featurizer", None)
        if feat is not None:
            if hasattr(feat, "dither"):
                feat.dither = 0.0
            if hasattr(feat, "pad_to"):
                feat.pad_to = 0
        try:
            self.decoding_strategy = str(self.model.cfg.decoding.strategy)
        except Exception:
            self.decoding_strategy = None

    def _decode_direct(self, wav: np.ndarray) -> str:
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        xl = torch.tensor([x.shape[1]], dtype=torch.long)
        with torch.inference_mode():
            enc, enc_len = self.model.forward(input_signal=x, input_signal_length=xl)[:2]
            if (self.decoder_type or "rnnt") == "ctc" or not hasattr(self.model, "decoding"):
                log_probs = self.model.ctc_decoder(encoder_output=enc)
                out = self.model.ctc_decoding.ctc_decoder_predictions_tensor(log_probs, decoder_lengths=enc_len)
            else:
                out = self.model.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc,
                                                                          encoded_lengths=enc_len)
        return _hyp_text(_first(out)).strip()

    def transcribe(self, wav: np.ndarray) -> Transcript:
        text = self._decode_direct(wav)
        if self.check_transcribe and self.matches_model_transcribe is None:
            ref = _hyp_text(_first(self.model.transcribe([np.asarray(wav, dtype=np.float32)],
                                                         batch_size=1, verbose=False))).strip()
            self.matches_model_transcribe = ref == text
            if ref != text:
                self.mismatch_example = {"direct": text, "model.transcribe": ref}
        return Transcript(text=text)

    def info(self) -> dict:
        d = super().info()
        d.update({"engine": "NeMo FP32", "decoding": f"greedy ({self.decoding_strategy or 'checkpoint default'}), no LM",
                  "decoder_type": self.decoder_type or "checkpoint default",
                  "matches_model_transcribe": self.matches_model_transcribe,
                  "mismatch_example": self.mismatch_example,
                  "versions": versions("torch", "nemo_toolkit", "transformers")})
        return d
