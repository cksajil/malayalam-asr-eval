"""Model registry. Add a model by adding an entry here; add an engine by adding an adapter.

`overlap` records what is known about each model's training data versus the test sources.
own_corpus_test_clean.csv is only filtered against sajilck/malayalam-asr-corpus's own train
split, so a third-party model trained on the same public corpora may have seen test
sentences. Treat per-source numbers flagged here with care.
"""
from __future__ import annotations

MODELS: dict[str, dict] = {
    # --- own models ----------------------------------------------------------------------
    "whisper-v2": dict(
        adapter="hf_whisper", model_id="sajilck/whisper-small-malayalam-v2",
        role="own (sanity check)", params="244M",
        expect={"WER": 0.312, "CER": 0.040},   # published, 231 clips, transformers 4.46.3
        overlap="Test set filtered against this model's training corpus."),
    "whisper-v1": dict(
        adapter="hf_whisper", model_id="sajilck/whisper-small-malayalam",
        role="own", params="244M", expect={"WER": 0.435, "CER": 0.073},
        overlap="Test set filtered against this model's training corpus."),
    "fw-v2-int8": dict(
        adapter="faster_whisper", model_id="sajilck/whisper-small-malayalam-v2",
        kwargs=dict(quantization="int8", compute_type="int8"),
        role="own, CPU engine", params="244M",
        overlap="Same weights as whisper-v2."),

    # --- competitors -----------------------------------------------------------------------
    "vrclc-medium": dict(
        adapter="hf_whisper", model_id="vrclc/Whisper-medium-Malayalam",
        role="competitor", params="769M",
        overlap="Training data not checked yet; read the model card before quoting per-source numbers."),
    "kavya-small": dict(
        adapter="hf_whisper", model_id="kavyamanohar/whisper-small-malayalam",
        role="competitor", params="244M",
        overlap="Card lists Common Voice 13; commonvoice test sentences may overlap its training data."),

    # --- non-autoregressive (CTC) ---------------------------------------------------------------
    "indicconformer-ctc": dict(
        adapter="indic_conformer", model_id="ai4bharat/indic-conformer-600m-multilingual",
        kwargs=dict(decoding="ctc", language="ml"),
        role="CTC baseline", params="600M", gated=True,
        overlap="Training data not stated on the card; AI4Bharat also published Shrutilipi, so the "
                "shrutilipi rows may overlap."),
    "indicconformer-rnnt": dict(
        adapter="indic_conformer", model_id="ai4bharat/indic-conformer-600m-multilingual",
        kwargs=dict(decoding="rnnt", language="ml"),
        role="optional (same model, RNNT decoding)", params="600M", gated=True, optional=True,
        overlap="As indicconformer-ctc."),
    "mms-1b-all": dict(
        adapter="wav2vec2_ctc", model_id="facebook/mms-1b-all", kwargs=dict(target_lang="mal"),
        role="wav2vec2 CTC baseline", params="965M",
        overlap="MMS fine-tuning data includes Common Voice and FLEURS; commonvoice rows may overlap."),
    "xlsr-ml-shawn": dict(
        adapter="wav2vec2_ctc", model_id="shawn-nyk/wav2vec-large-xlsr-malayalam-with-lm",
        role="optional wav2vec2 (XLS-R 300M)", params="315M", optional=True,
        overlap="Card says it was trained on Indic TTS, OpenSLR, SMC and IIIT-H Malayalam; indictts, "
                "openslr63 and smc rows likely overlap. Report its overall number with that caveat."),
}

ADAPTERS = {
    "hf_whisper": ("mlasr_eval.adapters.hf_whisper", "HFWhisper"),
    "faster_whisper": ("mlasr_eval.adapters.faster_whisper_ct2", "FasterWhisper"),
    "indic_conformer": ("mlasr_eval.adapters.indic_conformer", "IndicConformer"),
    "wav2vec2_ctc": ("mlasr_eval.adapters.wav2vec2_ctc", "Wav2Vec2CTC"),
}

DEFAULT_PLAN = ["whisper-v2", "fw-v2-int8", "indicconformer-ctc", "mms-1b-all", "vrclc-medium", "kavya-small"]


def build(name: str, threads: int, model_id: str | None = None, **overrides):
    """Instantiate the adapter for a registry entry (lazy import, so unused engines needn't be installed).

    `model_id` swaps the checkpoint while keeping the entry's engine and settings.
    """
    import importlib
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    spec = MODELS[name]
    module, cls = ADAPTERS[spec["adapter"]]
    kwargs = {**spec.get("kwargs", {}), **overrides}
    return getattr(importlib.import_module(module), cls)(model_id or spec["model_id"], threads=threads, **kwargs)
