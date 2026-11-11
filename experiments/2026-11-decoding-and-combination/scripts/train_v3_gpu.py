"""UNTESTED DRAFT, never run: v3 fine-tuning recipe (needs a GPU; e.g. Kaggle 2xT4 / 1xA100). Not runnable on the CPU box used for the
inference-side experiments.

Changes vs v2 (each is one of the training-side items in the report):
  1. Base model: openai/whisper-large-v3-turbo (809M, 4-layer decoder: per-token decode cost ~ whisper-small,
     encoder much stronger). Fallback: whisper-medium. Set BASE below.
  2. More, more varied data: sajilck/malayalam-asr-corpus + IndicVoices-ml + Kathbath-ml + FLEURS-ml (train splits
     only; re-run the leakage filter against own_corpus_test_clean.csv before training).
  3. Consistent target normalisation (NFC, atomic chillu, no ZWJ/ZWNJ) so the model is not trained on two
     spellings of the same word.
  4. Augmentation: speed perturb 0.9/1.0/1.1, additive noise at 5-20 dB SNR, SpecAugment (mask_time_prob).
  5. Checkpoint averaging of the last N evals (see average_checkpoints()).
Eval: compute WER/CER on the 90-clip dev set every 1000 steps; report test231 once, at the end.
"""
import os, sys, random, glob
import numpy as np, torch
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments)
sys.path.insert(0, os.path.dirname(__file__))
from common import norm_ml

BASE = os.environ.get("BASE", "openai/whisper-large-v3-turbo")
OUT = os.environ.get("OUT", "whisper-ml-v3")
proc = WhisperProcessor.from_pretrained(BASE, language="malayalam", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.config.apply_spec_augment, model.config.mask_time_prob, model.config.mask_feature_prob = True, 0.05, 0.05
model.generation_config.language, model.generation_config.task = "malayalam", "transcribe"
model.generation_config.forced_decoder_ids = None


def own():
    ds = load_dataset("sajilck/malayalam-asr-corpus", split="train")
    return ds.select_columns(["audio", "sentence"])


def extra():
    """Optional extra corpora. Check each licence before commercial use; IndicVoices is gated (accept terms)."""
    out = []
    for name, cfg, col in [("google/fleurs", "ml_in", "transcription"),
                           ("ai4bharat/kathbath", "malayalam", "text"),
                           ("ai4bharat/IndicVoices", "malayalam", "text")]:
        try:
            d = load_dataset(name, cfg, split="train").rename_column(col, "sentence").select_columns(["audio", "sentence"])
            out.append(d)
            print("added", name, len(d))
        except Exception as e:
            print("skip", name, e)
    return out


ds = concatenate_datasets([own()] + extra()).cast_column("audio", Audio(sampling_rate=16000))
ds = ds.map(lambda b: {"sentence": norm_ml(b["sentence"])})
ds = ds.filter(lambda s: 0 < len(proc.tokenizer(s).input_ids) < 440, input_columns="sentence")


def augment(wav):
    r = random.random()
    if r < 0.33:  # speed perturbation
        import librosa
        wav = librosa.effects.time_stretch(wav, rate=random.choice([0.9, 1.1]))
    if random.random() < 0.3:  # white/pink-ish noise at 5-20 dB SNR (swap in MUSAN for real noise)
        snr = random.uniform(5, 20)
        p = np.mean(wav ** 2) + 1e-9
        wav = wav + np.random.randn(len(wav)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
    return wav.astype(np.float32)


@dataclass
class Collator:
    def __call__(self, batch):
        wavs = [augment(np.asarray(b["audio"]["array"], dtype=np.float32)) for b in batch]
        feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features
        lab = proc.tokenizer([b["sentence"] for b in batch], padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == model.config.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": feats, "labels": labels}


args = Seq2SeqTrainingArguments(
    output_dir=OUT, per_device_train_batch_size=16, gradient_accumulation_steps=2,
    learning_rate=float(os.environ.get("LR", "5e-5")),  # large models want lower LR than small's 1e-4
    warmup_steps=500, max_steps=int(os.environ.get("STEPS", "12000")), lr_scheduler_type="linear",
    fp16=torch.cuda.is_available(), gradient_checkpointing=True, save_steps=1000, save_total_limit=5,
    logging_steps=50, remove_unused_columns=False, dataloader_num_workers=4, report_to=[])
trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=Collator())
trainer.train()


def average_checkpoints(paths, out):
    """Uniform average of the last few checkpoints' weights: a free ~0.5-1 pt WER gain in many ASR setups."""
    from safetensors.torch import load_file, save_file
    sd = [load_file(f"{p}/model.safetensors") for p in paths]
    avg = {k: sum(s[k].float() for s in sd) / len(sd) for k in sd[0]}
    model.load_state_dict({k: v.to(sd[0][k].dtype) for k, v in avg.items()})
    model.save_pretrained(out)
    proc.save_pretrained(out)


ck = sorted(glob.glob(f"{OUT}/checkpoint-*"), key=lambda p: int(p.rsplit("-", 1)[1]))[-3:]
average_checkpoints(ck, f"{OUT}/averaged")
