"""The CLI is what the Kaggle notebook calls: one subprocess per model."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
REPO = Path(__file__).resolve().parent.parent


def _cli(*args, cwd):
    env = {**os.environ, "PYTHONPATH": str(REPO), "HF_HUB_OFFLINE": "1"}
    p = subprocess.run([sys.executable, "-m", "mlasr_eval", *args], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout


@pytest.mark.parametrize("entry", ["whisper-v2", "fw-v2-int8"])
def test_cli_run_with_checkpoint_override(entry, tiny_whisper_dir, synthetic_eval, tmp_path):
    out = _cli("run", entry, "--model-id", tiny_whisper_dir, "--data", synthetic_eval, "--results",
               str(tmp_path / "res"), "--cores", "1", "--limit", "4", "--warmup", "1",
               "--max-new-tokens", "16", "--tag", f"tiny-{entry}", cwd=tmp_path)
    assert "WER" in out
    s = json.load(open(tmp_path / "res" / "balanced231" / f"tiny-{entry}" / "summary.json"))
    assert s["model_id"] == tiny_whisper_dir and "sanity" not in s and s["overlap_note"] is None
    assert s["hardware"]["threads"] == 1 and s["hardware"]["pinned_physical_cores"] == 1
    assert s["overall"]["n_clips"] == 4


def test_cli_list_and_report(tiny_whisper_dir, synthetic_eval, tmp_path):
    assert "indicconformer-ctc" in _cli("list", cwd=tmp_path)
    _cli("run", "whisper-v2", "--model-id", tiny_whisper_dir, "--data", synthetic_eval, "--results",
         str(tmp_path / "res"), "--cores", "1", "--limit", "3", "--warmup", "0", "--max-new-tokens", "8",
         cwd=tmp_path)
    md = _cli("report", "--results", str(tmp_path / "res"), cwd=tmp_path)
    assert "WER / CER % by source" in md
