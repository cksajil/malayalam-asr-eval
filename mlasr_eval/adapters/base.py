"""Common adapter interface: every model is loaded once, then called on one 16 kHz clip at a time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transcript:
    text: str
    n_tokens: int | None = None   # generated text tokens (autoregressive models), for the token-cap check
    extra: dict = field(default_factory=dict)


class Adapter:
    """Subclasses implement `load()` and `transcribe()`.

    `token_cap` is the most text tokens one call can emit (None for CTC models, which have
    no decoding loop to cut off). A clip whose output reaches the cap may be truncated.
    """

    family = "generic"
    token_cap: int | None = None

    def __init__(self, model_id: str, threads: int = 2, revision: str | None = None, **kwargs):
        self.model_id = model_id
        self.threads = threads
        self.revision = revision
        self.kwargs = kwargs

    def load(self) -> None:
        raise NotImplementedError

    def transcribe(self, wav: np.ndarray) -> Transcript:
        raise NotImplementedError

    def info(self) -> dict:
        """Settings and library versions recorded in the results file."""
        return {"family": self.family, "model_id": self.model_id, "revision": self.revision,
                "threads": self.threads, "token_cap": self.token_cap}


def versions(*packages: str) -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out
