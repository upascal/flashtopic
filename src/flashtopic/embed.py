"""Pluggable text encoders.

Default is **model2vec / potion** — static, CPU-fast, no torch — which is good enough
for clustering and keeps the base install light. The ``Encoder`` protocol lets any
other embedder slot in; two GPU-class encoders ship behind extras:

* ``QwenEncoder`` — instruction-tuned Qwen3 via MLX (Apple Silicon, ``[mlx]`` extra).
* ``SentenceTransformerEncoder`` — any sentence-transformers model on torch
  (``[st]`` extra); the device is capability-probed mps → cuda → cpu, so the same
  code path covers Apple MPS today and NVIDIA later.

Dispatch is by explicit ``backend`` (``model2vec`` / ``mlx`` / ``st``) with a
model-id sniff as the back-compat fallback. All encoders return ``(N, d)`` float32,
**L2-normalized** (so cosine = dot and the spherical k-means backends get unit
vectors).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

from flashtopic.config import DEFAULT_EMBED_MODEL

log = logging.getLogger("flashtopic.embed")

ENCODER_BACKENDS: tuple[str, ...] = ("model2vec", "mlx", "st")


def _l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


@runtime_checkable
class Encoder(Protocol):
    def encode(self, texts: list[str], *, batch_size: int = 1024,
               show_progress: bool = False) -> np.ndarray: ...


class Model2VecEncoder:
    """Static potion/model2vec embedder (default). CPU, fast, torch-free."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _get(self):
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, texts, *, batch_size: int = 1024, show_progress: bool = False) -> np.ndarray:
        vecs = self._get().encode(
            list(texts), batch_size=batch_size, show_progress_bar=show_progress
        )
        return _l2(np.asarray(vecs, dtype=np.float32))


class QwenEncoder:
    """Instruction-tuned Qwen3 embedder via MLX (Apple Silicon, ``[mlx]`` extra).

    Last-token pooled, L2-normalized; an optional ``instruction`` is prefixed as
    ``"Instruct: {instruction}\\nQuery: {text}"`` so passages land in the same
    instruction-conditioned space.

    Batched: sequences are length-sorted, chunked, and **right-padded** — safe
    because attention is causal, so padding after a position cannot influence it;
    each sequence's embedding is read at its own last real token.
    """

    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        instruction: str | None = None,
        pooling: str = "last_token",
    ):
        self.model_id = model_id
        self.instruction = instruction
        self.pooling = pooling
        self._state = None

    def _get(self):
        if self._state is None:
            from mlx_lm import load as mlx_load

            model, tokenizer = mlx_load(self.model_id)
            self._state = {"model": model, "tokenizer": tokenizer,
                           "eos_id": tokenizer.eos_token_id}
        return self._state

    def _tokenize(self, text: str, tokenizer, eos_id) -> list[int]:
        if self.instruction:
            text = f"Instruct: {self.instruction}\nQuery: {text}"
        tokens = tokenizer.encode(text)
        if self.pooling == "last_token" and eos_id is not None:
            tokens = tokens + [eos_id]
        return tokens

    def encode(self, texts, *, batch_size: int = 8, show_progress: bool = False) -> np.ndarray:
        import mlx.core as mx

        st = self._get()
        model, tokenizer, eos_id = st["model"], st["tokenizer"], st["eos_id"]
        texts = list(texts)
        seqs = [self._tokenize(t, tokenizer, eos_id) for t in texts]

        # Length-sort so each padded batch wastes as little compute as possible;
        # results go back to original positions at the end.
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        out: list[np.ndarray | None] = [None] * len(seqs)
        pad_id = eos_id if eos_id is not None else 0
        step = max(1, batch_size)
        for start in range(0, len(order), step):
            chunk = order[start:start + step]
            batch = [seqs[i] for i in chunk]
            lens = [len(s) for s in batch]
            maxlen = max(lens)
            padded = mx.array([s + [pad_id] * (maxlen - len(s)) for s in batch])
            hidden = model.model(padded)  # (B, T, d); pad positions are never read
            if self.pooling == "last_token":
                idx = mx.array(lens, dtype=mx.int32) - 1  # each row's last REAL token
                emb = mx.take_along_axis(
                    hidden, idx[:, None, None].astype(mx.int32), axis=1
                ).squeeze(1)
            else:
                mask = (mx.arange(maxlen)[None, :] < mx.array(lens)[:, None])
                emb = (hidden * mask[:, :, None].astype(hidden.dtype)).sum(axis=1)
                emb = emb / mx.array(lens, dtype=emb.dtype)[:, None]
            emb = emb.astype(mx.float32)
            mx.eval(emb)
            arr = np.array(emb)
            for row, i in enumerate(chunk):
                out[i] = arr[row]
        return _l2(np.stack(out))


# --------------------------------------------------------------------------- #
# torch device probes (cached) — same probe-don't-guess idiom as cluster/backends
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _probe_torch_mps() -> bool:
    try:
        import torch

        if not torch.backends.mps.is_available():
            return False
        (torch.ones(1, device="mps") + 1).cpu()  # confirm the device runs an op
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        log.debug("torch mps unavailable: %s", e)
        return False


@lru_cache(maxsize=1)
def _probe_torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception as e:  # pragma: no cover
        log.debug("torch cuda unavailable: %s", e)
        return False


def resolve_st_device(device: str = "auto") -> str:
    """Torch device for SentenceTransformerEncoder: probe mps → cuda → cpu.

    An explicit non-auto ``device`` is passed through untouched (torch errors
    clearly if it doesn't exist).
    """
    if device != "auto":
        return device
    if _probe_torch_mps():
        return "mps"
    if _probe_torch_cuda():
        return "cuda"
    return "cpu"


class SentenceTransformerEncoder:
    """Transformer embedder via sentence-transformers (``[st]`` extra, pulls torch).

    Covers Apple MPS today and NVIDIA CUDA later with one code path. An optional
    ``instruction`` uses the same ``"Instruct: {instruction}\\nQuery: {text}"``
    prefix as ``QwenEncoder`` — that literal format is the documented prompt for
    both Qwen3-Embedding and the e5-instruct family, so instruction semantics are
    uniform across backends.
    """

    def __init__(self, model_id: str, instruction: str | None = None,
                 device: str = "auto"):
        self.model_id = model_id
        self.instruction = instruction
        self.device = device
        self._model = None

    def _get(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id, device=resolve_st_device(self.device)
            )
        return self._model

    def encode(self, texts, *, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        if self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery: {t}" for t in texts]
        vecs = self._get().encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress,
            normalize_embeddings=True, convert_to_numpy=True,
        )
        return _l2(np.asarray(vecs, dtype=np.float32))


def sniff_backend(model: str) -> str:
    """Best-effort backend from a bare model id (the historical heuristic).

    Callers that know better should pass ``backend=`` to ``get_encoder`` instead;
    this exists so they can also ask what the fallback would do.
    """
    low = model.lower()
    if "qwen" in low or low.startswith("mlx-community"):
        return "mlx"
    return "model2vec"


def get_encoder(ws=None, *, model: str | None = None, instruction: str | None = None,
                backend: str | None = None) -> Encoder:
    """Build an encoder from a Workspace (or explicit model).

    ``backend`` (one of ``ENCODER_BACKENDS``) wins when given; otherwise the model
    id is sniffed (``sniff_backend``) for back-compat, so bare potion/Qwen names
    keep routing as they always did. ``instruction`` is ignored by the static
    model2vec backend (no instruction concept).
    """
    model = model or (ws.embed_model if ws is not None else DEFAULT_EMBED_MODEL)
    if instruction is None and ws is not None:
        instruction = ws.embedding_instruction
    backend = backend or sniff_backend(model)
    if backend == "mlx":
        return QwenEncoder(model_id=model, instruction=instruction)
    if backend == "st":
        return SentenceTransformerEncoder(model_id=model, instruction=instruction)
    if backend == "model2vec":
        return Model2VecEncoder(model_name=model)
    raise ValueError(f"unknown encoder backend {backend!r}; choose from {ENCODER_BACKENDS}")


def embed_texts(texts: Iterable[str], ws=None, *, encoder: Encoder | None = None,
                batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
    enc = encoder or get_encoder(ws)
    if batch_size is None:
        return enc.encode(list(texts), show_progress=show_progress)
    return enc.encode(list(texts), batch_size=batch_size, show_progress=show_progress)


def embed_corpus(ws, run_id: str, texts: list[str], *, encoder: Encoder | None = None,
                 batch_size: int | None = None, show_progress: bool = False) -> np.ndarray:
    """Embed the run's unique texts and save to ``ws.embeddings_path(run_id)``."""
    emb = embed_texts(texts, ws, encoder=encoder, batch_size=batch_size, show_progress=show_progress)
    path = ws.embeddings_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
