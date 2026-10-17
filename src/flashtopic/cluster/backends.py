"""Pluggable spherical k-means backends with capability-based auto-detection.

The flash-kmeans k-sweep is the point of the library, so clustering is GPU-first with
a portable fallback. Three backends behind one ``spherical_kmeans`` call:

* ``flash_mlx``  — Apple Silicon (``flash-kmeans-mlx``, native cosine).
* ``flash_cuda`` — NVIDIA/Triton (``flash-kmeans``, Euclidean on the unit sphere).
* ``fastkmeans`` — portable torch fallback (CPU or any torch device).

**Auto-detection is capability-based, never arch-string-based.** ``platform.machine()``
lies: it reports ``aarch64`` on ARM Linux (which can't run MLX) and ``x86_64`` under
Rosetta on Apple Silicon (which can). So we *probe* — import the backend and run a tiny
op — and pick the first that actually works. Arch info is logged for diagnostics only.

Every backend is fed L2-normalized vectors and returns L2-normalized centroids, so
Euclidean backends on the unit sphere are equivalent to spherical/cosine k-means.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

log = logging.getLogger("flashtopic.cluster")

BACKENDS = ("flash_mlx", "flash_cuda", "fastkmeans")


@dataclass
class KMeansResult:
    labels: np.ndarray  # (n,) int64
    centroids: np.ndarray  # (k, d) float32, L2-normalized
    inertia: float  # sum of cosine distance (1 - sim) to the assigned centroid
    backend: str


# --------------------------------------------------------------------------- #
# Capability probes (cached) — import + tiny op, no platform-string gating
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _probe_flash_mlx() -> bool:
    try:
        import mlx.core as mx
        import flash_kmeans_mlx  # noqa: F401

        if not mx.metal.is_available():
            return False
        mx.eval(mx.array([1.0]) + 1)  # confirm the device actually runs an op
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        log.debug("flash_mlx unavailable: %s", e)
        return False


@lru_cache(maxsize=1)
def _probe_flash_cuda() -> bool:
    try:
        import torch
        import flash_kmeans  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception as e:  # pragma: no cover
        log.debug("flash_cuda unavailable: %s", e)
        return False


@lru_cache(maxsize=1)
def _probe_fastkmeans() -> bool:
    try:
        import fastkmeans  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover
        log.debug("fastkmeans unavailable: %s", e)
        return False


_PROBES = {
    "flash_mlx": _probe_flash_mlx,
    "flash_cuda": _probe_flash_cuda,
    "fastkmeans": _probe_fastkmeans,
}


def available_backends() -> list[str]:
    return [b for b in BACKENDS if _PROBES[b]()]


def detect_backend() -> str:
    """First backend (flash_mlx → flash_cuda → fastkmeans) that actually runs."""
    for b in BACKENDS:
        if _PROBES[b]():
            log.info(
                "cluster backend %s (machine=%s system=%s) — capability-probed",
                b, platform.machine(), platform.system(),
            )
            return b
    raise RuntimeError(
        "no clustering backend available. Install one: "
        "`pip install flashtopic[mlx]` (Apple), `[cuda]` (NVIDIA), or `[cpu]` (portable). "
        f"(machine={platform.machine()}, system={platform.system()})"
    )


def resolve_backend(backend: str = "auto") -> str:
    if backend == "auto":
        return detect_backend()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS} or 'auto'")
    if not _PROBES[backend]():
        raise RuntimeError(f"backend {backend!r} requested but not importable/runnable")
    return backend


# --------------------------------------------------------------------------- #
# Shared finalization: normalize centroids, assign, compute cosine inertia
# --------------------------------------------------------------------------- #
def _finalize(X: np.ndarray, centroids: np.ndarray, backend: str,
              labels: np.ndarray | None = None) -> KMeansResult:
    C = centroids.astype(np.float32)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
    sims = X @ C.T
    if labels is None:
        labels = sims.argmax(1)
    labels = np.asarray(labels).astype(np.int64)
    inertia = float((1.0 - sims[np.arange(len(X)), labels]).sum())
    return KMeansResult(labels=labels, centroids=C, inertia=inertia, backend=backend)


def _l2norm(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


# --------------------------------------------------------------------------- #
# Backend implementations
# --------------------------------------------------------------------------- #
def _kmeans_flash_mlx(X, k, *, n_iter, seed, tol) -> KMeansResult:
    import mlx.core as mx
    import flash_kmeans_mlx as fk

    km = fk.FlashKMeans(d=X.shape[1], k=k, niter=n_iter, tol=tol, seed=seed, metric="cosine")
    labels = np.array(km.fit_predict(mx.array(X)))
    centroids = np.array(km.centroids_b[0])
    return _finalize(X, centroids, "flash_mlx", labels)


def _kmeans_flash_cuda(X, k, *, n_iter, seed, tol) -> KMeansResult:  # pragma: no cover - needs CUDA
    import torch
    import flash_kmeans as fk

    xt = torch.from_numpy(X).cuda()[None]  # (1, N, D)
    ids, centers, _ = fk.batch_kmeans_Euclid(xt, n_clusters=k, tol=tol)
    return _finalize(X, centers[0].cpu().numpy(), "flash_cuda", ids[0].cpu().numpy())


def _kmeans_fastkmeans(X, k, *, n_iter, seed, tol) -> KMeansResult:
    from fastkmeans import FastKMeans

    # fastkmeans' GPU path is CUDA/Triton; on a non-CUDA host (e.g. Apple Silicon)
    # gpu=True probes a device it can't use, so gate it on actual CUDA availability.
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    km = FastKMeans(d=X.shape[1], k=k, niter=n_iter, seed=seed, gpu=gpu)
    km.train(X)
    return _finalize(X, np.asarray(km.centroids), "fastkmeans")


_IMPLS = {
    "flash_mlx": _kmeans_flash_mlx,
    "flash_cuda": _kmeans_flash_cuda,
    "fastkmeans": _kmeans_fastkmeans,
}


def spherical_kmeans(
    X: np.ndarray,
    k: int,
    *,
    backend: str = "auto",
    n_iter: int = 50,
    seed: int = 0,
    tol: float = 1e-4,
    n_init: int = 1,
) -> KMeansResult:
    """Spherical (cosine) k-means on `X` (n, d) via the chosen/auto backend.

    `X` is L2-normalized here, so all backends — including Euclidean ones — cluster
    on the unit sphere. Returns labels, L2-normalized centroids, and cosine inertia.

    `n_init > 1` runs that many restarts (seeds ``seed, seed+1, …``) and keeps the
    lowest-inertia fit — cheap insurance against the local minima single-seed k-means
    hits at high k (the base a multi-resolution tree rests on).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    Xn = _l2norm(X)
    if Xn.shape[0] <= k:  # nothing to split
        return _finalize(Xn, Xn.copy(), resolve_backend(backend) if backend != "auto" else "trivial",
                         labels=np.arange(Xn.shape[0]))
    chosen = resolve_backend(backend)
    best: KMeansResult | None = None
    for i in range(max(1, n_init)):
        r = _IMPLS[chosen](Xn, k, n_iter=n_iter, seed=seed + i, tol=tol)
        if best is None or r.inertia < best.inertia:
            best = r
    return best
