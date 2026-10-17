"""Encoder protocol tests: routing (no model load) + a real potion encode."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import flashtopic.embed as embed_mod
from flashtopic.config import Workspace
from flashtopic.embed import (
    Model2VecEncoder,
    QwenEncoder,
    SentenceTransformerEncoder,
    get_encoder,
    resolve_st_device,
    sniff_backend,
)


def test_get_encoder_routes_by_model_id():
    assert isinstance(get_encoder(Workspace(base_dir="/tmp/ft")), Model2VecEncoder)
    assert isinstance(get_encoder(model="minishlab/potion-base-8M"), Model2VecEncoder)
    assert isinstance(
        get_encoder(model="mlx-community/Qwen3-Embedding-4B-4bit-DWQ"), QwenEncoder
    )


def test_get_encoder_explicit_backend_wins():
    # An st model id sniffs to model2vec, but backend= overrides.
    enc = get_encoder(model="BAAI/bge-m3", backend="st")
    assert isinstance(enc, SentenceTransformerEncoder) and enc.model_id == "BAAI/bge-m3"
    assert isinstance(get_encoder(model="anything", backend="mlx"), QwenEncoder)
    assert isinstance(get_encoder(model="anything", backend="model2vec"), Model2VecEncoder)
    with pytest.raises(ValueError):
        get_encoder(model="anything", backend="tensorflow")


def test_sniff_backend_matches_historical_routing():
    assert sniff_backend("minishlab/potion-retrieval-32M") == "model2vec"
    assert sniff_backend("minishlab/potion-multilingual-128M") == "model2vec"
    assert sniff_backend("mlx-community/Qwen3-Embedding-8B-4bit-DWQ") == "mlx"
    assert sniff_backend("Qwen/Qwen3-Embedding-0.6B") == "mlx"
    assert sniff_backend("BAAI/bge-m3") == "model2vec"  # sniff can't know; pass backend="st"


def test_workspace_instruction_flows_to_qwen():
    ws = Workspace(base_dir="/tmp/ft", embed_model="mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
                   embedding_instruction="Retrieve passages.")
    enc = get_encoder(ws)
    assert isinstance(enc, QwenEncoder) and enc.instruction == "Retrieve passages."


def test_resolve_st_device_probes_and_passthrough(monkeypatch):
    monkeypatch.setattr(embed_mod, "_probe_torch_mps", lambda: True)
    assert resolve_st_device() == "mps"
    monkeypatch.setattr(embed_mod, "_probe_torch_mps", lambda: False)
    monkeypatch.setattr(embed_mod, "_probe_torch_cuda", lambda: True)
    assert resolve_st_device() == "cuda"
    monkeypatch.setattr(embed_mod, "_probe_torch_cuda", lambda: False)
    assert resolve_st_device() == "cpu"
    assert resolve_st_device("cuda") == "cuda"  # explicit choice is never second-guessed


def test_sentence_transformer_encoder_prefix_and_norm(monkeypatch):
    seen = {}

    class FakeST:
        def __init__(self, model_id, device=None):
            seen["model_id"], seen["device"] = model_id, device

        def encode(self, texts, batch_size=32, show_progress_bar=False,
                   normalize_embeddings=False, convert_to_numpy=True):
            seen["texts"] = list(texts)
            rng = np.random.default_rng(0)
            return rng.normal(size=(len(texts), 8)).astype(np.float32)

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    enc = SentenceTransformerEncoder("intfloat/multilingual-e5-large-instruct",
                                     instruction="Cluster sentences by topic",
                                     device="cpu")
    E = enc.encode(["hello", "world"])
    assert seen["model_id"] == "intfloat/multilingual-e5-large-instruct"
    assert seen["device"] == "cpu"
    assert seen["texts"][0] == "Instruct: Cluster sentences by topic\nQuery: hello"
    assert E.shape == (2, 8) and E.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)


class _FakeQwenTokenizer:
    """Deterministic tokenizer: one token per character (ord), so lengths vary."""

    eos_token_id = 151643

    def encode(self, text):
        return [ord(c) % 500 + 1 for c in text]


class _FakeQwenInner:
    """Causal fake transformer: hidden[b, t, :] depends only on tokens[b, :t+1]
    (a cumulative sum), exactly the property that makes right-padding safe."""

    def __call__(self, arr):
        import mlx.core as mx

        x = arr.astype(mx.float32)
        c = mx.cumsum(x, axis=1)
        return mx.stack([c, c * 0.5, mx.sin(c * 0.01), mx.cos(c * 0.01)], axis=-1)


class _FakeQwenModel:
    model = _FakeQwenInner()


@pytest.fixture
def fake_qwen_encoder():
    pytest.importorskip("mlx.core")
    enc = QwenEncoder(model_id="fake/qwen")
    enc._state = {"model": _FakeQwenModel(), "tokenizer": _FakeQwenTokenizer(),
                  "eos_id": _FakeQwenTokenizer.eos_token_id}
    return enc


def test_qwen_batched_equals_unbatched(fake_qwen_encoder):
    texts = ["a", "bb longer sentence here", "medium one", "x" * 40, "yz"]
    E1 = fake_qwen_encoder.encode(texts, batch_size=1)   # old per-text semantics
    E3 = fake_qwen_encoder.encode(texts, batch_size=3)   # padded batches, length-sorted
    assert E1.shape == E3.shape == (5, 4)
    np.testing.assert_allclose(E1, E3, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(E3, axis=1), 1.0, atol=1e-5)


def test_qwen_mean_pooling_ignores_padding(fake_qwen_encoder):
    fake_qwen_encoder.pooling = "mean"
    texts = ["short", "a considerably longer text to force padding"]
    E1 = fake_qwen_encoder.encode(texts, batch_size=1)
    E2 = fake_qwen_encoder.encode(texts, batch_size=2)
    np.testing.assert_allclose(E1, E2, atol=1e-5)


def test_potion_encode_shape_norm_and_semantics():
    pytest.importorskip("model2vec")
    enc = Model2VecEncoder("minishlab/potion-retrieval-32M")
    texts = ["workers voted to strike over wages", "the union debated a strike",
             "the bus arrived at the station on time"]
    try:
        E = enc.encode(texts)
    except Exception as e:  # offline / model not cached
        pytest.skip(f"potion unavailable: {e}")
    assert E.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)
    S = E @ E.T
    assert S[0, 1] > S[0, 2]  # the two strike sentences are more alike
