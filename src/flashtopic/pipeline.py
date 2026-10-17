"""End-to-end topic-modeling pipeline: docs → clusters → labeled topics.

Two layers:

* :func:`cluster_corpus` — the **storeless core**. Docs (and optionally precomputed
  embeddings) in; a :class:`ClusterResult` of plain NumPy/Python objects out. It builds
  the sentence corpus, embeds the unique sentences, sweeps k with the flash-kmeans
  backend, scores distinctive terms + coverage exemplars per cluster, optionally merges
  near-duplicates, optionally LLM-labels, and returns everything in memory. No
  ``Workspace``, no disk, no DB — so it composes with any storage (pass ``embeddings=``
  straight from a vector store, BERTopic-style).
* :func:`run_pipeline` — a thin convenience wrapper that runs the core then persists the
  run's artifacts (membership, sentences, embeddings, topics) through a ``Workspace`` so
  the prevalence subpackage can load them. Signature/behavior unchanged.

Each stage is also importable on its own.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from flashtopic.cluster import SweepResult, spherical_kmeans, sweep_k
from flashtopic.corpus import SentenceRecord, UniqueIndex, build_sentence_corpus, build_unique_index
from flashtopic.embed import embed_texts, get_encoder
from flashtopic.sampling import cluster_coherence, coverage_exemplars, typical_exemplars
from flashtopic.terms import distinctive_terms

_SIBLING_CAP = 1500


@dataclass
class ClusterResult:
    """Everything :func:`cluster_corpus` produces, in memory — no disk/DB coupling."""

    labels: np.ndarray  # (n_uniques,) cluster id per unique sentence
    topics: list  # list[flashtopic.prevalence.topics.Topic]
    unique_index: UniqueIndex  # unique texts + sentence_idxs/doc_idxs (membership)
    embeddings: np.ndarray  # (n_uniques, d) L2-normalized — the matrix a store would hold
    records: list[SentenceRecord]  # so doc_texts / sentences can be reconstructed downstream
    k: int
    n_docs: int
    sweep: SweepResult | None = None
    dropped_generic_terms: list[str] = field(default_factory=list)
    terms_by_c: dict[int, list[str]] = field(default_factory=dict)
    exemplars_by_c: dict[int, list[str]] = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    n_docs: int
    n_uniques: int
    k: int
    labels: np.ndarray  # (n_uniques,) cluster id per unique sentence
    topics: list  # list[flashtopic.prevalence.topics.Topic]
    sweep: SweepResult | None = None
    dropped_generic_terms: list[str] = field(default_factory=list)


def _run_clusterer(clusterer, E, k, *, backend, seed) -> np.ndarray:
    """Default = spherical k-means; a custom ``clusterer`` (``.fit(X, k, seed=)``) overrides."""
    if clusterer is not None:
        return np.asarray(clusterer.fit(E, k, seed=seed).labels)
    return spherical_kmeans(E, k, backend=backend, seed=seed).labels


@dataclass
class ClusterEvidence:
    terms: dict[int, list[str]]
    exemplars: dict[int, list[str]]
    typical: dict[int, list[str]]
    coherence: dict[int, dict[str, float]]
    members: dict[int, np.ndarray]


def _cluster_evidence(
    labels, ui, E, *, domain_stopwords, n_exemplars, n_terms, seed,
    min_df: int = 2, method: str = "log_odds",
):
    """Per cluster: member uids, distinctive terms (vs a sibling sample), exemplars, coherence.

    Two exemplar sets: *coverage* (spans the cluster, for the labeler) and *typical*
    (nearest the centroid, for codebooks). Coherence is cosine-to-centroid.

    Distinctive terms are scored on *document*-weighted presence: each unique sentence
    contributes proportionally to the number of distinct documents it appears in
    (``len(set(ui.doc_idxs[uid]))``), and a term must clear ``min_df`` distinct documents
    to be eligible — so one verbose document can't dominate a cluster's label.
    """
    rng = np.random.default_rng(seed)
    # Distinct-document weight per unique sentence (dedup repeated doc ids).
    doc_w = [len(set(d)) for d in ui.doc_idxs]
    terms_by_c: dict[int, list[str]] = {}
    exemplars_by_c: dict[int, list[str]] = {}
    typical_by_c: dict[int, list[str]] = {}
    coherence_by_c: dict[int, dict[str, float]] = {}
    members_by_c: dict[int, np.ndarray] = {}
    for c in sorted(set(labels.tolist())):
        members = np.where(labels == c)[0]
        members_by_c[c] = members
        member_uids = members
        non = np.where(labels != c)[0]
        sib = non if len(non) <= _SIBLING_CAP else rng.choice(non, _SIBLING_CAP, replace=False)
        sib_texts = [ui.texts[i] for i in sib]
        sib_weights = [float(doc_w[i]) for i in sib]
        if len(member_uids) > _SIBLING_CAP:
            sel = rng.choice(len(member_uids), _SIBLING_CAP, replace=False)
            member_uids = member_uids[sel]
        member_texts = [ui.texts[i] for i in member_uids]
        member_weights = [float(doc_w[i]) for i in member_uids]
        member_doc_idxs = [ui.doc_idxs[i] for i in member_uids]
        terms_by_c[c] = distinctive_terms(
            member_texts, sib_texts, n_terms=n_terms, extra_stopwords=domain_stopwords,
            method=method, member_weights=member_weights, sibling_weights=sib_weights,
            member_doc_idxs=member_doc_idxs, min_df=min_df,
        )
        ex_ids = coverage_exemplars(E[members], members.tolist(), n_exemplars, seed=seed)
        exemplars_by_c[c] = [ui.texts[i] for i in ex_ids]
        typical_by_c[c] = [ui.texts[i] for i in typical_exemplars(E[members], members.tolist(), 10)]
        coherence_by_c[c] = cluster_coherence(E[members])
    return ClusterEvidence(terms_by_c, exemplars_by_c, typical_by_c, coherence_by_c, members_by_c)


def cluster_corpus(
    docs,
    *,
    embeddings: np.ndarray | None = None,
    encoder=None,
    clusterer=None,
    domain_stopwords: Iterable[str] = (),
    corpus_topic: str = "",
    k: int | None = None,
    k_range=None,
    backend: str = "auto",
    strategy: str = "elbow",
    n_exemplars: int = 20,
    n_terms: int = 12,
    merge: bool = False,
    merge_tau: float = 0.5,
    label_model: str | None = None,
    seed: int = 0,
    show_progress: bool = False,
) -> ClusterResult:
    """Storeless core: ``docs`` → :class:`ClusterResult` (no Workspace, no disk).

    Provide either ``k`` (fixed) or ``k_range`` (swept by ``strategy``). Pass
    ``embeddings=`` to reuse a precomputed ``(n_uniques, d)`` matrix (e.g. straight from a
    vector store) and skip the encoder entirely — BERTopic's ``fit_transform(docs,
    embeddings=...)`` convention. LLM labeling runs only when ``label_model`` is given.
    """
    from flashtopic.prevalence.topics import topics_from_labels

    domain_stopwords = frozenset(domain_stopwords)

    # 1. corpus
    records = build_sentence_corpus(docs)
    ui = build_unique_index(records, "body")
    n_docs = len({r.doc_idx for r in records})

    # 2. embed (or accept precomputed)
    if embeddings is not None:
        E = np.asarray(embeddings, dtype=np.float32)
        if E.shape[0] != len(ui.texts):
            raise ValueError(
                f"embeddings has {E.shape[0]} rows but the corpus has "
                f"{len(ui.texts)} unique sentences"
            )
    else:
        enc = encoder or get_encoder()
        E = embed_texts(ui.texts, encoder=enc, show_progress=show_progress)

    # 3. cluster (sweep or fixed)
    sweep = None
    if k_range is not None:
        sweep = sweep_k(E, k_range, backend=backend, strategy=strategy, seed=seed)
        labels, k = sweep.result.labels.copy(), sweep.best_k
    elif k is not None:
        labels = _run_clusterer(clusterer, E, k, backend=backend, seed=seed)
    else:
        raise ValueError("provide either k or k_range")

    # 4. per-cluster evidence
    ev = _cluster_evidence(
        labels, ui, E, domain_stopwords=domain_stopwords,
        n_exemplars=n_exemplars, n_terms=n_terms, seed=seed,
    )

    # 5. optional near-duplicate merge (remaps cluster ids)
    dropped: list[str] = []
    if merge:
        from flashtopic.merge import complete_linkage_term_merge

        cids = sorted(ev.terms)
        mapping, dropped = complete_linkage_term_merge(
            [ev.terms[c] for c in cids], tau=merge_tau,
            extra_stopwords=set(domain_stopwords),
        )
        old_to_new = {c: int(mapping[i]) for i, c in enumerate(cids)}
        labels = np.array([old_to_new[int(c)] for c in labels], dtype=np.int64)
        ev = _cluster_evidence(
            labels, ui, E, domain_stopwords=domain_stopwords,
            n_exemplars=n_exemplars, n_terms=n_terms, seed=seed,
        )
        k = len(set(labels.tolist()))

    # 6. optional LLM labeling
    label_text: dict[int, str] = {}
    label_meta: dict[int, dict] = {}
    if label_model:
        from flashtopic.label import AtomEvidence, label_atoms_all

        evidence = {c: AtomEvidence(atom_id=c, terms=ev.terms[c], exemplars=ev.exemplars[c])
                    for c in ev.terms}
        labeled = asyncio.run(label_atoms_all(
            evidence, model=label_model, corpus_topic=corpus_topic
        ))
        label_text = {c: (v.get("specific_topic") or None) for c, v in labeled.items()}
        label_meta = labeled

    # 7. topics
    topics = topics_from_labels(
        labels, label_text=label_text, distinctive_terms=ev.terms, exemplars=ev.exemplars,
        typical_exemplars=ev.typical, coherence=ev.coherence, label_meta=label_meta,
    )

    return ClusterResult(
        labels=labels, topics=topics, unique_index=ui, embeddings=E, records=records,
        k=k, n_docs=n_docs, sweep=sweep, dropped_generic_terms=dropped,
        terms_by_c=ev.terms, exemplars_by_c=ev.exemplars,
    )


def _write_sentences(ws, run_id, records):
    """Persist sentences.jsonl (full per-sentence records) so doc texts can be rebuilt."""
    with ws.sentences_path(run_id).open("w") as f:
        for r in records:
            f.write(json.dumps({"sentence_idx": r.sentence_idx, "doc_idx": r.doc_idx,
                                "doc_id": r.doc_id, "kind": r.kind, "text": r.text}) + "\n")


def _persist_result(ws, run_id, result: ClusterResult, store=None) -> None:
    """Write a :class:`ClusterResult`'s artifacts under ``ws`` in the standard layout.

    The chunks + embeddings go through the :class:`~flashtopic.store.VectorStore` (a
    :class:`~flashtopic.store.FileStore` over ``ws`` by default — same ``.npy``+JSONL
    files); sentences and topics are run artifacts written directly.
    """
    from flashtopic.store import FileStore, chunks_from_result

    store = store or FileStore.from_workspace(ws)
    ws.run_dir(run_id)
    _write_sentences(ws, run_id, result.records)
    store.create_set(run_id, dim=int(result.embeddings.shape[1]), model=ws.embed_model)
    store.upsert(run_id, chunks_from_result(result))
    with ws.labels_path(run_id).open("w") as f:
        for t in result.topics:
            f.write(json.dumps({"topic_id": t.topic_id, "label": t.label,
                                "n_members": t.n_members,
                                "distinctive_terms": t.distinctive_terms[:12],
                                "coherence": t.coherence,
                                "label_coherence": t.label_coherence}) + "\n")
    ws.write_latest(run_id)


def run_pipeline(
    docs,
    ws,
    *,
    run_id: str = "run",
    k: int | None = None,
    k_range=None,
    backend: str = "auto",
    strategy: str = "elbow",
    encoder=None,
    n_exemplars: int = 20,
    n_terms: int = 12,
    merge: bool = False,
    merge_tau: float = 0.5,
    label_model: str | None = None,
    seed: int = 0,
    show_progress: bool = False,
    store=None,
):
    """Run the full pipeline on `docs` and persist the run; returns a :class:`PipelineResult`.

    Thin wrapper over :func:`cluster_corpus` (the storeless core) plus a persistence step.
    Provide either ``k`` (fixed) or ``k_range`` (swept by ``strategy``). LLM labeling runs
    only when ``label_model`` is given (needs an API key); otherwise topics carry
    distinctive terms + exemplars but no label. Pass ``store=`` a
    :class:`~flashtopic.store.VectorStore` (e.g. ``DuckDBStore``) to persist chunks +
    embeddings there instead of the default ``FileStore`` over ``ws``.
    """
    result = cluster_corpus(
        docs,
        encoder=encoder or get_encoder(ws),
        domain_stopwords=ws.domain_stopwords,
        corpus_topic=ws.corpus_topic,
        k=k,
        k_range=k_range,
        backend=backend,
        strategy=strategy,
        n_exemplars=n_exemplars,
        n_terms=n_terms,
        merge=merge,
        merge_tau=merge_tau,
        label_model=label_model,
        seed=seed,
        show_progress=show_progress,
    )
    _persist_result(ws, run_id, result, store=store)

    return PipelineResult(
        run_id=run_id, n_docs=result.n_docs, n_uniques=len(result.unique_index.texts),
        k=result.k, labels=result.labels, topics=result.topics, sweep=result.sweep,
        dropped_generic_terms=result.dropped_generic_terms,
    )
