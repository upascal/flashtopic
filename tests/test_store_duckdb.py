"""DuckDBStore round-trip + the FileStore/DuckDB parity that prevalence relies on.

Skipped entirely when the ``[duckdb]`` extra isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("duckdb")

from flashtopic.store import Chunk, DuckDBStore, FileStore  # noqa: E402


def _chunks(n=10, d=8, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        v = rng.normal(size=d).astype(np.float32)
        v /= np.linalg.norm(v)
        out.append(Chunk(chunk_id=str(i), text=f"atom {i}", embedding=v,
                         doc_idxs=[i % 3], sentence_idxs=[i],
                         metadata={"persons": "X" if i % 2 == 0 else "Y", "year": 1900 + i}))
    return out, d


def test_duckdb_roundtrip_query_and_isolation():
    chunks, d = _chunks()
    st = DuckDBStore(":memory:", metadata_columns=["persons"])
    st.create_set("s1", dim=d, model="potion")
    st.upsert("s1", chunks)
    st.create_set("s2", dim=d)
    st.upsert("s2", chunks[:1])
    assert st.list_sets() == ["s1", "s2"]

    # contiguous matrix in insertion order (not lexicographic '0','1','10',...)
    M, ids = st.get_matrix("s1")
    assert M.shape == (10, d) and M.flags["C_CONTIGUOUS"]
    assert ids == [str(i) for i in range(10)]

    # metadata (promoted column) + vector similarity in one query
    q = chunks[0].embedding
    rows = st.query("s1", vector=q, meta_predicate="persons = 'X'", min_similarity=0.0)
    assert rows and all(r.metadata["persons"] == "X" for r in rows)
    assert rows[0].chunk_id == "0" and abs(rows[0].score - 1.0) < 1e-5

    # JSON catch-all predicate
    rows2 = st.query("s1", meta_predicate="CAST(json_extract(metadata,'$.year') AS INT) >= 1905")
    assert {r.chunk_id for r in rows2} == {str(i) for i in range(5, 10)}

    # in-place upsert: same row count, updated text
    st.upsert("s1", [Chunk("0", "EDITED", q, doc_idxs=[0], sentence_idxs=[0], metadata={"persons": "X"})])
    M2, ids2 = st.get_matrix("s1")
    assert M2.shape == (10, d) and ids2 == ids
    assert st.query("s1", meta_predicate="chunk_id = '0'")[0].text == "EDITED"

    st.drop_set("s2")
    assert st.list_sets() == ["s1"]


def test_duckdb_matrix_feeds_kmeans():
    from flashtopic.cluster import available_backends, spherical_kmeans

    if not available_backends():
        pytest.skip("no clustering backend")
    chunks, d = _chunks()
    st = DuckDBStore(":memory:")
    st.create_set("s", dim=d)
    st.upsert("s", chunks)
    M, _ = st.get_matrix("s")
    labels = spherical_kmeans(M, 3, seed=0).labels
    assert labels.shape == (10,)


def test_filestore_duckdb_parity_runcontext(tmp_path):
    from flashtopic.prevalence import RunContext, build_surrogate, topics_from_labels

    chunks, d = _chunks(n=30, d=8)
    fs = FileStore(tmp_path)
    fs.create_set("s", dim=d)
    fs.upsert("s", chunks)
    ds = DuckDBStore(":memory:")
    ds.create_set("s", dim=d)
    ds.upsert("s", chunks)

    cf = RunContext.from_store(fs, "s")
    cd = RunContext.from_store(ds, "s")
    assert cf.texts == cd.texts
    assert np.allclose(cf.E, cd.E, atol=1e-5)
    assert cf.uid_docs == cd.uid_docs and cf.n_docs == cd.n_docs

    labels = np.arange(len(cf.texts)) % 4
    topic = max(topics_from_labels(labels), key=lambda t: t.n_members)
    assert abs(build_surrogate(cf, topic).naive - build_surrogate(cd, topic).naive) < 1e-9
