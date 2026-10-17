# flashtopic

Fast GPU topic modeling + debiased prevalence scoring for large text corpora.

> **Status: early research code.** It works and is tested, but the APIs are still moving
> and may change without notice. Not on PyPI yet. Issues and questions welcome.

Two pillars:

1. **Flash topic modeling** — embed (potion by default), sweep k with GPU flash-kmeans,
   label clusters with distinctive terms + contrastive LLM prompts. The flash-kmeans
   backend makes sweeping k over large corpora actually feasible.
2. **Prevalence scoring** — estimate how prevalent each topic is across *documents*,
   debiasing the cheap topic assignment against a small probability-sampled gold set
   (PPI/ACC) for an unbiased estimate with a valid confidence interval.

Works on **any** text corpus — news, papers, scraped web, books, survey/LLM responses.
It assumes no generating function: a corpus needn't have a single driving topic. When it
*does* have one, an optional `corpus_topic` hint sharpens the labels. Everything
corpus-specific (the optional topic, domain stopwords, embedding instruction, backends)
lives in one `Workspace` config object.

## Install

Not on PyPI yet, so install straight from GitHub, picking the extra for your hardware:

```bash
pip install "flashtopic[mlx] @ git+https://github.com/upascal/flashtopic"    # Apple Silicon: flash-kmeans-mlx (cosine) + Qwen embeddings
pip install "flashtopic[cuda] @ git+https://github.com/upascal/flashtopic"   # NVIDIA: flash-kmeans (Triton)
pip install "flashtopic[cpu] @ git+https://github.com/upascal/flashtopic"    # portable torch fallback (fastkmeans)
python -m spacy download en_core_web_sm   # required for sentence parsing + NER

pip install "flashtopic[trf] @ git+https://github.com/upascal/flashtopic"    # optional: transformer-accuracy spaCy NER
python -m spacy download en_core_web_trf  # optional: higher-accuracy NER / lemmas
```

Set `Workspace(spacy_model="en_core_web_trf")` to run the NER + distinctiveness stages
on the transformer model; sentence segmentation stays on `en_core_web_sm`.

Install at least one clustering backend (`[mlx]`, `[cuda]`, or `[cpu]`); `backend="auto"`
**capability-probes** (import + tiny op) and picks whichever runs — it never trusts
`platform.machine()`, which lies under Rosetta and on ARM Linux. The clustering core
needs **no API key**; only the LLM-labeling and prevalence-judge stages do — copy
`.env.example` to `.env` and set `OPENROUTER_API_KEY`, or set it in the environment.

## Quickstart (Python)

```python
from flashtopic import Workspace
from flashtopic.ingest import load_docs
from flashtopic.pipeline import run_pipeline

ws = Workspace(base_dir="runs")              # no corpus_topic: a general, mixed corpus
docs = load_docs("corpus.jsonl.gz")          # defaults to the `text` field

res = run_pipeline(docs, ws, run_id="v1", k_range=(20, 200), strategy="elbow")
print(res.k, "topics")                       # swept k via GPU flash-kmeans
for t in sorted(res.topics, key=lambda t: -t.n_members)[:10]:
    print(t.topic_id, t.n_members, t.distinctive_terms[:5])
```

If your corpus *does* have a single overarching subject, pass it so labels land *below*
that level, and suppress corpus-ubiquitous words:

```python
ws = Workspace(base_dir="runs", corpus_topic="what happened at event X",
               domain_stopwords={"event", "x"})
docs = load_docs("corpus.jsonl.gz", text_field="completion")  # e.g. LLM-completion records
```

Add `label_model="anthropic/claude-haiku-4.5"` to LLM-label the clusters (needs a key).

## Storeless core + optional stores

`run_pipeline` is a thin wrapper over a **storeless core** you can use directly: docs (and
optionally precomputed embeddings) in, a `ClusterResult` of plain NumPy/Python objects out
— no `Workspace`, no disk, no DB. This is the part to reuse from other projects.

```python
from flashtopic import cluster_corpus

res = cluster_corpus(docs, embeddings=E, k=80)   # pass embeddings= to reuse a vector store's matrix
res.labels, res.topics, res.unique_index, res.embeddings   # everything, in memory
```

Persistence is a separate, optional layer — a `VectorStore` that keeps chunks + embeddings
+ metadata in **one keyed store**, so managing many embedding sets is a `set_id` (not a
sprawl of `.npy` files) and a `metadata AND cosine` filter is a single query:

```python
from flashtopic.store import DuckDBStore, chunks_from_result

store = DuckDBStore("runs.duckdb", metadata_columns=["persons"])   # or FileStore (no extra deps)
store.create_set("q1", dim=res.embeddings.shape[1])
store.upsert("q1", chunks_from_result(res))

# metadata + vector similarity in ONE call — no .npy ↔ sidecar join
hits = store.query("q1", vector=q, meta_predicate="persons = 'X'", min_similarity=0.7)
M, ids = store.get_matrix("q1", where="persons = 'X'")   # materialize a subset → re-cluster / score
```

`FileStore` (the current `.npy`+JSONL layout, zero extra deps) and `DuckDBStore` (the
`[duckdb]` extra; `FLOAT[]` columns + `list_cosine_similarity`, DuckDB-WASM friendly) both
satisfy the `VectorStore` protocol; sqlite-vec / pgvector are conforming alternates a host
project can add. Pass `store=` to `run_pipeline` to persist there instead of the default
`FileStore`.

## Multi-resolution levels

When no single K is obviously right, cluster at **several nested resolutions** (coarse →
fine) instead of agonizing over one. `cluster_levels` returns a ladder of clusterings,
the parent map between adjacent levels, and how cleanly each level nests:

```python
from flashtopic import cluster_levels

# guaranteed-nested: one fine clustering, cut an atom-centroid tree at several heights
res = cluster_levels(E, strategy="dendrogram", cuts=[20, 80, 260])
res.ks            # [20, 80, 260]  (coarse → fine)
res.labels        # per-level point labels; res.parents[i][fine_id] -> coarse id
res.nesting[0].purity   # ~1.0 — fine levels nest by construction

# or independent k-means at each level, with nesting *measured* not enforced
res = cluster_levels(E, strategy="flat", levels=[40, 150, 400])
```

The levels map onto the topic browser's `meta_topic_id` / `super_topic_id` hierarchy.
(Auto-selecting the best nesting-coherent k-ladder from a wide sweep is a planned
follow-up; the scoring helpers in `flashtopic.cluster.levels` are already there.)

**Before you commit to a resolution (or spend on LLM labels), gate on stability** — the
real question isn't *k*, it's whether the structure is reproducible:

```python
from flashtopic import cluster_levels, level_stability

# 1. stabilize the base: restarts kill the single-seed local minima the tree rests on
#    and min_size rolls singleton/trivia nodes into their nearest cluster
res = cluster_levels(E, strategy="dendrogram", cuts=[10, 60, 120],
                     atom_k=200, n_init=4, min_size=3)

# 2. the gate: re-cluster at a few seeds, look at per-level label agreement (ARI)
rep = level_stability(E, strategy="dendrogram", cuts=[10, 60, 120],
                      atom_k=200, n_init=4, seeds=(0, 1, 2, 3))
rep.ks, rep.mean_ari   # low fine-level ARI ⇒ don't tag yet — coarsen, prune, or raise n_init
```

`prune_small_clusters(labels, X, min_size)` is also exposed standalone for ad-hoc use.

## CLI

```bash
flashtopic backends                                  # what's installed
flashtopic cluster corpus.jsonl.gz --k-range 20 200 --strategy elbow
flashtopic cluster corpus.jsonl --k 60 --merge \
    --label-model anthropic/claude-haiku-4.5 --topic "what happened at X"  # --topic optional
```

## Prevalence

```python
from flashtopic.prevalence import (RunContext, build_surrogate, design_sample,
                                    build_codebook, run_judge, verdicts_to_y, estimate_prevalence)

ctx = RunContext.from_workspace(ws, "v1")            # loads the run's artifacts
# also: RunContext.from_result(res)  (storeless, in-memory)  ·  RunContext.from_store(store, "q1")
topic = max(res.topics, key=lambda t: t.n_members)
sur = build_surrogate(ctx, topic)                    # cheap per-doc membership + cosine score
design = design_sample(sur)                          # score-stratified gold sample
cb = build_codebook(topic, corpus_topic=ws.corpus_topic)
verdicts = run_judge(cb, design, ctx.doc_texts, model="anthropic/claude-haiku-4.5")
y = verdicts_to_y(design, verdicts)
res = estimate_prevalence(topic.topic_id, sur.f_memb.astype(int),
                          design.f_s, y, design.pi_s, design.strata)
print(res.naive, "→", res.pi_ppi, res.ci_low, res.ci_high)   # debiased prevalence + CI
```

The estimator is design-based (PPI primary, ACC cross-check) and pure-numpy: the naive
membership prevalence is what it *corrects*, using the gold set to subtract the cheap
surrogate's misclassification. Rare-topic CIs are recall-limited — the result carries a
`reliable`/`min_cell` flag.

### Classifier-fed surrogate (Jev pilot)

The estimator is unbiased for *any* surrogate; the surrogate only sets the CI width, through
the gold residuals `f − Y`. So a better `f` buys a tighter interval at the same gold budget —
or the same interval with fewer gold labels, i.e. more topics per dollar. `flashtopic.prevalence`
can build that `f` from a **decision model** (TypeSafe's Jev via OpenRouter's Decisions
endpoint, or any replica speaking the same wire format) used as a *feature extractor*: each
topic becomes a few typed questions (presence, depth-of-treatment, parent theme, sibling
choice), the answers plus the deterministic signals feed a small cross-fitted logistic
regression, and *that* is the surrogate. The generative judge stays the gold.

```python
from flashtopic.prevalence import JevClassifier, MockClassifier, run_pilot_sync, write_report

# designs: {topic_id: (SampleDesign, gold_y)} from design_sample + run_judge, as above
res = run_pilot_sync(ctx, topics, designs, JevClassifier("typesafe/jev-1.13"))
write_report(res)           # flashtopic_output/jev_pilot.md — SE per candidate, verdict
```

The pilot runs the classifier on the gold documents only (the SE needs nothing else), so it
costs ~gold-size input-token calls, and reports whether the classifier-fed regression beats
both today's membership surrogate and a deterministic-features-only regression. Run
`python -m flashtopic.prevalence.pilot` for a no-network self-check with `MockClassifier`.

## Backends

| backend | package | device | metric |
|---|---|---|---|
| `flash_mlx` | `flash-kmeans-mlx` | Apple Silicon | cosine (native) |
| `flash_cuda` | `flash-kmeans` | NVIDIA/Triton | Euclidean on unit sphere |
| `fastkmeans` | `fastkmeans` | any torch device | Euclidean on unit sphere |

All cluster L2-normalized vectors (≡ spherical k-means); centroids are renormalized.

## Term scorer

`distinctive_terms(..., method=...)`: `log_odds` (Monroe weighted log-odds-ratio with a
Dirichlet prior — default, most robust at *distinguishing*), `ctfidf` (BERTopic-style,
frequency-driven), or `tfidf` (legacy).

## Dev notes

- Setup: `scripts/dev-setup.sh [mlx|cuda|cpu]` (creates the venv, editable-installs,
  downloads the spaCy model). Then `pytest`.
- **macOS + uv editable installs:** uv marks its editable `.pth` files with the macOS
  `UF_HIDDEN` flag, and CPython's `site.py` skips hidden `.pth` files — so
  `import flashtopic` silently fails after `uv pip install -e .`. The `.pth` is correct
  (it just adds `src/`); this is a uv-on-macOS quirk, **not** a problem for normal
  `pip install flashtopic` from a wheel. `dev-setup.sh` fixes it by dropping a plain
  non-hidden `.pth` we own (survives future installs). `pytest` is immune either way
  (`pythonpath = ["src"]`); for ad-hoc scripts, `PYTHONPATH=src python …` also works.
