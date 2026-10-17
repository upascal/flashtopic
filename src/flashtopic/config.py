"""Workspace — the single config object that replaces global state.

The experimental ancestor of this library kept a module-level ``DATA_DIR`` and a
``run_dir(run_id)`` helper plus corpus-specific defaults (domain stopwords,
an embedding instruction) scattered across modules. ``Workspace`` folds all of that
into one explicit, query-agnostic object you pass to the stages:

* ``base_dir`` — root for every run's on-disk artifacts (corpus/, embeddings, labels).
* ``corpus_topic`` — *optional* overarching subject of the corpus, if it has one. When
  set, it is injected into LLM labeling prompts so clusters are named *below* that level
  rather than restating it; left empty (the default), labeling treats the corpus as an
  arbitrary collection with no single topic.
* ``embedding_instruction`` — optional instruction prefix for instruction-tuned embedders.
* ``domain_stopwords`` — *optional* corpus-wide words to suppress in distinctive-term
  scoring (terms so ubiquitous in this corpus they carry no distinguishing signal).
* ``embed_model`` / ``cluster_backend`` — which encoder and k-means backend to use.
* ``spacy_model`` — spaCy pipeline for the NER + distinctiveness stages (default the
  small English model; set ``"en_core_web_trf"`` for transformer-accuracy NER).

A run's files live under ``base_dir/<run_id>/`` with the same layout the pipeline has
always used, but rooted here instead of at an import-time global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EMBED_MODEL = "minishlab/potion-retrieval-32M"


@dataclass
class Workspace:
    base_dir: Path
    corpus_topic: str = ""
    embedding_instruction: str | None = None
    domain_stopwords: frozenset[str] = field(default_factory=frozenset)
    embed_model: str = DEFAULT_EMBED_MODEL
    cluster_backend: str = "auto"  # auto | flash_mlx | flash_cuda | fastkmeans
    spacy_model: str = "en_core_web_sm"  # NER + lemmatization; "en_core_web_trf" for transformer accuracy

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        if not isinstance(self.domain_stopwords, frozenset):
            self.domain_stopwords = frozenset(self.domain_stopwords)

    def replace(self, **changes) -> "Workspace":
        """A copy with some fields overridden (e.g. a different backend per call)."""
        from dataclasses import replace as _replace

        return _replace(self, **changes)

    # ---- run-scoped paths (created on demand, rooted at base_dir) ---------- #
    def run_dir(self, run_id: str) -> Path:
        p = self.base_dir / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def corpus_dir(self, run_id: str) -> Path:
        p = self.run_dir(run_id) / "corpus"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def embeddings_path(self, run_id: str) -> Path:
        """Unique-sentence embedding matrix (N, d), float32, L2-normalized."""
        return self.run_dir(run_id) / "embeddings.npy"

    def sentences_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "sentences.jsonl"

    def body_uniques_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "body_uniques.txt"

    def body_membership_path(self, run_id: str) -> Path:
        """Per unique body sentence → occurrences (counts, sentence_idxs, doc_idxs)."""
        return self.corpus_dir(run_id) / "body_membership.jsonl"

    def named_entities_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "named_entities.jsonl"

    def entity_df_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "entity_df.json"

    def distinctiveness_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "sentence_distinctiveness.jsonl"

    def lemma_df_path(self, run_id: str) -> Path:
        return self.corpus_dir(run_id) / "lemma_df.json"

    def labels_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "topics.jsonl"

    # ---- latest-run pointer ---------------------------------------------- #
    def latest_pointer(self) -> Path:
        return self.base_dir / "latest_run.txt"

    def write_latest(self, run_id: str) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.latest_pointer().write_text(run_id)

    def read_latest(self) -> str:
        p = self.latest_pointer()
        if not p.exists():
            raise FileNotFoundError(f"no latest-run pointer at {p}")
        return p.read_text().strip()
