"""Topic — the unit prevalence is measured for.

Generalized from an earlier pilot's two-file (thematic + person) schema to a single
query-agnostic record: an id, an optional label/definition, its member unique-sentence
ids (the cheap membership surrogate), and optional distinctive terms + exemplars (for
the codebook). Built directly from the clustering's per-unique labels.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Topic:
    topic_id: str
    member_uids: list[int]
    label: str | None = None
    definition: str | None = None
    distinctive_terms: list[str] = field(default_factory=list)
    exemplars: list[str] = field(default_factory=list)  # coverage: the cluster's range (for labeling)
    typical_exemplars: list[str] = field(default_factory=list)  # nearest the centroid (for codebooks)
    coherence: dict[str, float] | None = None  # cosine-to-centroid {median, p10}
    label_coherence: str | None = None  # labeler's verdict: coherent | mixed | noise
    discerned_subtopics: list[str] = field(default_factory=list)  # labeler's facets when mixed

    @property
    def n_members(self) -> int:
        return len(self.member_uids)


def topics_from_labels(
    labels,
    *,
    label_text: dict[int, str] | None = None,
    definitions: dict[int, str] | None = None,
    distinctive_terms: dict[int, list[str]] | None = None,
    exemplars: dict[int, list[str]] | None = None,
    typical_exemplars: dict[int, list[str]] | None = None,
    coherence: dict[int, dict[str, float]] | None = None,
    label_meta: dict[int, dict] | None = None,
) -> list[Topic]:
    """Group per-unique cluster ``labels`` into Topics, attaching any labels/terms.

    ``labels[uid]`` is the cluster id of unique sentence ``uid``. The optional maps are
    keyed by cluster id (e.g. from the LLM labeling + ``terms.distinctive_terms``).
    """
    label_text = label_text or {}
    definitions = definitions or {}
    distinctive_terms = distinctive_terms or {}
    exemplars = exemplars or {}
    typical_exemplars = typical_exemplars or {}
    coherence = coherence or {}
    label_meta = label_meta or {}

    by_cluster: dict[int, list[int]] = defaultdict(list)
    for uid, c in enumerate(labels):
        by_cluster[int(c)].append(uid)

    topics: list[Topic] = []
    for c in sorted(by_cluster):
        topics.append(
            Topic(
                topic_id=f"T{c}",
                member_uids=by_cluster[c],
                label=label_text.get(c),
                definition=definitions.get(c),
                distinctive_terms=distinctive_terms.get(c, []),
                exemplars=exemplars.get(c, []),
                typical_exemplars=typical_exemplars.get(c, []),
                coherence=coherence.get(c),
                label_coherence=label_meta.get(c, {}).get("coherence"),
                discerned_subtopics=list(label_meta.get(c, {}).get("discerned_subtopics") or []),
            )
        )
    return topics
