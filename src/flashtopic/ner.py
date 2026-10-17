"""Per-sentence named entities + entity rarity, computed once over the corpus.

Runs over ORIGINAL-case unique sentences (NER depends on capitalization). Entity
"rarity" is document frequency over unique sentences: an entity in one sentence pops,
a ubiquitous one does not. Normalization is light (case-fold + strip trailing
possessive / leading article); NO coreference, so "Curie" and "Marie Curie" count
separately. Query-agnostic — the core functions take a list of texts.
"""

from __future__ import annotations

import json
import logging
import math
import re

logger = logging.getLogger(__name__)

# Generic, useful entity labels. Numerics (CARDINAL/ORDINAL/MONEY/…) and EVENT are
# dropped as noise. Add "NORP" (nationalities / religious / political groups) via the
# `labels=` arg if group entities matter for your corpus.
DEFAULT_LABELS = ("PERSON", "ORG", "GPE", "LOC", "DATE")
_DEFAULT_MODEL = "en_core_web_sm"

_NLP = None
_NLP_MODEL = None
_GPU_ACTIVE = False


def _activate_gpu() -> bool:
    """Best-effort GPU for spaCy: CUDA via cupy, or Apple-Metal via torch MPS."""
    import spacy

    try:
        return bool(spacy.prefer_gpu())
    except Exception:
        return False


def _nlp(model: str = _DEFAULT_MODEL):
    """Lazily load an NER-only spaCy pipeline; reloads if `model` changes.

    Requests the GPU first (torch-MPS on Apple Silicon, CUDA elsewhere) so a
    transformer model's forward pass runs accelerated. Falls back to
    ``en_core_web_sm`` — loudly — when the requested model isn't installed, and
    warns when a transformer model ends up on CPU (a missing torch install would
    otherwise silently make the pass an order of magnitude slower).
    """
    global _NLP, _NLP_MODEL, _GPU_ACTIVE
    if _NLP is None or _NLP_MODEL != model:
        import spacy

        _GPU_ACTIVE = _activate_gpu()
        disable = ["parser", "lemmatizer", "tagger", "attribute_ruler"]
        try:
            nlp = spacy.load(model, disable=disable)
        except OSError:
            if model == _DEFAULT_MODEL:
                raise
            logger.warning(
                "spaCy model %r not installed (python -m spacy download %s) — "
                "falling back to %s", model, model, _DEFAULT_MODEL,
            )
            nlp = spacy.load(_DEFAULT_MODEL, disable=disable)
        if "transformer" in nlp.pipe_names and not _GPU_ACTIVE:
            logger.warning(
                "NER model %r is a transformer but no GPU is active "
                "(torch MPS/CUDA unavailable) — expect a much slower pass", model
            )
        else:
            logger.info(
                "NER pipeline %r loaded (components=%s, gpu=%s)",
                model, nlp.pipe_names, _GPU_ACTIVE,
            )
        _NLP = nlp
        _NLP_MODEL = model  # cache under the REQUESTED name so a fallback isn't re-tried
    return _NLP


def release() -> None:
    """Drop the cached pipeline so a long-lived process can reclaim its memory
    (a transformer model holds hundreds of MB; reload from page cache is cheap)."""
    global _NLP, _NLP_MODEL
    _NLP = None
    _NLP_MODEL = None


# --- Surface + key normalization ---------------------------------------------
_QUOTES = "\"'`“”‘’"
_EDGE_TRIM = " \t:;,.!?·*_#`-–—()[]{}/" + _QUOTES
_WS = re.compile(r"\s+")
_ACRONYM = re.compile(r"(?<![A-Za-z])(?:[A-Za-z]\.){2,}")
_LEAD_ARTICLE_LOWER = re.compile(r"^(?:the|a|an)\s+")
_LEAD_ARTICLE_KEY = re.compile(r"^(?:the|a|an)\s+")
_TRAIL_POSSESSIVE = re.compile(r"['’]s?$")

# A "." followed by whitespace inside a surface is a sentence boundary the NER span
# leaked across ("World War II. B.") — unless the token before it is a title-style
# abbreviation ("St. Louis", "Mt. Everest") or a middle initial ("John F. Kennedy").
_SENT_BOUNDARY = re.compile(r"\.\s+")
_ABBREV_KEEP = frozenset(
    "st mt dr mr mrs ms prof jr sr no gen col lt sgt capt rev hon fr vs".split()
)
# Tokenizer clitics that the small models sometimes mis-tag as entities.
_CLITICS = frozenset({"n't", "'s", "'d", "'ll", "'re", "'ve", "'m"})
_ALPHA_RUN = re.compile(r"[a-z]{2}")
_ALNUM_MIX = re.compile(r"^(?=.*[a-z])(?=.*\d)")


def _truncate_cross_sentence(s: str) -> str:
    for m in _SENT_BOUNDARY.finditer(s):
        head = s[: m.start()].rstrip()
        last = head.rsplit(None, 1)[-1] if head else ""
        last = last.strip(_EDGE_TRIM).lower()
        if len(last) == 1 or last in _ABBREV_KEEP:
            continue  # a real abbreviation / initial, not a sentence boundary
        return head
    return s


def _clean_surface(text: str) -> str:
    s = _WS.sub(" ", text).strip()
    # De-dot acronyms BEFORE boundary truncation so "U.S. Army" survives intact.
    s = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _truncate_cross_sentence(s)
    s = s.strip(_EDGE_TRIM)
    s = _LEAD_ARTICLE_LOWER.sub("", s)
    s = _TRAIL_POSSESSIVE.sub("", s)
    return _WS.sub(" ", s).strip().strip(_EDGE_TRIM)


def _normalize_entity(surface: str) -> str:
    t = _LEAD_ARTICLE_KEY.sub("", _clean_surface(surface).lower())
    return _WS.sub(" ", t).strip()


def _acceptable_norm(norm: str) -> bool:
    """Shape filter: a real entity key contains an actual word.

    Rejects clitics ("n't", "'s"), bare punctuation, and single stray letters —
    anything without two consecutive letters, a letter+digit mix ("u2" / "3m" /
    "ww2"), or a plain number ("1942" — a legitimate DATE).
    """
    if len(norm) < 2 or norm in _CLITICS:
        return False
    return bool(_ALPHA_RUN.search(norm) or _ALNUM_MIX.match(norm) or norm.isdigit())


# --- Feed hygiene (heading/label-stub/list-join cleanup before NER) -----------
_LABEL_STUB_PREFIX = re.compile(r"^([^.!?:]{1,90}):\s+(\S.*)$", re.DOTALL)
_TERMINAL_CHARS = ".!?…\")]'’”"
_STRAY_MARKUP = re.compile(r"[*`#]+")
_LIST_SEP = re.compile(r"\s+[-–—]\s+")
# Leading list markers: bullets, "1." / "1)" and lettered "A." / "b)" enumerators.
_LEAD_MARKER = re.compile(r"^\s*(?:[-*•·▪◦–—]+|\d+[.)]|[A-Za-z][.)])\s+")
# Same enumerator shape at the START of an entity span ("B. Auschwitz") — stripped
# for non-PERSON labels only (a leading "J. " on a PERSON is a real initial).
_LEAD_ENUM = re.compile(r"^[A-Za-z][.)]\s+")


def _strip_label_stub(text: str) -> str:
    m = _LABEL_STUB_PREFIX.match(text.strip())
    if m and len(m.group(1).split()) <= 8:
        return m.group(2).strip()
    return text


def _clean_feed(text: str) -> str:
    t = _STRAY_MARKUP.sub("", text)
    t = _LEAD_MARKER.sub("", t)
    t = _strip_label_stub(t)
    t = _LIST_SEP.sub(". ", t)
    t = _LEAD_MARKER.sub("", t)
    return _WS.sub(" ", t).strip()


def _is_heading_sentence(text: str) -> bool:
    t = text.strip()
    if not t or t[-1] in _TERMINAL_CHARS:
        return False
    toks = re.findall(r"[A-Za-z][\w'’\-]*", t)
    if not toks or len(toks) > 12:
        return False
    return sum(1 for w in toks if w[0].isupper()) / len(toks) >= 0.7


def extract_entities(
    texts: list[str],
    *,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    model: str = _DEFAULT_MODEL,
    batch_size: int | None = None,
    n_process: int = 1,
) -> tuple[list[list[dict]], dict[str, int], dict[str, str], dict[str, str]]:
    """Run NER → (entity_lists, entity_df, surface_by_norm, label_by_norm).

    `entity_lists[i]` is the distinct entities in text *i* (`{surface, label, norm}`,
    de-duped by `norm`); `entity_df` is norm → number of texts mentioning it.

    ``batch_size=None`` auto-sizes: 64 for transformer pipelines (GPU batches pad to
    the longest member), 256 for CNN models. ``n_process`` fans a CPU CNN pipeline
    across processes; it is ignored for transformer or GPU-active pipelines
    (multiprocessing and GPU inference don't mix).
    """
    nlp = _nlp(model)
    if _GPU_ACTIVE:
        # thinc's active ops live in a CONTEXTVAR: a caller running each batch in
        # a fresh thread/context (asyncio.to_thread) would otherwise pipe CPU
        # tensors into MPS/CUDA weights ("Passed CPU tensor to MPS op"). Re-arm
        # the GPU ops for THIS context — a no-op when already active.
        _activate_gpu()
    allow = set(labels)
    feed = []
    for t in texts:
        prose = _clean_feed(t)
        feed.append("" if _is_heading_sentence(prose) else prose)

    is_trf = "transformer" in nlp.pipe_names
    bs = batch_size or (64 if is_trf else 256)
    pipe_kw: dict = {"batch_size": bs}
    if n_process > 1 and not is_trf and not _GPU_ACTIVE:
        pipe_kw["n_process"] = n_process

    # Length-sorted feed: transformer batches pad to their longest member, so
    # grouping similar lengths cuts padding waste. Original order is restored below
    # (harmless for CNN models, so it's unconditional).
    order = sorted(range(len(feed)), key=lambda i: len(feed[i]))
    entity_lists: list[list[dict]] = [[] for _ in feed]
    for i, doc in zip(order, nlp.pipe((feed[i] for i in order), **pipe_kw)):
        seen: dict[str, dict] = {}
        for ent in doc.ents:
            if ent.label_ not in allow or ":" in ent.text:
                continue
            raw = ent.text
            if ent.label_ != "PERSON":
                raw = _LEAD_ENUM.sub("", raw)
            surface = _clean_surface(raw)
            norm = _normalize_entity(surface)
            if not _acceptable_norm(norm):
                continue
            if norm not in seen:
                seen[norm] = {"surface": surface, "label": ent.label_, "norm": norm}
        entity_lists[i] = list(seen.values())

    # Corpus-wide maps accumulate in ORIGINAL text order so first-seen surface/label
    # choices stay deterministic regardless of the sorted pipe order above.
    entity_df: dict[str, int] = {}
    surface_by_norm: dict[str, str] = {}
    label_by_norm: dict[str, str] = {}
    for ents in entity_lists:
        for e in ents:
            norm = e["norm"]
            entity_df[norm] = entity_df.get(norm, 0) + 1
            surface_by_norm.setdefault(norm, e["surface"])
            label_by_norm.setdefault(norm, e["label"])
    return entity_lists, entity_df, surface_by_norm, label_by_norm


def sentence_entity_rarity(
    entity_lists: list[list[dict]], entity_df: dict[str, int], *, n_terms: int = 8
) -> list[dict]:
    """Per text: rarity of its rarest entity + top rare entities (score in [0,1])."""
    n = max(2, len(entity_lists))
    log_n = math.log(n)
    out: list[dict] = []
    for ents in entity_lists:
        ranked = sorted(((entity_df[e["norm"]], e) for e in ents), key=lambda x: x[0])
        if not ranked:
            out.append({"rarest_df": n, "score": 0.0, "terms": []})
            continue
        rarest_df = ranked[0][0]
        out.append({
            "rarest_df": rarest_df,
            "score": math.log(n / rarest_df) / log_n,
            "terms": [f"{e['surface']} ·{df}" for df, e in ranked[:n_terms]],
        })
    return out


def compute_named_entities(ws, run_id: str, *, model: str | None = None, n_terms: int = 8):
    """Compute per-unique entities + rarity for a run and write the artifacts.

    Reads unique texts from the run's body membership (canonical = embedding rows) and
    writes ``named_entities.jsonl`` + ``entity_df.json`` under the run's corpus dir.
    Uses ``ws.spacy_model`` unless ``model`` overrides it.
    """
    model = model or ws.spacy_model
    membership = [
        json.loads(line)
        for line in ws.body_membership_path(run_id).read_text().splitlines()
        if line
    ]
    texts = [m["text"] for m in membership]
    entity_lists, entity_df, surface_by_norm, label_by_norm = extract_entities(texts, model=model)
    rows = sentence_entity_rarity(entity_lists, entity_df, n_terms=n_terms)

    out_path = ws.named_entities_path(run_id)
    with out_path.open("w") as f:
        for uid, (ents, rarity) in enumerate(zip(entity_lists, rows)):
            f.write(json.dumps({"unique_id": uid, "entities": ents, **rarity}) + "\n")
    ws.entity_df_path(run_id).write_text(
        json.dumps({
            norm: {"df": df, "surface": surface_by_norm[norm], "label": label_by_norm[norm]}
            for norm, df in entity_df.items()
        })
    )
    return out_path
