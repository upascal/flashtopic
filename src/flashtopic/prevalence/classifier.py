"""Typed-question document classifier (Jev-style) behind a protocol, with a disk cache.

A *decision model* such as TypeSafe's Jev takes one piece of state (here: a document) and a
set of typed questions, and returns a probability per answer in a single forward pass:

* ``noul``   — yes/no                → ``p_yes``
* ``choice`` — one of N options      → a distribution over the options
* ``score``  — an ordered scale      → a distribution over the levels

Its answers are **features**, not verdicts: the prevalence pillar feeds them (plus the
deterministic per-document signals) to a small cross-fitted regression that produces the
whole-corpus surrogate ``f`` (see ``surrogate.py``). The gold judge stays ``judge.py``.

The model sits behind :class:`DocClassifier` so the OpenRouter Decisions endpoint, an
open-source replica, and a :class:`MockClassifier` (tests, the pilot self-check) are
interchangeable. :class:`JevClassifier` takes an injectable ``transport`` so the request
building and response parsing are testable without network.

Answers are cached on disk per ``(classifier, document, question content)`` — mirroring
``judge.py`` — so re-runs and question-set edits only pay for what changed. Errors are
never cached.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, Sequence, runtime_checkable

from flashtopic.llm import _backoff, _config, _get_semaphore, configure_concurrency

QUESTION_KINDS = ("noul", "choice", "score")
_DEFAULT_CACHE_DIR = Path.cwd() / ".flashtopic_cache" / "classifier"
_DOC_CHAR_CAP = 16000  # same cap as the judge — one full document
_DEFAULT_MODEL = "typesafe/jev-1.13"
_DECISIONS_PATH = "/api/alpha/decisions"


# --------------------------------------------------------------------------- #
# Wire-agnostic question / answer records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Question:
    """One typed question. ``options`` are the choice options or the ordered score levels."""

    qid: str
    kind: str  # noul | choice | score
    text: str  # the instructions
    options: tuple[str, ...] = ()
    criteria: str | None = None  # operational definition / examples (the codebook)
    criteria_false: str | None = None  # noul only: what counts as "no" (e.g. passing mentions)

    def __post_init__(self):
        if self.kind not in QUESTION_KINDS:
            raise ValueError(f"{self.qid}: unknown question kind {self.kind!r}")
        if self.kind == "choice" and not (2 <= len(self.options) <= 255):
            raise ValueError(f"{self.qid}: choice needs 2..255 options, got {len(self.options)}")
        if self.kind == "score" and not (2 <= len(self.options) <= 10):
            raise ValueError(f"{self.qid}: score needs 2..10 levels, got {len(self.options)}")
        if self.kind == "noul" and self.options:
            raise ValueError(f"{self.qid}: noul takes no options")
        if self.criteria_false and self.kind != "noul":
            raise ValueError(f"{self.qid}: criteria_false is noul-only")

    @property
    def content_hash(self) -> str:
        """Hash of what the model is asked (not the qid) — the cache key."""
        h = hashlib.sha256()
        h.update(self.kind.encode())
        h.update(b"\x1f" + self.text.encode())
        h.update(b"\x1f" + "\x1e".join(self.options).encode())
        h.update(b"\x1f" + (self.criteria or "").encode())
        if self.criteria_false:
            h.update(b"\x1f" + self.criteria_false.encode())
        return h.hexdigest()[:16]

    def to_wire(self) -> dict:
        """The Decisions-API question body ``{type, instructions, criteria}``.

        ``criteria`` is required and typed by kind (verified live against
        ``typesafe/jev-1.13``): noul → ``{"true": ..., "false": ...}``; score → the ordered
        level labels (answers come back keyed by level index); choice → ``{option: description}``
        (answers keyed by option). Score/choice have no slot for free-text criteria, so the
        codebook is appended to their instructions.
        """
        text = self.text
        if self.kind == "noul":
            criteria: Any = {
                "true": self.criteria or "Yes.",
                "false": self.criteria_false or (
                    "The document does not meet the criteria above." if self.criteria else "No."),
            }
        else:
            if self.criteria:
                text = f"{text}\n\nCriteria:\n{self.criteria}"
            if self.kind == "score":
                criteria = list(self.options)
            else:
                criteria = {o: o for o in self.options}
        return {"type": self.kind, "instructions": text, "criteria": criteria}


@dataclass
class Answer:
    qid: str
    p_yes: float | None = None  # noul
    dist: tuple[float, ...] | None = None  # choice / score, aligned with Question.options
    confidence: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and (self.p_yes is not None or self.dist is not None)

    def to_dict(self) -> dict:
        return {"qid": self.qid, "p_yes": self.p_yes,
                "dist": list(self.dist) if self.dist is not None else None,
                "confidence": self.confidence, "error": self.error}

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        return cls(d["qid"], d.get("p_yes"), tuple(d["dist"]) if d.get("dist") is not None else None,
                   d.get("confidence"), d.get("error"))


@runtime_checkable
class DocClassifier(Protocol):
    name: str  # part of the cache key — distinct models never share answers

    async def classify(self, doc_text: str, questions: Sequence[Question]) -> list[Answer]: ...


# --------------------------------------------------------------------------- #
# Mock (tests, pilot self-check)
# --------------------------------------------------------------------------- #
class MockClassifier:
    """Deterministic stand-in: ``rule(doc_text, question) -> Answer``. Counts calls."""

    name = "mock"

    def __init__(self, rule: Callable[[str, Question], Answer] | None = None):
        self.rule = rule or keyword_rule
        self.calls = 0
        self.questions_seen = 0

    async def classify(self, doc_text: str, questions: Sequence[Question]) -> list[Answer]:
        self.calls += 1
        self.questions_seen += len(questions)
        return [self.rule(doc_text, q) for q in questions]


def keyword_rule(doc_text: str, q: Question) -> Answer:
    """Default mock: overlap between the question's words and the document → probability.

    Crude on purpose; it only has to be deterministic and vaguely informative.
    """
    words = {w for w in re.findall(r"[a-z]{4,}", (q.text + " " + (q.criteria or "")).lower())}
    doc = doc_text.lower()
    hits = sum(1 for w in words if w in doc)
    p = 1.0 / (1.0 + math.exp(-(hits - 2.0)))
    if q.kind == "noul":
        return Answer(q.qid, p_yes=p)
    n = len(q.options)
    if q.kind == "score":
        peak = min(n - 1, int(round(p * (n - 1))))
        dist = [0.02] * n
        dist[peak] += 1.0
    else:  # choice: put mass on the option with the most overlap, remainder on the last
        scores = [sum(1 for w in re.findall(r"[a-z]{4,}", o.lower()) if w in doc) for o in q.options]
        dist = [0.02 + s for s in scores]
    z = sum(dist)
    dist = tuple(x / z for x in dist)
    return Answer(q.qid, dist=dist, confidence=max(dist))


# --------------------------------------------------------------------------- #
# Jev over the OpenRouter Decisions endpoint
# --------------------------------------------------------------------------- #
Transport = Callable[[dict], Awaitable[dict]]


def decisions_url(base_url: str | None = None) -> str:
    """``https://openrouter.ai/api/v1`` → ``https://openrouter.ai/api/alpha/decisions``."""
    base = (base_url or _config["base_url"] or "https://openrouter.ai/api/v1").rstrip("/")
    base = re.sub(r"/api/v\d+$", "", base)
    return base + _DECISIONS_PATH


def build_request(model: str, doc_text: str, questions: Sequence[Question],
                  *, doc_char_cap: int = _DOC_CHAR_CAP) -> dict:
    return {
        "model": model,
        "state": doc_text[:doc_char_cap],
        "questions": {q.qid: q.to_wire() for q in questions},
    }


def _find_answers(resp: dict, qids: Sequence[str]) -> dict:
    """Locate the per-question answers in a response, tolerating wrapper keys."""
    if not isinstance(resp, dict):
        return {}
    if any(q in resp for q in qids):
        return resp
    for key in ("answers", "decisions", "results", "data", "output"):
        inner = resp.get(key)
        if isinstance(inner, dict) and any(q in inner for q in qids):
            return inner
        if isinstance(inner, dict):
            found = _find_answers(inner, qids)
            if found:
                return found
    return {}


def _as_dist(probs: Any, options: Sequence[str]) -> tuple[float, ...] | None:
    """Align a ``{option: p}`` map (or a list) to the question's option order."""
    if isinstance(probs, dict):
        vals = []
        for i, o in enumerate(options):
            v = probs.get(o, probs.get(str(i)))
            if v is None:
                return None
            vals.append(float(v))
    elif isinstance(probs, (list, tuple)) and len(probs) == len(options):
        vals = [float(v) for v in probs]
    else:
        return None
    z = sum(vals)
    return tuple(v / z for v in vals) if z > 0 else None


def parse_answers(questions: Sequence[Question], resp: dict) -> list[Answer]:
    """Turn a Decisions response into :class:`Answer` rows (per-question errors, no raise)."""
    found = _find_answers(resp, [q.qid for q in questions])
    out: list[Answer] = []
    for q in questions:
        a = found.get(q.qid)
        if not isinstance(a, dict):
            out.append(Answer(q.qid, error="missing"))
            continue
        try:
            conf = a.get("confidence")
            conf = float(conf) if conf is not None else None
            if q.kind == "noul":
                p = a.get("noul", a.get("probability", a.get("p_yes")))
                if p is None:
                    raise ValueError("no noul probability")
                out.append(Answer(q.qid, p_yes=min(1.0, max(0.0, float(p))), confidence=conf))
            else:
                dist = _as_dist(a.get("probabilities", a.get("distribution")), q.options)
                if dist is None:
                    raise ValueError("no aligned distribution")
                out.append(Answer(q.qid, dist=dist, confidence=conf))
        except (TypeError, ValueError) as e:
            out.append(Answer(q.qid, error=f"parse: {e}"))
    return out


async def _http_transport(payload: dict, *, url: str, api_key: str, timeout: float) -> dict:
    import httpx  # a dependency of `openai`, so always present

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url, json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.json()


class JevClassifier:
    """Jev via the OpenRouter Decisions endpoint (or any server speaking the same format).

    ``transport(payload) -> response_json`` is injectable; the default posts to
    :func:`decisions_url` with the same key/base URL ``flashtopic.llm`` uses. Questions are
    sent in chunks of ``max_questions_per_request`` (the endpoint's cap is undocumented;
    32 is conservative) under ``llm``'s concurrency semaphore, with its backoff on
    transient failures.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        transport: Transport | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_questions_per_request: int = 32,
        max_attempts: int = 4,
        timeout: float = 60.0,
        doc_char_cap: int = _DOC_CHAR_CAP,
    ):
        self.model = model
        self.name = f"jev:{model}"
        self.max_q = max(1, max_questions_per_request)
        self.max_attempts = max_attempts
        self.doc_char_cap = doc_char_cap
        if transport is None:
            url = decisions_url(base_url)
            key = api_key or _config["api_key"]

            async def transport(payload: dict) -> dict:
                if not key:
                    raise RuntimeError("OPENROUTER_API_KEY not set (see flashtopic.llm)")
                return await _http_transport(payload, url=url, api_key=key, timeout=timeout)

        self.transport = transport

    async def _call(self, doc_text: str, questions: Sequence[Question]) -> list[Answer]:
        payload = build_request(self.model, doc_text, questions, doc_char_cap=self.doc_char_cap)
        sem = _get_semaphore()
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                async with sem:
                    resp = await self.transport(payload)
                return parse_answers(questions, resp)
            except Exception as e:  # network / HTTP / JSON — retry with backoff
                last = e
                if attempt < self.max_attempts - 1:
                    await asyncio.sleep(_backoff(attempt))
        return [Answer(q.qid, error=f"call failed: {str(last)[:200]}") for q in questions]

    async def classify(self, doc_text: str, questions: Sequence[Question]) -> list[Answer]:
        chunks = [questions[i:i + self.max_q] for i in range(0, len(questions), self.max_q)]
        results = await asyncio.gather(*(self._call(doc_text, c) for c in chunks))
        return [a for chunk in results for a in chunk]


# --------------------------------------------------------------------------- #
# Cache + batch driver
# --------------------------------------------------------------------------- #
def _cache_path(clf_name: str, doc_id: int, cache_dir: Path) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", clf_name)
    return Path(cache_dir) / safe / f"{doc_id}.json"


def _load_cache(path: Path) -> dict[str, dict]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


async def classify_doc(
    clf: DocClassifier, doc_id: int, doc_text: str, questions: Sequence[Question],
    *, cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> list[Answer]:
    """Answer every question for one document, calling the model only for uncached ones."""
    path = _cache_path(clf.name, doc_id, cache_dir)
    cached = _load_cache(path)
    by_hash = {q.content_hash: q for q in questions}
    missing = [q for h, q in by_hash.items() if h not in cached]
    fresh: dict[str, Answer] = {}
    if missing:
        answers = await clf.classify(doc_text, missing)
        for q, a in zip(missing, answers):
            a.qid = q.qid
            fresh[q.content_hash] = a
        good = {h: a.to_dict() for h, a in fresh.items() if a.ok}
        if good:
            cached.update(good)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cached))
    out: list[Answer] = []
    for q in questions:
        h = q.content_hash
        a = Answer.from_dict(fresh[h].to_dict() if h in fresh else cached[h])
        a.qid = q.qid  # keyed by content; two qids with identical content get their own row
        out.append(a)
    return out


async def classify_docs(
    clf: DocClassifier,
    doc_texts: dict[int, str],
    questions: Sequence[Question],
    *,
    concurrency: int = 8,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[int, list[Answer]]:
    """Classify every document (same question set), bounded concurrency, cached."""
    configure_concurrency(concurrency)
    gate = asyncio.Semaphore(concurrency)
    done = 0
    total = len(doc_texts)

    async def _one(d: int, text: str):
        nonlocal done
        async with gate:
            ans = await classify_doc(clf, d, text, questions, cache_dir=cache_dir)
        done += 1
        if on_progress:
            on_progress(done, total)
        return d, ans

    pairs = await asyncio.gather(*(_one(int(d), t) for d, t in doc_texts.items()))
    return dict(pairs)


def run_classify(clf, doc_texts, questions, **kw) -> dict[int, list[Answer]]:
    """Synchronous wrapper around :func:`classify_docs`."""
    return asyncio.run(classify_docs(clf, doc_texts, questions, **kw))
