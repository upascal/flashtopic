"""Core data records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Doc:
    """One input document — the unit prevalence is ultimately reported over."""

    doc_id: str
    text: str
