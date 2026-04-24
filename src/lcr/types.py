from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

StanceType = Literal["supporting", "contrasting", "mentioning", "not_supported"]


@dataclass(frozen=True)
class LCRChunk:
    """单个文献片段。chunk_id 格式: "{doc_id}#{seq:04d}"。"""
    chunk_id: str
    doc_id: str            # DOI
    text: str
    section: str
    page: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)
    rcs_score: float | None = None
    rcs_summary: str | None = None


@dataclass
class CitationEntry:
    display_index: int
    chunk_id: str
    doc_id: str
    snippet: str
    metadata: dict
    rcs_score: float
    from_previous_turn: bool = False
    previous_display_index: int | None = None


@dataclass
class CitationMap:
    session_id: str
    turn_id: int
    entries: dict[int, CitationEntry] = field(default_factory=dict)
    reverse_lookup: dict[str, int] = field(default_factory=dict)

    def lookup_chunk_id(self, idx: int) -> str | None:
        entry = self.entries.get(idx)
        return entry.chunk_id if entry is not None else None

    def lookup_display_index(self, chunk_id: str) -> int | None:
        return self.reverse_lookup.get(chunk_id)


@dataclass
class SentenceValidation:
    sentence: str
    citations: list[dict]  # [{display_index, stance, confidence, nli_entails}]


@dataclass
class LCRResponse:
    answer_text: str
    citations: list[CitationEntry]
    sentence_level_validation: list[SentenceValidation]
    overall_confidence: float
    co_citation_suggestions: dict[int, list[int]] = field(default_factory=dict)
    insufficient_evidence: bool = False
