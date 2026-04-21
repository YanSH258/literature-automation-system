from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

StanceType = Literal["supporting", "contrasting", "mentioning", "not_supported"]


@dataclass(frozen=True)
class LCRChunk:
    """单个文献片段，兼容 PaperQA2.Text 转换而来。"""
    chunk_id: str          # 格式: "{doc_id}#{seq:04d}"
    doc_id: str            # DOI 优先，无则 UUID
    text: str
    section: str
    page: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)
    rcs_score: float | None = None
    rcs_summary: str | None = None

    @classmethod
    def from_paperqa_text(cls, text, seq: int) -> "LCRChunk":
        """从 PaperQA2 的 Text 对象转换。"""
        doc = text.doc
        return cls(
            chunk_id=f"{doc.dockey}#{seq:04d}",
            doc_id=doc.dockey,
            text=text.text,
            section=getattr(text, "name", ""),
            page=getattr(text, "page", 0),
            char_start=0,
            char_end=len(text.text),
            metadata={
                "title": getattr(doc, "title", ""),
                "authors": getattr(doc, "authors", []),
                "year": getattr(doc, "year", None),
                "doi": getattr(doc, "doi", ""),
                "journal": getattr(doc, "journal", ""),
            },
        )


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
