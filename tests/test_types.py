import pytest
from dataclasses import replace
from lcr.types import LCRChunk, CitationEntry, CitationMap

def test_citation_map_lookup_chunk_id():
    entry1 = CitationEntry(
        display_index=1,
        chunk_id="doc1#0001",
        doc_id="doc1",
        snippet="snippet1",
        metadata={},
        rcs_score=9.0
    )
    cmap = CitationMap(
        session_id="session1",
        turn_id=1,
        entries={1: entry1},
        reverse_lookup={"doc1#0001": 1}
    )
    
    # 对存在的 index 行为正确
    assert cmap.lookup_chunk_id(1) == "doc1#0001"
    # 对不存在的 index 行为正确
    assert cmap.lookup_chunk_id(2) is None

def test_citation_map_lookup_display_index():
    entry1 = CitationEntry(
        display_index=1,
        chunk_id="doc1#0001",
        doc_id="doc1",
        snippet="snippet1",
        metadata={},
        rcs_score=9.0
    )
    cmap = CitationMap(
        session_id="session1",
        turn_id=1,
        entries={1: entry1},
        reverse_lookup={"doc1#0001": 1}
    )
    
    # 对存在的 chunk_id 行为正确
    assert cmap.lookup_display_index("doc1#0001") == 1
    # 对不存在的 chunk_id 行为正确
    assert cmap.lookup_display_index("doc2#0001") is None

def test_lcr_chunk_frozen():
    chunk = LCRChunk(
        chunk_id="doc1#0001",
        doc_id="doc1",
        text="text1",
        section="section1",
        page=1,
        char_start=0,
        char_end=5,
        rcs_score=9.0
    )
    
    # 验证不可变（修改属性应抛出异常）
    with pytest.raises(AttributeError):
        chunk.text = "new text" # type: ignore
