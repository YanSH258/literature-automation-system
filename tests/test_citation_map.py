from lcr.types import LCRChunk
from lcr.citation.citation_map import build_citation_map, merge_citation_maps

def test_build_citation_map_single_turn():
    chunks = [
        LCRChunk(chunk_id="doc1#0001", doc_id="doc1", text="Text 1" * 100, section="S1", page=1, char_start=0, char_end=600),
        LCRChunk(chunk_id="doc1#0002", doc_id="doc1", text="Text 2", section="S1", page=1, char_start=0, char_end=6),
        LCRChunk(chunk_id="doc2#0001", doc_id="doc2", text="Text 3", section="S2", page=1, char_start=0, char_end=6),
    ]
    
    cmap = build_citation_map(chunks, "sess1", 0)
    
    assert len(cmap.entries) == 3
    assert cmap.entries[1].display_index == 1
    assert cmap.entries[1].chunk_id == "doc1#0001"
    assert len(cmap.entries[1].snippet) == 300
    assert cmap.reverse_lookup["doc1#0001"] == 1
    assert cmap.reverse_lookup["doc2#0001"] == 3
    assert cmap.entries[1].from_previous_turn is False
    assert cmap.entries[1].previous_display_index is None

def test_build_citation_map_multi_turn():
    chunks1 = [
        LCRChunk(chunk_id="doc1#0001", doc_id="doc1", text="Text 1", section="S1", page=1, char_start=0, char_end=6),
    ]
    cmap1 = build_citation_map(chunks1, "sess1", 0)
    
    chunks2 = [
        LCRChunk(chunk_id="doc2#0001", doc_id="doc2", text="Text 2", section="S2", page=1, char_start=0, char_end=6),
        LCRChunk(chunk_id="doc1#0001", doc_id="doc1", text="Text 1", section="S1", page=1, char_start=0, char_end=6),
    ]
    cmap2 = build_citation_map(chunks2, "sess1", 1, previous_map=cmap1)
    
    assert cmap2.entries[1].chunk_id == "doc2#0001"
    assert cmap2.entries[1].from_previous_turn is False
    
    assert cmap2.entries[2].chunk_id == "doc1#0001"
    assert cmap2.entries[2].from_previous_turn is True
    assert cmap2.entries[2].previous_display_index == 1

def test_merge_citation_maps():
    chunks1 = [
        LCRChunk(chunk_id="doc1#0001", doc_id="doc1", text="T1", section="S1", page=1, char_start=0, char_end=2),
        LCRChunk(chunk_id="doc2#0001", doc_id="doc2", text="T2", section="S2", page=1, char_start=0, char_end=2),
    ]
    cmap1 = build_citation_map(chunks1, "sess1", 0)
    
    chunks2 = [
        LCRChunk(chunk_id="doc2#0001", doc_id="doc2", text="T2", section="S2", page=1, char_start=0, char_end=2),
        LCRChunk(chunk_id="doc3#0001", doc_id="doc3", text="T3", section="S3", page=1, char_start=0, char_end=2),
    ]
    cmap2 = build_citation_map(chunks2, "sess1", 1, previous_map=cmap1)
    
    merged = merge_citation_maps([cmap1, cmap2])
    
    assert merged["doc1#0001"] == 1
    assert merged["doc2#0001"] == 2  # First seen in turn 0
    assert merged["doc3#0001"] == 2  # First seen in turn 1
