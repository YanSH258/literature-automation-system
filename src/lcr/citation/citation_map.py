from lcr.types import LCRChunk, CitationEntry, CitationMap

def build_citation_map(
    chunks: list[LCRChunk],
    session_id: str,
    turn_id: int,
    previous_map: CitationMap | None = None
) -> CitationMap:
    citation_map = CitationMap(session_id=session_id, turn_id=turn_id)
    
    for i, chunk in enumerate(chunks, start=1):
        prev_idx = previous_map.reverse_lookup.get(chunk.chunk_id) if previous_map else None
        
        entry = CitationEntry(
            display_index=i,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            snippet=chunk.text[:300],
            metadata=chunk.metadata,
            rcs_score=chunk.rcs_score if chunk.rcs_score is not None else 0.0,
            from_previous_turn=prev_idx is not None,
            previous_display_index=prev_idx
        )
        
        citation_map.entries[i] = entry
        citation_map.reverse_lookup[chunk.chunk_id] = i
        
    return citation_map


def merge_citation_maps(maps: list[CitationMap]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for cmap in maps:
        for chunk_id, idx in cmap.reverse_lookup.items():
            if chunk_id not in merged:
                merged[chunk_id] = idx
    return merged
