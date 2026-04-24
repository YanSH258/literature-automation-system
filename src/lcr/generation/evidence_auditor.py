import json
import asyncio
import dataclasses
import litellm
import logging
from typing import List, Tuple
from lcr.types import LCRChunk
from lcr.config import settings as lcr_settings

logger = logging.getLogger(__name__)

EVIDENCE_AUDIT_PROMPT = """You are an evidence auditor for scientific literature.

Question: {question}

Below are {n} excerpts from research papers. For EACH excerpt, determine:
1. Does it contain DIRECT evidence relevant to the question? (true/false)
2. If yes, rate the strength of evidence (0-10)
3. Extract the single most relevant sentence (max 200 chars)

Return ONLY a JSON array in this exact format:
[
  {{"index": 1, "has_evidence": true, "score": 8, "key_sentence": "..."}},
  {{"index": 2, "has_evidence": false, "score": 0, "key_sentence": null}},
  ...
]

Excerpts:
{excerpts}
"""

async def _audit_one_batch(
    question: str, 
    batch: List[LCRChunk], 
    llm: str,
    api_key: str | None = None,
    api_base: str | None = None
) -> List[Tuple[LCRChunk, float, str | None]]:
    """审计单批次证据。"""
    _PREFIX = {"table": "[TABLE DATA]\n", "equation": "[EQUATION]\n"}
    excerpts_text = ""
    for i, chunk in enumerate(batch, start=1):
        ctype = chunk.metadata.get("chunk_type", "text")
        prefix = _PREFIX.get(ctype, "")
        excerpts_text += f"--- Excerpt {i} ---\n{prefix}{chunk.text}\n\n"
    
    prompt = EVIDENCE_AUDIT_PROMPT.format(
        question=question,
        n=len(batch),
        excerpts=excerpts_text
    )
    
    try:
        json_mode = any(x in llm.lower() for x in ["gpt-", "claude-", "deepseek"])
        
        response = await litellm.acompletion(
            model=llm,
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            api_base=api_base,
            response_format={ "type": "json_object" } if json_mode else None
        )
        content = response.choices[0].message.content
        
        from lcr.utils import extract_json_from_llm_output
        content = extract_json_from_llm_output(content)
            
        results = json.loads(content)
        if not isinstance(results, list):
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            else:
                raise ValueError("Model did not return a list")

        audited = []
        for res in results:
            idx = res.get("index", 1) - 1
            if 0 <= idx < len(batch):
                score = float(res.get("score", 0.0))
                has_ev = bool(res.get("has_evidence", False))
                key_sent = res.get("key_sentence")
                
                final_score = score if has_ev else 0.0
                audited.append((batch[idx], final_score, key_sent))
        
        return audited
        
    except Exception:
        logger.exception(f"Failed to audit batch of {len(batch)} chunks")
        return [(c, c.rcs_score or 0.0, None) for c in batch]

async def filter_evidence(
    question: str,
    chunks: List[LCRChunk],
    min_score: float = lcr_settings.EVIDENCE_MIN_SCORE,
    llm: str = lcr_settings.AUDIT_LLM,
    api_key: str | None = None,
    api_base: str | None = None,
    batch_size: int = lcr_settings.EVIDENCE_BATCH_SIZE,
) -> List[LCRChunk]:
    """过滤低质量证据，返回高分并补全了审计信息的 chunks。"""
    if not chunks:
        return []
        
    # 分批处理
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    # 使用信号量控制并发
    semaphore = asyncio.Semaphore(lcr_settings.EVIDENCE_SEMAPHORE)
    
    async def _audit_with_semaphore(b):
        async with semaphore:
            return await _audit_one_batch(question, b, llm, api_key, api_base)
            
    tasks = [_audit_with_semaphore(b) for b in batches]
    all_results = await asyncio.gather(*tasks)
    
    flat_results = []
    for batch_res in all_results:
        flat_results.extend(batch_res)
        
    # 更新 chunk 对象并筛选
    final_chunks = []
    for chunk, score, key_sent in flat_results:
        updated_chunk = dataclasses.replace(
            chunk, 
            rcs_score=score,
            rcs_summary=key_sent 
        )
        
        if score >= min_score:
            final_chunks.append(updated_chunk)
            
    final_chunks.sort(key=lambda x: x.rcs_score, reverse=True)
    
    if not final_chunks and flat_results:
        flat_results.sort(key=lambda x: x[1], reverse=True)
        best_chunk, best_score, best_key = flat_results[0]
        final_chunks.append(dataclasses.replace(best_chunk, rcs_score=best_score, rcs_summary=best_key))

    return final_chunks
