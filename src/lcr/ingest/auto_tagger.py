import os
import json
import asyncio
import litellm
from pathlib import Path
from typing import List, Dict, Any
from lcr.ingest.zotero import ZoteroRecord
from lcr.config import settings as lcr_settings

CACHE_DIR = Path.home() / ".cache" / "lcr"
TAGS_CACHE_FILE = CACHE_DIR / "tags.json"

CHEM_TAG_TAXONOMY = {
    "计算方法": ["DFT", "molecular_dynamics", "Monte_Carlo", "QM_MM", "machine_learning_potential", "force_field", "semi-empirical"],
    "理论层次": ["ab_initio", "HF", "MP2", "CCSD", "B3LYP", "GGA", "TD-DFT", "hybrid_functional"],
    "研究对象": ["organic_molecule", "transition_metal_complex", "catalyst_surface", "biomolecule", "crystal", "nanoparticle"],
    "研究类型": ["reaction_mechanism", "thermodynamics", "kinetics", "electronic_structure", "property_prediction", "conformational_analysis"],
    "应用方向": ["homogeneous_catalysis", "heterogeneous_catalysis", "drug_design", "photochemistry", "energy_storage", "materials"]
}

PROMPT_TEMPLATE = """You are a computational chemistry expert. For each paper below, select 3-6 tags from the provided taxonomy that best describe the paper's methods and topics. Return a JSON array of objects ONLY.

Taxonomy:
{taxonomy}

Papers:
{papers_json}

Return format: [{{"doi": "...", "tags": ["...", "..."]}}, ...]"""

import logging

logger = logging.getLogger(__name__)

async def tag_batch(records: List[ZoteroRecord], semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:

    async with semaphore:
        papers_data = []
        for r in records:
            papers_data.append({
                "doi": r.doi,
                "title": r.title or "Untitled",
                "abstract": (r.abstract[:1500] + "...") if r.abstract and len(r.abstract) > 1500 else (r.abstract or "No abstract")
            })
        
        prompt = PROMPT_TEMPLATE.format(
            taxonomy=json.dumps(CHEM_TAG_TAXONOMY, ensure_ascii=False),
            papers_json=json.dumps(papers_data, ensure_ascii=False)
        )
        
        try:
            response = await litellm.acompletion(
                model="minimax/MiniMax-M2.5",
                api_key=lcr_settings.MINIMAX_API_KEY,
                api_base=lcr_settings.MINIMAX_API_BASE,
                messages=[{"role": "user", "content": prompt}],
            )
            
            content = response.choices[0].message.content
            from lcr.utils import extract_json_from_llm_output
            content = extract_json_from_llm_output(content)
            
            # 有时 LLM 会返回 {"results": [...]} 或直接是 [...]
            data = json.loads(content)
            if isinstance(data, dict):
                # 兼容不同格式
                for val in data.values():
                    if isinstance(val, list):
                        return val
            return data if isinstance(data, list) else []
            
        except Exception:
            logger.exception("Error tagging batch")
            return []

async def tag_all_records(records: List[ZoteroRecord], batch_size: int = lcr_settings.TAGGING_BATCH_SIZE) -> Dict[str, List[str]]:
    CACHE_DIR.mkdir(exist_ok=True)
    cache = {}
    if TAGS_CACHE_FILE.exists():
        try:
            cache = json.loads(TAGS_CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    
    to_process = [r for r in records if r.doi not in cache and r.abstract]
    if not to_process:
        return cache
        
    if not lcr_settings.MINIMAX_API_KEY:
        logger.info("LCR_MINIMAX_API_KEY not configured, skipping auto-tagging.")
        return cache

    logger.info(f"Tagging {len(to_process)} new papers using MiniMax...")
    
    semaphore = asyncio.Semaphore(lcr_settings.TAGGING_SEMAPHORE)
    batches = [to_process[i:i + batch_size] for i in range(0, len(to_process), batch_size)]
    
    tasks = [tag_batch(b, semaphore) for b in batches]
    results = await asyncio.gather(*tasks)
    
    new_tags_count = 0
    for batch_res in results:
        for item in batch_res:
            if "doi" in item and "tags" in item:
                cache[item["doi"]] = item["tags"]
                new_tags_count += 1
    
    _tmp = TAGS_CACHE_FILE.with_suffix(".tmp")
    _tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    _tmp.replace(TAGS_CACHE_FILE)
    logger.info(f"Tagging complete. Added {new_tags_count} new tags. Total cached: {len(cache)}")
    return cache

def tag_all_records_sync(records: List[ZoteroRecord]) -> Dict[str, List[str]]:
    return asyncio.run(tag_all_records(records))
