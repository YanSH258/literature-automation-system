import os
import json
import asyncio
import litellm
from pathlib import Path
from typing import List, Dict, Any
from lcr.ingest.zotero import ZoteroRecord

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

async def tag_batch(batch: List[ZoteroRecord], semaphore: asyncio.Semaphore) -> List[Dict[str, Any]]:
    async with semaphore:
        papers_data = []
        for r in batch:
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
            # 使用 litellm 路由到 MiniMax
            # 注意：MiniMax 的 model 名称通常需要带前缀或直接指定提供商
            # 这里按照 litellm 标准：openai/MiniMax-Text-01 (需配置 api_base)
            response = await litellm.acompletion(
                model="minimax/MiniMax-M2.5",
                api_key=os.environ.get("MINIMAX_API_KEY"),
                api_base=os.environ.get("MINIMAX_API_BASE", "https://api.minimax.io/v1"),
                messages=[{"role": "user", "content": prompt}],
            )
            
            content = response.choices[0].message.content
            # 处理可能的 markdown 代码块
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # 有时 LLM 会返回 {"results": [...]} 或直接是 [...]
            data = json.loads(content)
            if isinstance(data, dict):
                # 兼容不同格式
                for val in data.values():
                    if isinstance(val, list):
                        return val
            return data if isinstance(data, list) else []
            
        except Exception as e:
            print(f"Error tagging batch: {e}")
            return []

async def tag_all_records(records: List[ZoteroRecord], batch_size: int = 20) -> Dict[str, List[str]]:
    # 1. 加载缓存
    CACHE_DIR.mkdir(exist_ok=True)
    cache = {}
    if TAGS_CACHE_FILE.exists():
        try:
            cache = json.loads(TAGS_CACHE_FILE.read_text())
        except:
            pass
    
    # 2. 筛选需要打标的（有摘要且不在缓存中）
    to_process = [r for r in records if r.doi not in cache and r.abstract]
    if not to_process:
        return cache

    print(f"Tagging {len(to_process)} new papers using MiniMax...")
    
    # 3. 分批异步处理
    semaphore = asyncio.Semaphore(5) # 降低并发以免触发 MiniMax 频率限制
    batches = [to_process[i:i + batch_size] for i in range(0, len(to_process), batch_size)]
    
    tasks = [tag_batch(b, semaphore) for b in batches]
    results = await asyncio.gather(*tasks)
    
    # 4. 更新缓存
    new_tags_count = 0
    for batch_res in results:
        for item in batch_res:
            if "doi" in item and "tags" in item:
                cache[item["doi"]] = item["tags"]
                new_tags_count += 1
    
    TAGS_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    print(f"Tagging complete. Added {new_tags_count} new tags. Total cached: {len(cache)}")
    return cache

def tag_all_records_sync(records: List[ZoteroRecord]) -> Dict[str, List[str]]:
    return asyncio.run(tag_all_records(records))
