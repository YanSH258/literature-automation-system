from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import litellm
import json
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr
from lcr.config import settings as lcr_settings
from pathlib import Path
from typing import List, Optional

from lcr.ingest.zotero import ZoteroIngestor, ZoteroRecord
from lcr.ingest.auto_tagger import tag_all_records, tag_all_records_sync
from lcr.ingest.chroma_ingestor import get_index_stats
from lcr.retrieval.chroma_retriever import retrieve_chunks

import uuid
import dataclasses
import logging

logger = logging.getLogger(__name__)

from lcr.types import LCRChunk
from lcr.citation.citation_map import build_citation_map
from lcr.validation.structural import structural_check
from lcr.generation.evidence_auditor import filter_evidence

app = FastAPI(title="LCR · Literature Citation-RAG")

# 全局索引状态
INDEX_STATUS = {
    "is_running": False,
    "current_step": "idle",
    "processed": 0,
    "total": 0,
    "error": None
}

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class LLMSettings(BaseModel):
    llm: str = lcr_settings.DEFAULT_LLM
    api_key: SecretStr = SecretStr("")
    api_base: str = ""
    evidence_k: int = lcr_settings.EVIDENCE_K
    chunks_per_paper: int = lcr_settings.CHUNKS_PER_PAPER

class AskRequest(BaseModel):
    question: str
    dois: Optional[List[str]] = None
    collection_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None
    llm_settings: Optional[LLMSettings] = None
    conversation_history: Optional[List[dict]] = None  # [{role: "user"/"assistant", content: "..."}]

async def query_chromadb_workflow(
    question: str,
    llm_settings: Optional[LLMSettings],
    dois: Optional[List[str]],
    collection_filter: Optional[str],
    tag_filter: Optional[List[str]],
    conversation_history: Optional[List[dict]]
) -> dict:
    logger.debug("Using ChromaDB retrieval for question: %s", question)
    # 增加召回数量以供审计筛选
    raw_k = max(llm_settings.evidence_k if llm_settings else 20, 30)
    
    chunks, stage_info = await asyncio.to_thread(
        retrieve_chunks,
        question, 
        n_results=raw_k,
        doi_filter=dois,
        collection_filter=collection_filter,
        tag_filter=tag_filter,
        chunks_per_paper=llm_settings.chunks_per_paper if llm_settings else 4
    )
    
    if not chunks:
        return {
            "answer": "未找到与该问题相关的文献片段。请确认所选分类下的文献已完成索引，或尝试更换问题。",
            "citations": [],
            "structural_check": {"passed": True, "issues": [], "uncited_sentences": [], "cited_indices": []},
            "retrieval_source": "chromadb",
            "retrieval_info": stage_info
        }

    # --- 批处理证据审计与筛选 ---
    # 使用较便宜的模型进行审计
    audit_llm = lcr_settings.AUDIT_LLM
    chunks = await filter_evidence(
        question=question,
        chunks=chunks,
        llm=audit_llm,
        api_key=None,
        api_base=None,
        min_score=lcr_settings.EVIDENCE_MIN_SCORE
    )
    
    # 截断到最终展示数量
    display_k = llm_settings.evidence_k if llm_settings else 10
    chunks = chunks[:display_k]

    # 构造 Prompt 让 LLM 直接基于 chunks 回答
    target_llm = llm_settings.llm if llm_settings else lcr_settings.DEFAULT_LLM

    context_parts = []
    for i, c in enumerate(chunks, start=1):
        meta = c.metadata
        authors = meta.get('creators', '')
        pub = meta.get('publication', '')
        year = meta.get('year', '')
        header_parts = [f"doi: {meta.get('doi')}"]
        if authors: header_parts.append(f"authors: {authors}")
        if pub: header_parts.append(f"journal: {pub}")
        if year: header_parts.append(f"year: {year[:4]}")
        
        text_to_show = c.text
        if c.rcs_summary:
            text_to_show = f"KEY EVIDENCE: {c.rcs_summary}\nCONTEXT: {c.text}"
            
        context_parts.append(f"[{i}] ({', '.join(header_parts)})\n{text_to_show}")
    
    context_str = "\n\n".join(context_parts)

    system_msg = lcr_settings.SYSTEM_PROMPT
    user_msg = f"""Context (scientific paper chunks):\n{context_str}\n\nQuestion: {question}\n\nAnswer (cite every claim with [n], no uncited summary at the end):"""

    messages = [{"role": "system", "content": system_msg}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_msg})

    try:
        response = await litellm.acompletion(
            model=target_llm,
            api_key=llm_settings.api_key.get_secret_value() if llm_settings and llm_settings.api_key.get_secret_value() else None,
            api_base=llm_settings.api_base if llm_settings and llm_settings.api_base else None,
            messages=messages
        )
    except Exception:
        raise HTTPException(status_code=502, detail="LLM service error")
    answer_text = response.choices[0].message.content

    session_id = str(uuid.uuid4())
    turn_id = len(conversation_history) // 2 if conversation_history else 0
    cmap = build_citation_map(chunks, session_id=session_id, turn_id=turn_id)
    structural = structural_check(answer_text, max_index=len(chunks))

    citations_out = []
    for idx, entry in cmap.entries.items():
        citations_out.append({
            "display_index": entry.display_index,
            "chunk_id": entry.chunk_id,
            "doc_id": entry.doc_id,
            "snippet": entry.snippet,
            "rcs_score": entry.rcs_score,
            "metadata": entry.metadata,
        })

    return {
        "answer": answer_text,
        "citations": citations_out,
        "structural_check": {
            "passed": structural.passed,
            "issues": structural.issues,
            "uncited_sentences": structural.uncited_sentences,
            "cited_indices": structural.cited_indices,
        },
        "retrieval_source": "chromadb",
        "retrieval_info": stage_info
    }

async def query_paperqa(question: str,
                        dois: Optional[List[str]] = None,
                        collection_filter: Optional[str] = None,
                        tag_filter: Optional[List[str]] = None,
                        llm_settings: Optional[LLMSettings] = None,
                        conversation_history: Optional[List[dict]] = None) -> dict:
    if not (lcr_settings.CHROMA_DIR / "chroma.sqlite3").exists():
        raise HTTPException(status_code=503, detail="ChromaDB index not built yet. Run python run_index.py first.")
    return await query_chromadb_workflow(
        question, llm_settings, dois, collection_filter, tag_filter, conversation_history
    )

@app.post("/admin/test-llm")
async def test_llm_endpoint(settings: LLMSettings):
    """测试 LLM 配置是否能正常连接。"""
    try:
        # 构造极简测试 Prompt
        messages = [{"role": "user", "content": "ping"}]
        
        # 显式传递参数，不污染全局环境变量
        target_model = settings.llm
        api_key = settings.api_key if settings.api_key else None
        api_base = settings.api_base if settings.api_base else None
        
        # 发起请求，设置较短的超时和极小的 token 消耗
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=target_model,
                messages=messages,
                api_key=api_key.get_secret_value() if api_key else None,
                api_base=api_base,
                max_tokens=5
            ),
            timeout=10.0
        )
        return {"status": "success", "content": response.choices[0].message.content}
    except Exception:
        logger.exception("LLM test connection failed")
        return {"status": "error", "message": "Connection failed"}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = static_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return index_path.read_text(encoding="utf-8")

@app.get("/prompts")
async def list_prompts():
    """列出所有可用的 prompt 模板。"""
    templates = []
    if not lcr_settings.PROMPTS_DIR.exists():
        lcr_settings.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        
    for path in sorted(lcr_settings.PROMPTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            templates.append({
                "filename": path.name,
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
                "template": data.get("template", ""),
            })
        except Exception:
            pass
    return {"templates": templates}

@app.post("/prompts")
async def save_prompt(payload: dict):
    """保存新的 prompt 模板。"""
    name = payload.get("name", "").strip()
    template = payload.get("template", "").strip()
    if not name or not template:
        raise HTTPException(status_code=400, detail="name and template are required")
    # 校验变量：只允许 {text} {question} {citation} {summary_length}
    import string
    allowed = {"text", "question", "citation", "summary_length"}
    try:
        fields = {f[1] for f in string.Formatter().parse(template) if f[1]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid template syntax: {e}")
    invalid = fields - allowed
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid variables: {invalid}. Allowed: {allowed}"
        )
    filename = name.lower().replace(" ", "_").replace("/", "_") + ".json"
    path = lcr_settings.PROMPTS_DIR / filename
    path.write_text(json.dumps({
        "name": name,
        "description": payload.get("description", ""),
        "template": template,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"filename": filename, "name": name}

@app.get("/zotero/collections")
async def get_zotero_collections():
    try:
        ingestor = ZoteroIngestor()
        tree = ingestor.load_collections_tree()
        return tree
    except Exception:
        logger.exception("Failed to load Zotero collections")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/analyse")
async def analyse_endpoint(payload: dict):
    return {
        "status": "not_implemented", 
        "message": "Coming soon: Structured batch analysis for selected papers.",
        "dois": payload.get("dois", [])
    }

@app.get("/admin/index-status")
async def index_status():
    """返回索引状态。"""
    stats = get_index_stats() if lcr_settings.CHROMA_DIR.exists() else {"total_chunks": 0, "total_docs": 0}
    return {**INDEX_STATUS, "stats": stats}

@app.post("/admin/build-index")
async def build_index_endpoint():
    """后台触发全量索引。"""
    if INDEX_STATUS["is_running"]:
        return {"status": "already_running"}
    
    async def run_indexing():
        global INDEX_STATUS
        try:
            INDEX_STATUS["is_running"] = True
            INDEX_STATUS["error"] = None
            
            INDEX_STATUS["current_step"] = "loading_zotero"
            ingestor = ZoteroIngestor()
            records = ingestor.load_records()
            INDEX_STATUS["total"] = len(records)
            
            INDEX_STATUS["current_step"] = "auto_tagging"
            tags_map = await tag_all_records(records)
            
            INDEX_STATUS["current_step"] = "chroma_indexing"
            loop = asyncio.get_running_loop()
            
            def run_with_progress():
                from lcr.ingest.chroma_ingestor import ChromaIngestor
                ingestor = ChromaIngestor()
                for i, rec in enumerate(records):
                    tags = tags_map.get(rec.doi, [])
                    if rec.pdf_path:
                        ingestor.ingest_record(rec, tags)
                    INDEX_STATUS["processed"] = i + 1
                    
            await loop.run_in_executor(None, run_with_progress)
            
            INDEX_STATUS["current_step"] = "completed"
        except Exception as e:
            logger.exception("Background indexing failed")
            INDEX_STATUS["error"] = str(e)
            INDEX_STATUS["current_step"] = "failed"
        finally:
            INDEX_STATUS["is_running"] = False

    task = asyncio.create_task(run_indexing())
    task.add_done_callback(
        lambda t: logger.error("Indexing task failed", exc_info=t.exception()) if t.exception() else None
    )
    return {"status": "started"}

@app.get("/admin/tags")
async def list_tags():
    """返回可用的标签体系。"""
    from lcr.ingest.auto_tagger import CHEM_TAG_TAXONOMY
    return CHEM_TAG_TAXONOMY

@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    try:
        return await query_paperqa(
            payload.question,
            payload.dois,
            payload.collection_filter,
            payload.tag_filter,
            payload.llm_settings,
            payload.conversation_history,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ask endpoint failed")
        raise HTTPException(status_code=500, detail="Internal server error")
