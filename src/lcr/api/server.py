from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import asyncio
import litellm
import hashlib
import json
import glob as glob_module
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from paperqa import Settings, ask
from pathlib import Path
from typing import List, Optional

from lcr.ingest.zotero import ZoteroIngestor, ZoteroRecord
from lcr.ingest.normalizer import generate_manifest_csv
from lcr.ingest.auto_tagger import tag_all_records, tag_all_records_sync
from lcr.ingest.chroma_ingestor import ingest_all, get_index_stats, CHROMA_DIR
from lcr.retrieval.chroma_retriever import retrieve_chunks

import uuid
import dataclasses
from lcr.types import LCRChunk
from lcr.citation.citation_map import build_citation_map
from lcr.validation.structural import structural_check

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
PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class LLMSettings(BaseModel):
    llm: str = "deepseek/deepseek-chat"
    summary_llm: str = ""  # 留空时与 llm 相同，用于 RCS 打分的便宜模型
    api_key: str = ""
    api_base: str = ""
    multimodal: int = 0
    evidence_k: int = 10
    max_sources: int = 5
    prompt_template: str = ""

class AskRequest(BaseModel):
    question: str
    paper_dir: str = "./papers"
    dois: Optional[List[str]] = None
    collection_filter: Optional[str] = None
    tag_filter: Optional[List[str]] = None
    llm_settings: Optional[LLMSettings] = None

async def query_paperqa(question: str, paper_dir: str, 
                        dois: Optional[List[str]] = None, 
                        collection_filter: Optional[str] = None,
                        tag_filter: Optional[List[str]] = None,
                        llm_settings: Optional[LLMSettings] = None) -> dict:
    
    # --- 优先尝试 ChromaDB 检索流程 ---
    chroma_ready = (CHROMA_DIR / "chroma.sqlite3").exists()
    if chroma_ready:
        print(f"Using ChromaDB retrieval for question: {question}")
        chunks = retrieve_chunks(
            question, 
            n_results=llm_settings.evidence_k if llm_settings else 20,
            doi_filter=dois,
            collection_filter=collection_filter,
            tag_filter=tag_filter
        )
        
        if chunks:
            # 构造 Prompt 让 LLM 直接基于 chunks 回答
            target_llm = llm_settings.llm if llm_settings else "deepseek/deepseek-chat"
            
            context_str = ""
            for i, c in enumerate(chunks, start=1):
                meta = c.metadata
                authors = meta.get('creators', '')
                pub = meta.get('publication', '')
                year = meta.get('year', '')
                header_parts = [f"doi: {meta.get('doi')}"]
                if authors: header_parts.append(f"authors: {authors}")
                if pub: header_parts.append(f"journal: {pub}")
                if year: header_parts.append(f"year: {year[:4]}")
                
                context_str += f"[{i}] ({', '.join(header_parts)})\n{c.text}\n\n"
            
            prompt = f"""Answer the question based on the provided context from scientific papers.
For every factual claim, cite the source using [n] notation where n is the chunk number.
If context is insufficient, say so.

Context:
{context_str}

Question: {question}

Answer:"""
            
            try:
                response = await litellm.acompletion(
                    model=target_llm,
                    api_key=llm_settings.api_key if llm_settings and llm_settings.api_key else None,
                    api_base=llm_settings.api_base if llm_settings and llm_settings.api_base else None,
                    messages=[{"role": "user", "content": prompt}]
                )
                answer_text = response.choices[0].message.content
                
                session_id = str(uuid.uuid4())
                cmap = build_citation_map(chunks, session_id=session_id, turn_id=1)
                structural = structural_check(answer_text, max_index=len(chunks))
                
                citations_out = []
                for idx, entry in cmap.entries.items():
                    citations_out.append({
                        "display_index": entry.display_index,
                        "chunk_id": entry.chunk_id,
                        "doc_id": entry.doc_id,
                        "snippet": entry.snippet,
                        "rcs_score": 0.0,
                        "metadata": entry.metadata,
                    })

                return {
                    "answer": answer_text,
                    "citations": citations_out,
                    "structural_check": {
                        "passed": structural.passed,
                        "issues": structural.issues,
                        "cited_indices": structural.cited_indices,
                    },
                    "retrieval_source": "chromadb"
                }
            except Exception as e:
                print(f"ChromaDB workflow failed, falling back to PaperQA2: {e}")

    # --- Fallback: 原始 PaperQA2 流程 ---
    # 1. 环境与路径准备
    if not os.path.exists(paper_dir):
        os.makedirs(paper_dir, exist_ok=True)

    if dois:
        ingestor = ZoteroIngestor()
        all_records = ingestor.load_records()
        doi_to_record = {r.doi.lower(): r for r in all_records}

        # 用排序后的 DOI 列表生成稳定哈希，相同选纸集合复用同一目录
        doi_hash = hashlib.md5(",".join(sorted(d.lower() for d in dois)).encode()).hexdigest()[:12]
        cache_dir = Path(tempfile.gettempdir()) / f"lcr_papers_{doi_hash}"
        cache_dir.mkdir(exist_ok=True)

        linked = 0
        for d in dois:
            rec = doi_to_record.get(d.lower())
            if rec and rec.pdf_path and os.path.exists(rec.pdf_path):
                pdf = Path(rec.pdf_path)
                dest = cache_dir / f"{pdf.parent.name}_{pdf.name}"
                if not dest.exists():
                    os.symlink(pdf, dest)
                linked += 1
        if linked == 0:
            raise ValueError("No valid PDFs found for the selected DOIs")
        paper_dir = str(cache_dir)

    # 2. 模型与 API 配置
    target_llm = "deepseek/deepseek-chat"
    if llm_settings:
        target_llm = llm_settings.llm
        if llm_settings.api_key:
            if "deepseek" in target_llm.lower(): os.environ["DEEPSEEK_API_KEY"] = llm_settings.api_key
            elif "claude" in target_llm.lower() or "anthropic" in target_llm.lower(): os.environ["ANTHROPIC_API_KEY"] = llm_settings.api_key
            elif "moonshot" in target_llm.lower() or "kimi" in target_llm.lower():
                os.environ["MOONSHOT_API_KEY"] = llm_settings.api_key
                os.environ["OPENAI_API_KEY"] = llm_settings.api_key  # kimi-k2.x 使用 openai/ 前缀路由时需要
            elif "gemini" in target_llm.lower(): os.environ["GEMINI_API_KEY"] = llm_settings.api_key
            else: os.environ["OPENAI_API_KEY"] = llm_settings.api_key
        
        # 全局设置 litellm.api_base 以便支持自定义服务商
        litellm.api_base = llm_settings.api_base if llm_settings.api_base else None

    target_summary_llm = (llm_settings.summary_llm.strip()
                          if llm_settings and llm_settings.summary_llm.strip()
                          else target_llm)

    # 3. 提问预处理
    search_question = question

    # 4. 构造 PaperQA2 设置
    multimodal_val = llm_settings.multimodal if llm_settings else 0
    evidence_k_val = llm_settings.evidence_k if llm_settings else 10
    max_sources_val = llm_settings.max_sources if llm_settings else 5

    # 优先使用用户传入的自定义模板，否则用默认值
    base_summary_prompt = (
        llm_settings.prompt_template.strip()
        if llm_settings and llm_settings.prompt_template.strip()
        else "Summarize the text relevant to the question: {question}. Text: {text}. Focus on objective findings."
    )

    settings = Settings(
        llm=target_llm,
        summary_llm=target_summary_llm,
        embedding="st-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # 多语言470MB，比BGE-M3快10倍
        parsing={"multimodal": multimodal_val},
        answer={
            "evidence_k": evidence_k_val,
            "answer_max_sources": max_sources_val,
        },
        agent={
            "agent_llm": target_llm,
            "search_count": 3, # 限制 Agent 最大搜索次数，防止死循环
            "index": {
                "paper_directory": paper_dir,
                "index_directory": str(Path(tempfile.gettempdir()) / f"lcr_index_{doi_hash if dois else 'default'}"),
                "recurse_subdirectories": False,
            },
        },
        prompts={
            "summary": base_summary_prompt
        }
    )

    # 5. 执行查询（带超时保护）
    try:
        result = await asyncio.wait_for(
            ask(search_question, settings=settings),
            timeout=600.0 # 大批量文献首次索引+查询，给 600 秒
        )
    except asyncio.TimeoutError:
        return {"answer": "The analysis timed out. Please try selecting fewer papers or a simpler question.", "citations": []}

    # 6. 构造返回结果
    answer_text = result.session.answer if result.session else "No answer generated."

    chunks: list[LCRChunk] = []
    if result.session and result.session.contexts:
        for seq, ctx in enumerate(result.session.contexts, start=1):
            chunk = LCRChunk.from_paperqa_text(ctx.text, seq)
            chunk = dataclasses.replace(chunk, rcs_score=float(ctx.score or 0.0))
            chunks.append(chunk)

    session_id = str(uuid.uuid4())
    cmap = build_citation_map(chunks, session_id=session_id, turn_id=1)

    # M6: 将 PaperQA2 的 (pqac-xxx) 引用 key 替换为 [n] 编号
    # 先构建 doc_id -> display_index 映射（每个 doc 取最小编号的 chunk）
    doc_to_idx: dict[str, int] = {}
    for idx, entry in cmap.entries.items():
        if entry.doc_id not in doc_to_idx or idx < doc_to_idx[entry.doc_id]:
            doc_to_idx[entry.doc_id] = idx

    def replace_pqac_citation(match: re.Match) -> str:
        inner = match.group(1)  # e.g. "pqac-d79ef6fa, pqac-0f650d59"
        keys = [k.strip() for k in inner.split(",")]
        nums = []
        for k in keys:
            if k in doc_to_idx:
                nums.append(str(doc_to_idx[k]))
        if nums:
            # 使用 [1][2] 格式以绕过 structural_check 对 [1, 2] 的禁止
            return "".join(f"[{n}]" for n in nums)
        return match.group(0)

    # 捕获组包含 key 列表，以便 replace_pqac_citation 使用 match.group(1)
    answer_text = re.sub(r'\((pqac-[a-f0-9]+(?:,\s*pqac-[a-f0-9]+)*)\)',
                        replace_pqac_citation, answer_text)

    structural = structural_check(answer_text, max_index=len(chunks))

    citations_out = []
    for idx, entry in cmap.entries.items():
        citations_out.append({
            "display_index": entry.display_index,
            "chunk_id": entry.chunk_id,
            "doc_id": entry.doc_id,
            "snippet": entry.snippet,
            "rcs_score": round(entry.rcs_score, 2),
            "metadata": entry.metadata,
            "from_previous_turn": entry.from_previous_turn,
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
    }

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
    for path in sorted(PROMPTS_DIR.glob("*.json")):
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
    path = PROMPTS_DIR / filename
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
    stats = get_index_stats() if CHROMA_DIR.exists() else {"total_chunks": 0, "total_docs": 0}
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
            
            # 1. 加载 Zotero
            INDEX_STATUS["current_step"] = "loading_zotero"
            ingestor = ZoteroIngestor()
            records = ingestor.load_records()
            INDEX_STATUS["total"] = len(records)
            
            # 2. 打标 (LLM)
            INDEX_STATUS["current_step"] = "auto_tagging"
            tags_map = await tag_all_records(records)
            
            # 3. 索引 (Chroma)
            INDEX_STATUS["current_step"] = "chroma_indexing"
            # 这里简单处理，同步转异步或直接跑在线程池
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ingest_all, records, tags_map)
            
            INDEX_STATUS["current_step"] = "completed"
        except Exception as e:
            import traceback
            traceback.print_exc()
            INDEX_STATUS["error"] = str(e)
            INDEX_STATUS["current_step"] = "failed"
        finally:
            INDEX_STATUS["is_running"] = False

    asyncio.create_task(run_indexing())
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
            payload.paper_dir, 
            payload.dois, 
            payload.collection_filter,
            payload.tag_filter,
            payload.llm_settings
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
