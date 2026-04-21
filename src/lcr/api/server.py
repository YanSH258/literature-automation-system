from dotenv import load_dotenv
load_dotenv()

import os
import re
import tempfile
import asyncio
import litellm
import hashlib
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from paperqa import Settings, ask
from pathlib import Path
from typing import List, Optional

from lcr.ingest.zotero import ZoteroIngestor
from lcr.ingest.normalizer import generate_manifest_csv

import uuid
import dataclasses
from lcr.types import LCRChunk
from lcr.citation.citation_map import build_citation_map
from lcr.validation.structural import structural_check

app = FastAPI(title="LCR · Literature Citation-RAG")

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class LLMSettings(BaseModel):
    llm: str = "deepseek/deepseek-chat"
    api_key: str = ""
    api_base: str = ""
    multimodal: int = 0
    evidence_k: int = 10
    max_sources: int = 5

class AskRequest(BaseModel):
    question: str
    paper_dir: str = "./papers"
    dois: Optional[List[str]] = None
    llm_settings: Optional[LLMSettings] = None

async def translate_to_english(text: str, llm: str) -> str:
    """如果检测到中文，将其翻译为英文以便搜索。"""
    try:
        resp = await litellm.acompletion(
            model=llm,
            messages=[{
                "role": "user",
                "content": f"Translate the following scientific query into concise English for literature search. Return ONLY the English translation, no other text:\n\n{text}"
            }],
            temperature=0
        )
        translation = resp.choices[0].message.content.strip()
        print(f"Translation: '{text}' -> '{translation}'")
        return translation
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

async def query_paperqa(question: str, paper_dir: str, dois: Optional[List[str]] = None, llm_settings: Optional[LLMSettings] = None) -> dict:
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

    # 3. 中文提问预处理
    search_question = question
    answer_language_instruction = ""
    if re.search(r'[\u4e00-\u9fff]', question):
        # 翻译成英文搜索，但要求模型用中文回答
        search_question = await translate_to_english(question, target_llm)
        answer_language_instruction = " Please provide the final answer in Chinese (简体中文)."

    # 4. 构造 PaperQA2 设置
    multimodal_val = llm_settings.multimodal if llm_settings else 0
    evidence_k_val = llm_settings.evidence_k if llm_settings else 10
    max_sources_val = llm_settings.max_sources if llm_settings else 5

    settings = Settings(
        llm=target_llm,
        summary_llm=target_llm,
        embedding="st-BAAI/bge-m3", # 使用多语言模型支持中文匹配
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
                "recurse_subdirectories": False,
            },
        },
        prompts={
            "summary": "Summarize the text relevant to the question: {question}. Text: {text}. Focus on objective findings." + answer_language_instruction
        }
    )

    # 5. 执行查询（带超时保护）
    try:
        result = await asyncio.wait_for(
            ask(search_question, settings=settings),
            timeout=120.0 # 文献阅读可能耗时，给 120 秒
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

@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    try:
        return await query_paperqa(payload.question, payload.paper_dir, payload.dois, payload.llm_settings)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
