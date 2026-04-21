from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from paperqa import Settings, ask
from pathlib import Path
from typing import List, Optional

from lcr.ingest.zotero import ZoteroIngestor
from lcr.ingest.normalizer import generate_manifest_csv

app = FastAPI(title="LCR · Literature Citation-RAG")

# 挂载静态文件目录
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

class LLMSettings(BaseModel):
    llm: str = "deepseek/deepseek-chat"
    api_key: str = ""          # 可选，覆盖环境变量
    api_base: str = ""         # 可选，自定义接口地址

class AskRequest(BaseModel):
    question: str
    paper_dir: str = "./papers"
    dois: Optional[List[str]] = None
    llm_settings: Optional[LLMSettings] = None

async def query_paperqa(question: str, paper_dir: str, dois: Optional[List[str]] = None, llm_settings: Optional[LLMSettings] = None) -> dict:
    tmp_dir = None

    if dois:
        # 用软链接把选中的 PDF 平铺到临时目录，避免 relative_to() 失败
        ingestor = ZoteroIngestor()
        all_records = ingestor.load_records()
        doi_to_record = {r.doi.lower(): r for r in all_records}

        tmp_dir = tempfile.mkdtemp(prefix="lcr_papers_")
        linked = 0
        for d in dois:
            rec = doi_to_record.get(d.lower())
            if rec and rec.pdf_path and os.path.exists(rec.pdf_path):
                pdf = Path(rec.pdf_path)
                dest = Path(tmp_dir) / f"{pdf.parent.name}_{pdf.name}"
                if not dest.exists():
                    os.symlink(pdf, dest)
                linked += 1

        if linked == 0:
            raise ValueError("No valid PDFs found for the selected DOIs")

        paper_dir = tmp_dir
        print(f"Linked {linked} PDFs into {tmp_dir}")

    # 处理 LLM 设置
    target_llm = "deepseek/deepseek-chat"
    llm_config = {}
    if llm_settings:
        target_llm = llm_settings.llm
        if llm_settings.api_key:
            key_name = ""
            if "deepseek" in target_llm.lower(): key_name = "DEEPSEEK_API_KEY"
            elif "claude" in target_llm.lower() or "anthropic" in target_llm.lower(): key_name = "ANTHROPIC_API_KEY"
            elif "gpt" in target_llm.lower() or "openai" in target_llm.lower(): key_name = "OPENAI_API_KEY"
            if key_name:
                os.environ[key_name] = llm_settings.api_key
        
        if llm_settings.api_base:
            llm_config["api_base"] = llm_settings.api_base

    settings = Settings(
        llm=target_llm,
        summary_llm=target_llm,
        llm_config=llm_config if llm_config else None,
        summary_llm_config=llm_config if llm_config else None,
        embedding="st-BAAI/bge-small-en-v1.5",
        parsing={"use_doc_details": False},   # 禁用图片/多模态分析，避免调 gpt-4o
        agent={
            "agent_llm": target_llm,
            "agent_llm_config": {"model_list": [{"model_name": target_llm}]} if llm_config else None,
            "index": {
                "paper_directory": paper_dir,
                "recurse_subdirectories": False,
            },
        },
    )

    result = await ask(question, settings=settings)

    citations = []
    if result.session and result.session.contexts:
        for ctx in result.session.contexts:
            citations.append({
                "text": ctx.context[:300],
                "source": ctx.text.doc.citation,
            })

    return {
        "answer": result.session.answer if result.session else "No answer generated.",
        "citations": citations,
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
    # Mock endpoint for Task 9
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
