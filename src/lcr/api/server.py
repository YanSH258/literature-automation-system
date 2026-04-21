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

class AskRequest(BaseModel):
    question: str
    paper_dir: str = "./papers"
    dois: Optional[List[str]] = None

async def query_paperqa(question: str, paper_dir: str, dois: Optional[List[str]] = None) -> dict:
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
                os.symlink(pdf, dest)
                linked += 1

        if linked == 0:
            raise ValueError("No valid PDFs found for the selected DOIs")

        paper_dir = tmp_dir
        print(f"Linked {linked} PDFs into {tmp_dir}")

    settings = Settings(
        llm="deepseek/deepseek-chat",
        summary_llm="deepseek/deepseek-chat",
        embedding="st-BAAI/bge-small-en-v1.5",
        agent={
            "agent_llm": "deepseek/deepseek-chat",
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

@app.post("/ask")
async def ask_endpoint(payload: AskRequest):
    try:
        return await query_paperqa(payload.question, payload.paper_dir, payload.dois)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
