# LCR — Literature Citation-RAG

A self-contained RAG system for scientific literature Q&A, built on top of your local Zotero library. Ask questions in natural language and get answers with precise `[n]` citations traceable to the original paper chunks.

---

## How it works

```text
Zotero SQLite  →  ZoteroIngestor  →  ChromaDB (abstracts + full-text chunks)
                                              ↓
              User question  →  HyDE + two-stage retrieval
                                              ↓
                              LLM evidence scoring  →  Answer with citations
```

**Two-stage retrieval:**

- **Stage 1** — Generate a hypothetical abstract (HyDE) with an LLM, query it against the abstract collection alongside the original question. Union of papers within cosine distance threshold.
- **Stage 2** — Retrieve full-text chunks from the selected papers. Chunk count per paper scales inversely with number of papers selected, targeting ~60 chunks total.

**Evidence auditing** — A cheaper LLM scores each chunk 0–10 for relevance. Chunks below the threshold are dropped before the answer is generated.

---

## Requirements

- Python 3.11+
- A local [Zotero](https://www.zotero.org/) installation with PDFs
- A LiteLLM-compatible LLM API key (DeepSeek, OpenAI, Anthropic, etc.)
- [MinerU](https://github.com/opendatalab/MinerU) *(optional, for structured PDF parsing)*

---

## Setup

```bash
conda create -n lcr python=3.11
conda activate lcr
pip install -e .
```

```bash
cp .env.example .env
# Fill in your API keys
```

---

## Ingestion

**Step 1 — Parse PDFs with MinerU** *(optional but recommended)*

```bash
python run_mineru.py
```

Structured JSON output is saved to `data/mineru_cache/`. If skipped, the system falls back to `pypdf`.

### Step 2 — Build the ChromaDB index

```bash
python run_index.py
```

Options:

| Flag | Description |
|---|---|
| `--skip-tagging` | Skip LLM auto-tagging (saves API cost) |
| `--limit N` | Index only the first N records (for testing) |
| `--backfill-keywords` | Write Zotero paper keywords into existing index |
| `--backfill-abstracts` | Re-index abstract collection only |
| `--backfill` | Re-fill author/journal metadata from zotero-mcp |

---

## Running the server

```bash
uvicorn lcr.api.server:app --reload --port 8000
```

Open `http://localhost:8000` for the web UI.

---

## API

交互式文档：服务启动后访问 `http://localhost:8000/docs`（Swagger UI）可直接在浏览器里测试所有接口。

---

### POST /ask — 提问

发送问题，返回带引用编号的答案。

```json
// 请求
{
  "question": "什么材料掺杂可以提升钙钛矿太阳能电池效率？",
  "collection_filter": "钙钛矿",      // 可选：限定在某个 Zotero 分类下检索
  "tag_filter": ["DFT"],              // 可选：按自动标签过滤
  "dois": ["10.1002/adfm.xxx"],       // 可选：只在指定 DOI 的论文里检索
  "conversation_history": [],         // 可选：多轮对话上下文
  "llm_settings": {
    "llm": "deepseek/deepseek-chat",
    "api_key": "sk-...",
    "api_base": ""
  }
}

// 响应
{
  "answer": "研究表明，Cs 掺杂可以显著提升稳定性 [1]，而 Pb 部分替换为 Sn 能拓宽光谱吸收 [2]。",
  "citations": [
    {
      "display_index": 1,
      "doc_id": "10.1002/adfm.xxx",
      "snippet": "原文片段...",
      "rcs_score": 8.5,
      "metadata": { "title": "...", "year": "2023", "creators": "..." }
    }
  ],
  "structural_check": { "passed": true, "uncited_sentences": [] },
  "retrieval_info": { "stage1_found": 12, "final_chunks": 8 }
}
```

---

### GET /zotero/collections — 获取文献分类树

返回你 Zotero 里所有分类（Collection）的层级结构，每个分类包含其中的论文列表。用于前端渲染分类选择器。

---

### POST /admin/build-index — 重建索引

触发后台全量索引任务（读取 Zotero → 打标 → 写入 ChromaDB）。立即返回 `{"status": "started"}`，实际进度通过 `/admin/index-status` 查询。

---

### GET /admin/index-status — 查询索引进度

```json
{
  "is_running": true,
  "current_step": "chroma_indexing",  // loading_zotero / auto_tagging / chroma_indexing / completed
  "processed": 42,
  "total": 150,
  "stats": { "total_chunks": 8300 }
}
```

---

### POST /admin/test-llm — 测试 LLM 连通性

用当前填写的 API Key 和模型名发一条 `ping`，验证是否能正常调用。

```json
// 请求
{ "llm": "deepseek/deepseek-chat", "api_key": "sk-...", "api_base": "" }

// 响应
{ "status": "success", "content": "pong" }
// 或
{ "status": "error", "message": "Connection failed" }
```

---

### GET /prompts / POST /prompts — Prompt 模板管理

- `GET /prompts` — 列出 `prompts/` 目录下所有已保存的 prompt 模板
- `POST /prompts` — 保存新模板，支持变量 `{question}` `{text}` `{citation}` `{summary_length}`

---

### GET /admin/tags — 查看标签体系

返回 `auto_tagger.py` 中定义的 `CHEM_TAG_TAXONOMY`，即 MiniMax 打标时使用的标签分类词典。

---

## Configuration

All settings use the `LCR_` env prefix. Override any in `.env`:

| Variable | Default | Description |
|---|---|---|
| `LCR_DEFAULT_LLM` | `deepseek/deepseek-chat` | LLM for answer generation and HyDE |
| `LCR_AUDIT_LLM` | `deepseek/deepseek-chat` | LLM for evidence scoring (use a cheap/fast model) |
| `LCR_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model, no API key needed |
| `LCR_STAGE1_PREFILTER_K` | `100` | Abstracts scanned per sub-query in Stage 1. **Increase if too few papers are recalled.** |
| `LCR_STAGE1_DIST_THRESHOLD` | `0.55` | Cosine distance cutoff for Stage 1 (higher = more papers pass) |
| `LCR_TARGET_FETCH_CHUNKS` | `60` | Target total chunks for Stage 2 |
| `LCR_EVIDENCE_K` | `50` | Max chunks passed to the answer LLM |
| `LCR_EVIDENCE_MIN_SCORE` | `6.0` | Minimum evidence score (0–10). Lower if answers lack evidence. |
| `LCR_CHUNK_SIZE` | `800` | Character limit per chunk |
| `LCR_MINIMAX_API_KEY` | *(empty)* | MiniMax API key for auto-tagging. Leave empty to skip. |
| `LCR_ZOTERO_DIR` | *(auto-detect)* | Path to Zotero data directory. Auto-detected on WSL/Linux if unset. |
| `LCR_ZOTERO_MCP_DB_PATH` | `~/.config/zotero-mcp/chroma_db` | Path to [zotero-mcp](https://github.com/zotero-mcp/zotero-mcp) ChromaDB. Leave empty to disable. |

See `.env.example` for the full list.

---

## Project structure

```text
src/lcr/
├── config.py              # All settings (pydantic-settings)
├── types.py               # LCRChunk dataclass
├── utils.py               # Shared utilities
├── ingest/
│   ├── zotero.py          # Read Zotero SQLite → ZoteroRecord
│   ├── auto_tagger.py     # LLM-based auto-tagging via MiniMax (optional)
│   ├── chroma_ingestor.py # Write to ChromaDB (lcr_abstracts + lcr_papers)
│   ├── mineru_chunker.py  # Parse MinerU JSON output into chunks
│   └── mineru_parser.py   # Call MinerU API
├── retrieval/
│   └── chroma_retriever.py  # Two-stage retrieval with HyDE
├── generation/
│   └── evidence_auditor.py  # LLM evidence scoring and filtering
├── citation/
│   └── citation_map.py    # display_index ↔ chunk_id mapping
├── validation/
│   └── structural.py      # Citation format validation
└── api/
    ├── server.py           # FastAPI routes
    └── static/             # Web UI
```

---

## Notes on citation IDs

- `chunk_id` — `{doi}#{seq:04d}`, globally unique and permanent
- `display_index` — per-response `[1][2][3]...` numbering shown to the user, resets each turn

All persistent storage uses `chunk_id`. The LLM only sees `display_index`.
