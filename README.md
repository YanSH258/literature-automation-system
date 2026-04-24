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

| Endpoint | Description |
|---|---|
| `POST /ask` | Ask a question. Accepts `question`, optional `dois`, `collection_filter`, `tag_filter`, `conversation_history` |
| `GET /zotero/collections` | Zotero collection tree for filtering |
| `POST /admin/build-index` | Trigger background re-indexing |
| `GET /admin/index-status` | Index build progress and chunk count |
| `GET /admin/tags` | Available auto-tag taxonomy |
| `POST /admin/test-llm` | Test LLM connectivity with current settings |
| `GET /prompts` | List saved prompt templates |
| `POST /prompts` | Save a new prompt template |

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
