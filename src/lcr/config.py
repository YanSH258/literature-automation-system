from pathlib import Path
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LCR_",
        case_sensitive=False,
    )

    # 路径配置
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    CHROMA_DIR: Path = DATA_DIR / "chroma"
    PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
    MINERU_OUTPUT_DIR: Path = PROJECT_ROOT / "output"
    MINERU_CACHE_DIR: Path = DATA_DIR / "mineru_cache"

    # 外部工具配置
    ZOTERO_MCP_DB_PATH: Optional[Path] = Path.home() / ".config" / "zotero-mcp" / "chroma_db"

    @field_validator("ZOTERO_MCP_DB_PATH", mode="before")
    @classmethod
    def _check_empty_path(cls, v):
        if not v or str(v).strip() == "":
            return None
        return v

    # 模型配置
    EMBED_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_LLM: str = "deepseek/deepseek-chat"
    AUDIT_LLM: str = "deepseek/deepseek-chat"

    # 分块参数
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    CHROMA_BATCH_SIZE: int = 100

    # Stage 1 检索参数
    STAGE1_DIST_THRESHOLD: float = 0.55
    STAGE1_PREFILTER_K: int = 100   # Stage1 每个子查询最多扫描的摘要数
    MIN_STAGE1_RESULTS: int = 5
    TARGET_FETCH_CHUNKS: int = 60

    # 前端/API 默认值
    EVIDENCE_K: int = 50            # 最终返回给 LLM 的 chunk 上限
    CHUNKS_PER_PAPER: int = 4       # Stage2 每篇论文取 chunk 数的参考上限

    # 证据审计参数
    EVIDENCE_MIN_SCORE: float = 6.0
    EVIDENCE_BATCH_SIZE: int = 10
    EVIDENCE_SEMAPHORE: int = 5

    # Auto-tagging（MINIMAX_API_KEY 留空则跳过打标）
    MINIMAX_API_KEY: str = ""
    MINIMAX_API_BASE: str = "https://api.minimax.io/v1"
    TAGGING_BATCH_SIZE: int = 20
    TAGGING_SEMAPHORE: int = 5

    # 默认 Prompt
    SYSTEM_PROMPT: str = """You are a rigorous scientific literature analyst. Rules you MUST follow:
1. Answer in the same language as the question.
2. Every single sentence with a factual claim MUST end with a citation like [n]. No exceptions.
3. Citation format: [n] for single, [n][m] for multiple. NEVER use [n, m] or [n,m].
4. Do NOT write any concluding summary paragraph — end after your last cited point.
5. Only use information from the provided context. Do not hallucinate."""

settings = Settings()
