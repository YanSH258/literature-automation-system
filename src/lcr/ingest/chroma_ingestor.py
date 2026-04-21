import os
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from lcr.ingest.zotero import ZoteroRecord

CACHE_DIR = Path.home() / ".cache" / "lcr"
CHROMA_DIR = CACHE_DIR / "chroma"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class ChromaIngestor:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name="lcr_papers",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        # 加载 zotero-mcp 元数据
        self._zotero_meta = self._load_zotero_meta()

    def _load_zotero_meta(self) -> Dict[str, Dict[str, str]]:
        """从 zotero-mcp 向量库加载增强元数据。"""
        zotero_db_path = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
        if not zotero_db_path.exists():
            return {}
        
        try:
            z_client = chromadb.PersistentClient(path=str(zotero_db_path))
            z_coll = z_client.get_collection(name="zotero_library")
            # 取出所有元数据
            all_data = z_coll.get(include=["metadatas"])
            meta_map = {}
            if all_data and all_data["metadatas"]:
                for m in all_data["metadatas"]:
                    doi = m.get("doi")
                    if doi:
                        clean_doi = doi.lower().strip()
                        meta_map[clean_doi] = {
                            "creators": m.get("creators", ""),
                            "publication": m.get("publication", ""),
                            "citation_key": m.get("citation_key", "")
                        }
            return meta_map
        except Exception as e:
            print(f"Warning: Failed to load zotero-mcp meta: {e}")
            return {}

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从 PDF 提取纯文本。"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
        """简单的滑动窗口分块。"""
        if not text:
            return []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += (chunk_size - overlap)
        return chunks

    def ingest_record(self, record: ZoteroRecord, tags: List[str]):
        """将单篇文献索引到 ChromaDB。"""
        if not record.pdf_path or not os.path.exists(record.pdf_path):
            return

        # 1. 检查是否已存在（增量索引）
        existing = self.collection.get(where={"doi": record.doi})
        if existing and existing["ids"]:
            return # Skip

        # 2. 提取并分块
        text = self.extract_text_from_pdf(record.pdf_path)
        chunks = self.chunk_text(text)
        if not chunks:
            return

        # 3. 准备元数据
        doi_safe = record.doi.replace("/", "_")
        z_meta = self._zotero_meta.get(record.doi.lower().strip(), {})
        
        ids = [f"{doi_safe}#{i:04d}" for i in range(len(chunks))]
        metadatas = [{
            "doi": record.doi,
            "title": record.title or "Unknown",
            "year": str(record.year or "N/A"),
            "collection_paths": "|".join(record.collection_paths),
            "auto_tags": "|".join(tags),
            "chunk_seq": i,
            "creators": z_meta.get("creators", ""),
            "publication": z_meta.get("publication", ""),
            "citation_key": z_meta.get("citation_key", "")
        } for i in range(len(chunks))]

        # 4. 写入 Chroma (分批写入以防触发 SQL 变量过多错误)
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=chunks[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

    def ingest_all(self, records: List[ZoteroRecord], tags_map: Dict[str, List[str]]):
        """全量同步。"""
        total = len(records)
        for i, rec in enumerate(records):
            if rec.pdf_path:
                print(f"[{i+1}/{total}] Indexing {rec.doi}...")
                tags = tags_map.get(rec.doi, [])
                self.ingest_record(rec, tags)

    def get_stats(self) -> Dict[str, Any]:
        """获取索引库统计 (优化版，防止在大规模数据下崩溃)。"""
        try:
            count = self.collection.count()
            # 统计文献数：因为目前是按 DOI#0000 命名的，所以文献数约为 chunk 数 / 100 (估算)
            # 或者为了保持性能，暂时只返回 chunk 总数
            return {
                "total_chunks": count,
                "total_docs": "calculating...", # 或者返回大致估算
                "chroma_dir": str(self.persist_dir)
            }
        except Exception as e:
            return {"error": str(e)}

def get_index_stats() -> Dict[str, Any]:
    return ChromaIngestor().get_stats()

def ingest_all(records: List[ZoteroRecord], tags_map: Dict[str, List[str]]):
    ingestor = ChromaIngestor()
    ingestor.ingest_all(records, tags_map)
