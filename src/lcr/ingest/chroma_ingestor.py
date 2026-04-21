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
        ids = [f"{doi_safe}#{i:04d}" for i in range(len(chunks))]
        metadatas = [{
            "doi": record.doi,
            "title": record.title or "Unknown",
            "year": str(record.year or "N/A"),
            "collection_paths": "|".join(record.collection_paths),
            "auto_tags": "|".join(tags),
            "chunk_seq": i
        } for i in range(len(chunks))]

        # 4. 写入 Chroma
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
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
        """获取索引库统计。"""
        count = self.collection.count()
        # 获取不重复的 DOI 数量
        all_meta = self.collection.get(include=['metadatas'])
        dois = set(m['doi'] for m in all_meta['metadatas']) if all_meta['metadatas'] else set()
        
        return {
            "total_chunks": count,
            "total_docs": len(dois),
            "chroma_dir": str(self.persist_dir)
        }

def get_index_stats() -> Dict[str, Any]:
    return ChromaIngestor().get_stats()

def ingest_all(records: List[ZoteroRecord], tags_map: Dict[str, List[str]]):
    ingestor = ChromaIngestor()
    ingestor.ingest_all(records, tags_map)
