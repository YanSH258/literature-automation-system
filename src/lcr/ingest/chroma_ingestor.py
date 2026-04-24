import os
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from lcr.ingest.zotero import ZoteroRecord
from lcr.ingest.mineru_chunker import MineruOutputIndex, chunk_from_content_list

from lcr.config import settings as lcr_settings
import logging

logger = logging.getLogger(__name__)

class ChromaIngestor:
    def __init__(self, persist_dir: Path = lcr_settings.CHROMA_DIR, skip_model: bool = False):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        if not skip_model:
            self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=lcr_settings.EMBED_MODEL)
            self.collection = self.client.get_or_create_collection(
                name="lcr_papers",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.abstracts_collection = self.client.get_or_create_collection(
                name="lcr_abstracts",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.embedding_fn = None
            self.collection = self.client.get_collection(name="lcr_papers")
            self.abstracts_collection = self.client.get_collection(name="lcr_abstracts")

        self._zotero_meta = self._load_zotero_meta()
        # MinerU 预生成输出索引（同时扫描 output/ 和 data/mineru_cache/ 两种结构）
        self._mineru_index = MineruOutputIndex(
            lcr_settings.MINERU_OUTPUT_DIR,
            lcr_settings.MINERU_CACHE_DIR,
        )

    def _load_zotero_meta(self) -> Dict[str, Dict[str, str]]:
        """从 zotero-mcp 向量库加载增强元数据。"""
        zotero_db_path = lcr_settings.ZOTERO_MCP_DB_PATH
        if not zotero_db_path or not zotero_db_path.exists():
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
        except Exception:
            logger.exception("Failed to load zotero-mcp meta")
            return {}

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """解析 PDF：优先 MinerU，降级 pypdf。"""
        from lcr.ingest.mineru_parser import parse_pdf_to_markdown
        md = parse_pdf_to_markdown(pdf_path)
        if md:
            return md
        try:
            reader = PdfReader(pdf_path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            logger.warning("pypdf fallback failed for %s: %s", pdf_path, e)
            return ""

    def chunk_text(self, text: str, chunk_size: int = lcr_settings.CHUNK_SIZE, overlap: int = lcr_settings.CHUNK_OVERLAP) -> List[str]:
        """Section 感知分块：按 Markdown 标题拆分，跳过参考文献。"""
        import re
        if not text:
            return []

        _REF = re.compile(
            r'^#{1,3} (?:References|Bibliography|参考文献|致谢|Acknowledgements?)',
            re.IGNORECASE
        )
        sections = re.split(r'(?=\n#{1,3} )', '\n' + text)
        chunks: List[str] = []

        for sec in sections:
            sec = sec.strip()
            if not sec or _REF.match(sec):
                continue
            if len(sec) <= chunk_size:
                chunks.append(sec)
                continue

            # 超长 section：贪心按段落合并，保留标题前缀
            m = re.match(r'^(#{1,3} [^\n]+)\n', sec)
            header = m.group(1) + "\n" if m else ""
            body = sec[len(header):]
            current = header

            for para in re.split(r'\n{2,}', body):
                para = para.strip()
                if not para:
                    continue
                candidate = (current + "\n\n" + para) if current.strip() else para
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current)
                    if len(para) > chunk_size:
                        for i in range(0, len(para), chunk_size - overlap):
                            chunks.append(para[i:i + chunk_size])
                        current = header
                    else:
                        current = (header + para) if header else para

            if current.strip():
                chunks.append(current)

        return chunks or [text[:chunk_size]]

    def ingest_record(self, record: ZoteroRecord, tags: List[str]):
        """将单篇文献索引到 ChromaDB。"""
        doi_safe = record.doi.lower().strip().replace("/", "_")
        z_meta = self._zotero_meta.get(record.doi.lower().strip(), {})

        keywords_str = "|".join(record.keywords) if record.keywords else ""

        existing_abstract = self.abstracts_collection.get(ids=[doi_safe])
        if not existing_abstract["ids"]:
            abstract_text = f"{record.title or ''}\n{record.abstract or ''}".strip()
            if abstract_text:
                self.abstracts_collection.add(
                    ids=[doi_safe],
                    documents=[abstract_text],
                    metadatas=[{
                        "doi": record.doi.lower().strip(),
                        "title": record.title or "",
                        "year": str(record.year or ""),
                        "collection_paths": "|".join(record.collection_paths),
                        "keywords": keywords_str,
                        "creators": z_meta.get("creators", ""),
                        "publication": z_meta.get("publication", ""),
                        "citation_key": z_meta.get("citation_key", ""),
                    }]
                )

        if not record.pdf_path or not os.path.exists(record.pdf_path):
            return

        existing = self.collection.get(where={"doi": record.doi.lower().strip()})
        if existing and existing["ids"]:
            return

        cl_path = self._mineru_index.find(record.pdf_path)
        if cl_path:
            chunk_pairs = chunk_from_content_list(cl_path)
            chunk_texts = [t for t, _ in chunk_pairs]
            chunk_types = [ct for _, ct in chunk_pairs]
        else:
            text = self.extract_text_from_pdf(record.pdf_path)
            chunk_texts = self.chunk_text(text)
            chunk_types = ["text"] * len(chunk_texts)
        if not chunk_texts:
            return

        ids = [f"{doi_safe}#{i:04d}" for i in range(len(chunk_texts))]
        metadatas = [{
            "doi": record.doi.lower().strip(),
            "title": record.title or "Unknown",
            "year": str(record.year or "N/A"),
            "collection_paths": "|".join(record.collection_paths),
            "auto_tags": "|".join(tags),
            "keywords": keywords_str,
            "chunk_seq": i,
            "chunk_type": chunk_types[i],
            "creators": z_meta.get("creators", ""),
            "publication": z_meta.get("publication", ""),
            "citation_key": z_meta.get("citation_key", "")
        } for i in range(len(chunk_texts))]

        batch_size = lcr_settings.CHROMA_BATCH_SIZE
        for i in range(0, len(chunk_texts), batch_size):
            end_idx = min(i + batch_size, len(chunk_texts))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=chunk_texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )

    def ingest_all(self, records: List[ZoteroRecord], tags_map: Dict[str, List[str]]):
        """全量同步。"""
        total = len(records)
        for i, rec in enumerate(records):
            logger.info(f"[{i+1}/{total}] Indexing {rec.doi}...")
            tags = tags_map.get(rec.doi, [])
            self.ingest_record(rec, tags)

    def get_stats(self) -> Dict[str, Any]:
        """获取索引库统计 (优化版，防止在大规模数据下崩溃)。"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "total_docs": "calculating...", 
                "chroma_dir": str(self.persist_dir)
            }
        except Exception as e:
            return {"error": str(e)}

def get_index_stats() -> Dict[str, Any]:
    try:
        client = chromadb.PersistentClient(path=str(lcr_settings.CHROMA_DIR))
        count = client.get_collection(name="lcr_papers").count()
        return {"total_chunks": count, "total_docs": "n/a", "chroma_dir": str(lcr_settings.CHROMA_DIR)}
    except Exception as e:
        return {"error": str(e)}

def run_ingest_all(records: List[ZoteroRecord], tags_map: Dict[str, List[str]]):
    ingestor = ChromaIngestor()
    ingestor.ingest_all(records, tags_map)
