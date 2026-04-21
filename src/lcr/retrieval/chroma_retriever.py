import chromadb
from pathlib import Path
from typing import List, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from lcr.types import LCRChunk

CACHE_DIR = Path.home() / ".cache" / "lcr"
CHROMA_DIR = CACHE_DIR / "chroma"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class ChromaRetriever:
    def __init__(self, persist_dir: Path = CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name="lcr_papers",
            embedding_function=self.embedding_fn
        )

    def retrieve(
        self,
        question: str,
        n_results: int = 20,
        doi_filter: Optional[List[str]] = None,
        collection_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None
    ) -> List[LCRChunk]:
        """
        检索并转换为 LCRChunk 格式。
        """
        # 构建 Chroma 过滤条件 (where)
        where_clauses = []
        
        # 1. DOI 显式过滤 (优先级最高)
        if doi_filter:
            if len(doi_filter) == 1:
                where_clauses.append({"doi": doi_filter[0]})
            else:
                where_clauses.append({"doi": {"$in": doi_filter}})
        
        # 2. 集合过滤 (模糊匹配，Chroma $contains)
        if collection_filter:
            where_clauses.append({"collection_paths": {"$contains": collection_filter}})
            
        # 3. 标签过滤
        if tag_filter:
            if len(tag_filter) == 1:
                where_clauses.append({"auto_tags": {"$contains": tag_filter[0]}})
            else:
                # 多个 tag 使用 $or 或 $and
                tag_parts = [{"auto_tags": {"$contains": t}} for t in tag_filter]
                where_clauses.append({"$or": tag_parts})

        # 合并过滤条件
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # 执行检索
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            where=where
        )

        # 转换为 LCRChunk
        chunks = []
        if results["documents"]:
            for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
                chunks.append(LCRChunk(
                    chunk_id=f"{meta['doi']}#{meta['chunk_seq']:04d}",
                    doc_id=meta["doi"],
                    text=text,
                    section="Body",
                    page=0,
                    char_start=0,
                    char_end=len(text),
                    metadata={
                        "title": meta["title"],
                        "year": meta["year"],
                        "doi": meta["doi"],
                        "collection_paths": meta["collection_paths"],
                        "auto_tags": meta["auto_tags"]
                    },
                    rcs_score=0.0 # Placeholder
                ))
        return chunks

def retrieve_chunks(question, **kwargs) -> List[LCRChunk]:
    if not CHROMA_DIR.exists():
        return []
    return ChromaRetriever().retrieve(question, **kwargs)
