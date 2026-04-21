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
        # 初始化 zotero-mcp 客户端
        self.zotero_db_path = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
        self._zotero_client = None
        if self.zotero_db_path.exists():
            self._zotero_client = chromadb.PersistentClient(path=str(self.zotero_db_path))

    def retrieve_two_stage(
        self,
        question: str,
        n_results: int = 20,
        doi_filter: Optional[List[str]] = None,
        collection_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        zotero_prefilter_k: int = 50,
    ) -> List[LCRChunk]:
        """
        两阶段检索：
        1. 第一阶段：从 zotero-mcp 检索 Top-K 摘要，提取 DOI。
        2. 第二阶段：在 LCR 正文库中，限定在上述 DOI 范围内进行全文检索。
        """
        effective_doi_filter = doi_filter
        zotero_metas = {} # 用于补全元数据

        # 第一阶段：如果未显式指定 DOI 过滤，且 zotero-mcp 可用
        if not effective_doi_filter and self._zotero_client:
            try:
                # 方案：不直接 query_texts，而是先对 question 做 embedding，然后用 query_embeddings
                z_coll = self._zotero_client.get_collection(name="zotero_library")
                q_embeds = self.embedding_fn([question])
                
                z_res = z_coll.query(
                    query_embeddings=q_embeds,
                    n_results=zotero_prefilter_k,
                    include=["metadatas"]
                )
                
                if z_res and z_res["metadatas"]:
                    found_dois = []
                    for m in z_res["metadatas"][0]:
                        doi = m.get("doi")
                        if doi:
                            doi_clean = doi.lower().strip()
                            found_dois.append(doi_clean)
                            zotero_metas[doi_clean] = {
                                "creators": m.get("creators", ""),
                                "publication": m.get("publication", ""),
                                "citation_key": m.get("citation_key", "")
                            }
                    if found_dois:
                        effective_doi_filter = list(set(found_dois))
            except Exception as e:
                import traceback
                # traceback.print_exc()
                print(f"Warning: Zotero-mcp pre-filter failed: {e}")

        # 第二阶段：执行常规检索
        chunks = self.retrieve(
            question=question,
            n_results=n_results,
            doi_filter=effective_doi_filter,
            collection_filter=collection_filter,
            tag_filter=tag_filter
        )

        # 补全元数据
        for c in chunks:
            z_m = zotero_metas.get(c.doc_id.lower().strip(), {})
            # 只有当 metadata 中缺失这些字段时才补全（或者直接覆盖以确保最新）
            c.metadata["creators"] = c.metadata.get("creators") or z_m.get("creators", "")
            c.metadata["publication"] = c.metadata.get("publication") or z_m.get("publication", "")
            c.metadata["citation_key"] = c.metadata.get("citation_key") or z_m.get("citation_key", "")

        return chunks

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
                        "auto_tags": meta["auto_tags"],
                        "creators": meta.get("creators", ""),
                        "publication": meta.get("publication", ""),
                        "citation_key": meta.get("citation_key", "")
                    },
                    rcs_score=0.0 # Placeholder
                ))
        return chunks

def retrieve_chunks(question, **kwargs) -> List[LCRChunk]:
    if not CHROMA_DIR.exists():
        return []
    return ChromaRetriever().retrieve_two_stage(question, **kwargs)
