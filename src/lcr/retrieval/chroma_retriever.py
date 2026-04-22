import chromadb
from pathlib import Path
from typing import List, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from lcr.types import LCRChunk

CHROMA_DIR = Path(__file__).resolve().parents[3] / "data" / "chroma"
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
        zotero_prefilter_k: int = 1000,
    ) -> List[LCRChunk]:
        """
        两阶段检索：
        1. 第一阶段：从 zotero-mcp 检索 Top-K 摘要，提取 DOI。
        2. 第二阶段：在 LCR 正文库中，限定在上述 DOI 范围内进行全文检索。
           每篇论文只保留最相关的 1 个 chunk，确保来源多样性。
        """
        effective_doi_filter = doi_filter
        zotero_metas = {}

        # 第一阶段
        if not effective_doi_filter and self._zotero_client:
            try:
                z_coll = self._zotero_client.get_collection(name="zotero_library")
                q_embeds = self.embedding_fn([question])
                z_res = z_coll.query(
                    query_embeddings=q_embeds,
                    n_results=zotero_prefilter_k,
                    include=["metadatas"]
                )
                if z_res and z_res["metadatas"]:
                    for m in z_res["metadatas"][0]:
                        doi = m.get("doi")
                        if doi:
                            doi_clean = doi.lower().strip()
                            zotero_metas[doi_clean] = {
                                "creators": m.get("creators", ""),
                                "publication": m.get("publication", ""),
                                "citation_key": m.get("citation_key", "")
                            }
                    if zotero_metas:
                        effective_doi_filter = list(zotero_metas.keys())
            except Exception as e:
                print(f"Warning: Zotero-mcp pre-filter failed: {e}")

        # 如果用户传了显式 doi_filter，但 zotero_metas 为空，尝试补充加载
        if doi_filter and not zotero_metas and self._zotero_client:
            try:
                z_coll = self._zotero_client.get_collection(name="zotero_library")
                all_meta = z_coll.get(include=["metadatas"])
                if all_meta and all_meta["metadatas"]:
                    doi_set = {d.lower().strip() for d in doi_filter}
                    for m in all_meta["metadatas"]:
                        doi = m.get("doi")
                        if doi and doi.lower().strip() in doi_set:
                            zotero_metas[doi.lower().strip()] = {
                                "creators": m.get("creators", ""),
                                "publication": m.get("publication", ""),
                                "citation_key": m.get("citation_key", "")
                            }
            except Exception as e:
                print(f"Warning: Failed to load zotero meta for doi_filter: {e}")

        # 第二阶段：多取候选 chunk，再按论文去重，保证来源多样性
        fetch_k = n_results * 5  # 多取 5 倍候选
        raw_chunks = self.retrieve(
            question=question,
            n_results=fetch_k,
            doi_filter=effective_doi_filter,
            collection_filter=collection_filter,
            tag_filter=tag_filter
        )

        # 每篇论文只保留语义最相关的 1 个 chunk（Chroma 已按相关度排序，first = best）
        seen_dois: dict = {}
        for c in raw_chunks:
            doi = c.doc_id
            if doi not in seen_dois:
                seen_dois[doi] = c
        chunks = list(seen_dois.values())[:n_results]

        # 补全元数据
        for c in chunks:
            z_m = zotero_metas.get(c.doc_id.lower().strip(), {})
            c.metadata["creators"] = c.metadata.get("creators") or z_m.get("creators", "")
            c.metadata["publication"] = c.metadata.get("publication") or z_m.get("publication", "")
            c.metadata["citation_key"] = c.metadata.get("citation_key") or z_m.get("citation_key", "")

        stage_info = {
            "stage1_docs": len(zotero_metas),
            "stage2_raw_chunks": len(raw_chunks),
            "final_chunks": len(chunks),
            "used_doi_filter": effective_doi_filter is not None,
        }
        return chunks, stage_info

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
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # 转换为 LCRChunk
        chunks = []
        if results["documents"]:
            distances = results["distances"][0] if results.get("distances") else []
            for i, (text, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), start=1):
                # cosine distance → 0-10 relevance score (distance=0 最相关)
                dist = distances[i - 1] if i - 1 < len(distances) else 1.0
                rcs = round(max(0.0, (1.0 - dist) * 10), 2)
                chunks.append(LCRChunk(
                    chunk_id=f"{meta['doi']}#{int(meta['chunk_seq']):04d}",
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
                    rcs_score=rcs
                ))
        return chunks

_retriever_instance: ChromaRetriever | None = None

def _get_retriever() -> ChromaRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ChromaRetriever()
    return _retriever_instance

def retrieve_chunks(question, **kwargs) -> tuple[List[LCRChunk], dict]:
    if not CHROMA_DIR.exists():
        return [], {}
    return _get_retriever().retrieve_two_stage(question, **kwargs)

