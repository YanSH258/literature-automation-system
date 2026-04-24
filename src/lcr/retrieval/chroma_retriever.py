import chromadb
import threading
from pathlib import Path
from typing import List, Optional
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from lcr.types import LCRChunk

import logging

logger = logging.getLogger(__name__)

from lcr.config import settings as lcr_settings


def _generate_hyde(question: str) -> str:
    """HyDE: ask LLM to write a hypothetical abstract a relevant paper would have.

    The generated abstract uses domain vocabulary that matches real paper language,
    so its embedding lands closer to actual papers than a short user query would.
    Falls back to the original question on any error.
    """
    import litellm
    prompt = (
        "Write a 3-5 sentence scientific abstract of a research paper that would best answer "
        "the following question. Use precise scientific terminology. Write in English.\n\n"
        f"Question: {question}\n\nHypothetical abstract:"
    )
    try:
        resp = litellm.completion(
            model=lcr_settings.DEFAULT_LLM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        text = resp.choices[0].message.content.strip()
        if text:
            return text
    except Exception:
        logger.exception("HyDE generation failed, using original query")
    return question


class ChromaRetriever:
    def __init__(self, persist_dir: Path = lcr_settings.CHROMA_DIR):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name=lcr_settings.EMBED_MODEL)
        self.collection = self.client.get_or_create_collection(
            name="lcr_papers",
            embedding_function=self.embedding_fn
        )
        try:
            self.abstracts_collection = self.client.get_collection(
                name="lcr_abstracts",
                embedding_function=self.embedding_fn
            )
        except Exception:
            self.abstracts_collection = None
        # 初始化 zotero-mcp 客户端
        self.zotero_db_path = lcr_settings.ZOTERO_MCP_DB_PATH
        self._zotero_client = None
        if self.zotero_db_path and self.zotero_db_path.exists():
            self._zotero_client = chromadb.PersistentClient(path=str(self.zotero_db_path))

    def retrieve_two_stage(
        self,
        question: str,
        n_results: int = 20,
        doi_filter: Optional[List[str]] = None,
        collection_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        chunks_per_paper: int = 4,
    ) -> tuple[List[LCRChunk], dict]:
        """
        两阶段检索：
        1. 第一阶段：从 zotero-mcp 检索 Top-K 摘要，提取 DOI。
        2. 第二阶段：在 LCR 正文库中，限定在上述 DOI 范围内进行全文检索。
           每篇论文保留相关度最高的 chunks_per_paper 个片段。
        """
        effective_doi_filter = doi_filter
        zotero_metas = {}

        # 如果指定了 collection_filter 且没有显式 doi_filter，
        # 用 SQLite LIKE 查找属于该 collection 的 DOI。
        # 注意：ChromaDB 1.5.x 的 $contains 算子已失效（总返回空），
        # 必须绕过 SDK 直接查 SQLite。
        if not effective_doi_filter and collection_filter:
            try:
                import sqlite3 as _sqlite3
                _db = Path(str(self.client.get_settings().persist_directory)).resolve() / "chroma.sqlite3"
                _conn = _sqlite3.connect(f"file:{_db}?mode=ro", uri=True)
                _rows = _conn.execute("""
                    SELECT DISTINCT em_doi.string_value
                    FROM embedding_metadata em_cp
                    JOIN embedding_metadata em_doi ON em_doi.id = em_cp.id
                    WHERE em_cp.key = 'collection_paths'
                      AND em_cp.string_value LIKE ?
                      AND em_doi.key = 'doi'
                """, (f"%{collection_filter}%",)).fetchall()
                _conn.close()
                doi_set = {r[0] for r in _rows if r[0]}
                if doi_set:
                    effective_doi_filter = list(doi_set)
            except Exception:
                logger.exception("collection pre-filter failed")

        # Stage 1 候选池大小：固定扫描 STAGE1_PREFILTER_K 篇摘要，保证覆盖足够多的候选文章
        zotero_prefilter_k = lcr_settings.STAGE1_PREFILTER_K

        # 第一阶段 (优先查自己的 lcr_abstracts，带相似度阈值截断)
        # STAGE1_DIST_THRESHOLD: cosine distance 上限，越小越严格
        # 低于阈值的论文被认为不相关直接丢弃，避免宽泛问题把无关文章凑进来
        STAGE1_DIST_THRESHOLD = lcr_settings.STAGE1_DIST_THRESHOLD
        MIN_STAGE1_RESULTS = lcr_settings.MIN_STAGE1_RESULTS

        if not effective_doi_filter and self.abstracts_collection:
            try:
                hyde_abstract = _generate_hyde(question)
                sub_queries = [question, hyde_abstract]
                hyde_label = "HyDE abstract" if hyde_abstract != question else "original query (HyDE fallback)"
                logger.info(f"Stage1 {hyde_label} (first 120 chars): {hyde_abstract[:120]!r}")

                doi_votes: dict[str, int] = {}
                doi_best_dist: dict[str, float] = {}   # best distance seen for fallback
                doi_best_meta: dict[str, dict] = {}

                for sq in sub_queries:
                    sq_embeds = self.embedding_fn([sq])
                    a_res = self.abstracts_collection.query(
                        query_embeddings=sq_embeds,
                        n_results=zotero_prefilter_k,
                        include=["metadatas", "distances"]
                    )
                    if not (a_res and a_res["metadatas"]):
                        continue
                    metas = a_res["metadatas"][0]
                    dists = a_res["distances"][0] if a_res.get("distances") else [0.0] * len(metas)
                    for dist, meta in zip(dists, metas):
                        doi = meta.get("doi")
                        if not doi:
                            continue
                        doi_clean = doi.lower().strip()
                        # Track best distance for ALL papers (for fallback)
                        if doi_clean not in doi_best_dist or dist < doi_best_dist[doi_clean]:
                            doi_best_dist[doi_clean] = dist
                            doi_best_meta[doi_clean] = meta
                        # Only vote if under threshold
                        if dist <= STAGE1_DIST_THRESHOLD:
                            doi_votes[doi_clean] = doi_votes.get(doi_clean, 0) + 1

                # Union: any paper seen by at least one sub-query under threshold is kept.
                # Intersection (min_votes=2) improves precision for specific questions but
                # destroys recall for survey queries ("give me all doping papers"). Since
                # filter_evidence handles quality filtering downstream, favour recall here.
                selected_dois = set(doi_votes.keys())

                # Fallback: pad using closest papers by distance (regardless of threshold)
                if len(selected_dois) < MIN_STAGE1_RESULTS:
                    ranked = sorted(doi_best_dist.keys(), key=lambda d: doi_best_dist[d])
                    selected_dois = set(ranked[:MIN_STAGE1_RESULTS])

                for doi_clean in selected_dois:
                    meta = doi_best_meta[doi_clean]
                    zotero_metas[doi_clean] = {
                        "creators": meta.get("creators", ""),
                        "publication": meta.get("publication", ""),
                        "citation_key": meta.get("citation_key", "")
                    }
                if zotero_metas:
                    effective_doi_filter = list(zotero_metas.keys())
                logger.info(
                    f"Stage1 lcr_abstracts: {len(doi_votes)} unique DOIs → {len(selected_dois)} selected "
                    f"(min_votes=1, sub_queries={len(sub_queries)})"
                )
            except Exception:
                logger.exception("lcr_abstracts Stage 1 failed")

        # 第一阶段 (Fallback: zotero-mcp)
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
            except Exception:
                logger.exception("Zotero-mcp pre-filter failed")

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
            except Exception:
                logger.exception("Failed to load zotero meta for doi_filter")

        # 自适应 chunks_per_paper：论文少→每篇取更多（深度），论文多→每篇取更少（广度）
        # 目标：Stage2 总 chunk 约 60 个，经过 filter_evidence 后剩 15-25 个传给 LLM
        # 用户传入的 chunks_per_paper 只作为上限（不超过该值），实际值由 Stage1 结果数推导
        _TARGET_CHUNKS = lcr_settings.TARGET_FETCH_CHUNKS
        if effective_doi_filter:
            _n_papers = len(effective_doi_filter)
            # 60 // 35 = 1 → max(2,1) = 2；60 // 5 = 12 → min(8,12) = 8
            adaptive_cpp = max(2, min(chunks_per_paper, _TARGET_CHUNKS // max(1, _n_papers)))
        else:
            adaptive_cpp = chunks_per_paper
        logger.info(f"Stage2 adaptive chunks_per_paper={adaptive_cpp} (hint={chunks_per_paper}, papers={len(effective_doi_filter or [])})")

        # 若 Stage 1 运行过但未找到任何候选论文，直接返回空结果
        # ChromaDB 对 {"doi": {"$in": []}} 的行为未定义，必须提前拦截
        if effective_doi_filter is not None and len(effective_doi_filter) == 0:
            return [], {
                "stage1_screened": zotero_prefilter_k,
                "stage1_found": 0,
                "stage2_raw_chunks": 0,
                "final_chunks": 0,
                "chunks_per_paper": adaptive_cpp,
                "scope": collection_filter or "global",
                "used_doi_filter": doi_filter is not None,
            }

        # 第二阶段：召回候选 chunk
        fetch_k = len(effective_doi_filter) * adaptive_cpp * 2 if effective_doi_filter else n_results * 5
        fetch_k = max(fetch_k, n_results)

        # Stage 2：用 effective_doi_filter（来自 SQLite 或用户显式传入）做 $in 检索。
        # 不再传 collection_filter，因为 ChromaDB 1.5.x $contains 算子已失效。
        raw_chunks = self.retrieve(
            question=question,
            n_results=fetch_k,
            doi_filter=effective_doi_filter,
            collection_filter=None,
            tag_filter=tag_filter
        )

        # 每篇保留相关度最高的 adaptive_cpp 个 chunk
        from collections import defaultdict
        paper_chunks: dict = defaultdict(list)
        for c in raw_chunks:
            paper_chunks[c.doc_id].append(c)

        chunks = []
        for doi_chunks in paper_chunks.values():
            chunks.extend(doi_chunks[:adaptive_cpp])

        # 按相关度重新排序，截断到 n_results
        chunks.sort(key=lambda c: c.rcs_score, reverse=True)
        chunks = chunks[:n_results]

        # 补全元数据
        for c in chunks:
            z_m = zotero_metas.get(c.doc_id.lower().strip(), {})
            c.metadata["creators"] = c.metadata.get("creators") or z_m.get("creators", "")
            c.metadata["publication"] = c.metadata.get("publication") or z_m.get("publication", "")
            c.metadata["citation_key"] = c.metadata.get("citation_key") or z_m.get("citation_key", "")

        stage_info = {
            "stage1_screened": zotero_prefilter_k,
            "stage1_found": len(zotero_metas) if zotero_metas else len(effective_doi_filter or []),
            "stage2_raw_chunks": len(raw_chunks),
            "final_chunks": len(chunks),
            "chunks_per_paper": adaptive_cpp,
            "scope": collection_filter or "global",
            "used_doi_filter": doi_filter is not None,
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
        
        if doi_filter:
            if len(doi_filter) == 1:
                where_clauses.append({"doi": doi_filter[0]})
            else:
                where_clauses.append({"doi": {"$in": doi_filter}})
        
        if collection_filter:
            where_clauses.append({"collection_paths": {"$contains": collection_filter}})
            
        if tag_filter:
            if len(tag_filter) == 1:
                where_clauses.append({"auto_tags": {"$contains": tag_filter[0]}})
            else:
                tag_parts = [{"auto_tags": {"$contains": t}} for t in tag_filter]
                where_clauses.append({"$or": tag_parts})

        # 合并过滤条件
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        # 执行检索
        try:
            total = self.collection.count()
            n_results = min(n_results, max(1, total))
        except Exception:
            pass

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
                        "chunk_type": meta.get("chunk_type", "text"),
                        "creators": meta.get("creators", ""),
                        "publication": meta.get("publication", ""),
                        "citation_key": meta.get("citation_key", "")
                    },
                    rcs_score=rcs
                ))
        return chunks

_retriever_instance: ChromaRetriever | None = None
_retriever_lock = threading.Lock()

def _get_retriever() -> ChromaRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        with _retriever_lock:
            if _retriever_instance is None:
                _retriever_instance = ChromaRetriever()
    return _retriever_instance

def retrieve_chunks(
    question: str,
    n_results: int = 20,
    doi_filter: Optional[List[str]] = None,
    collection_filter: Optional[str] = None,
    tag_filter: Optional[List[str]] = None,
    chunks_per_paper: int = 4,
) -> tuple[List[LCRChunk], dict]:
    if not lcr_settings.CHROMA_DIR.exists():
        return [], {}
    return _get_retriever().retrieve_two_stage(
        question, 
        n_results=n_results, 
        doi_filter=doi_filter, 
        collection_filter=collection_filter, 
        tag_filter=tag_filter, 
        chunks_per_paper=chunks_per_paper
    )
