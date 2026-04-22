#!/usr/bin/env python3
"""
独立索引脚本，绕过 FastAPI 直接构建 ChromaDB 索引。
用法：
    conda activate lcr
    python run_index.py [--skip-tagging] [--limit N]
"""
import sys
import asyncio
import logging
import argparse
from pathlib import Path

# 用绝对路径，不依赖 cwd
PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ_ROOT / "src"))

LOG_FILE = PROJ_ROOT / "run_index.log"
BACKFILL_LOG_FILE = PROJ_ROOT / "backfill.log"

def _setup_logging(log_file: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

def backfill_metadata():
    """补填存量 Chunks 的元数据字段 (creators, publication, citation_key)。"""
    import sqlite3
    import chromadb

    log.info("=== Starting Metadata Backfill ===")

    # 1. 加载 zotero-mcp 元数据（1120 条，速度快）
    zotero_db_path = Path.home() / ".config" / "zotero-mcp" / "chroma_db"
    if not zotero_db_path.exists():
        log.error(f"zotero-mcp database not found at {zotero_db_path}")
        sys.exit(1)

    z_client = chromadb.PersistentClient(path=str(zotero_db_path))
    z_res = z_client.get_collection("zotero_library").get(include=["metadatas"])
    z_meta = {}
    for m in (z_res["metadatas"] or []):
        doi = m.get("doi")
        if doi:
            z_meta[doi.lower().strip()] = {
                "creators": m.get("creators", ""),
                "publication": m.get("publication", ""),
                "citation_key": m.get("citation_key", ""),
            }
    log.info(f"Loaded {len(z_meta)} records from zotero-mcp.")

    # 2. 直接查 SQLite 获取全部 chunk ID（避免 collection.get() 全量挂死）
    lcr_sqlite = PROJ_ROOT / "data" / "chroma" / "chroma.sqlite3"
    conn = sqlite3.connect(f"file:{lcr_sqlite}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT e.embedding_id
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        JOIN collections c ON s.collection = c.id
        WHERE c.name = 'lcr_papers'
    """).fetchall()
    conn.close()
    all_ids = [r[0] for r in rows]
    log.info(f"Total chunks to scan: {len(all_ids)}")

    # 3. 用 ChromaDB SDK 分批读取 metadata + 更新
    lcr_client = chromadb.PersistentClient(path=str(PROJ_ROOT / "data" / "chroma"))
    collection = lcr_client.get_collection("lcr_papers")

    batch_size = 500
    total_updated = total_scanned = 0

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_res = collection.get(ids=batch_ids, include=["metadatas"])

        update_ids, update_metas = [], []
        for cid, meta in zip(batch_res["ids"], batch_res["metadatas"]):
            doi = meta.get("doi", "").lower().strip()
            if doi in z_meta and (not meta.get("creators") or not meta.get("publication")):
                new_meta = {**meta, **z_meta[doi]}
                update_ids.append(cid)
                update_metas.append(new_meta)

        if update_ids:
            collection.update(ids=update_ids, metadatas=update_metas)
            total_updated += len(update_ids)

        total_scanned += len(batch_ids)
        if total_scanned % 10000 == 0 or total_scanned == len(all_ids):
            log.info(f"  Progress: {total_scanned}/{len(all_ids)} | updated={total_updated}")

    log.info(f"=== Backfill Done: updated={total_updated} / scanned={total_scanned} ===")

async def main(skip_tagging: bool, limit: int | None):
    from lcr.ingest.zotero import ZoteroIngestor
    from lcr.ingest.auto_tagger import tag_all_records
    from lcr.ingest.chroma_ingestor import ChromaIngestor

    log.info("=== LCR indexing started ===")

    # 1. 加载 Zotero 记录
    ingestor = ZoteroIngestor()
    records = ingestor.load_records()
    if limit:
        records = records[:limit]
    log.info(f"Loaded {len(records)} records ({sum(1 for r in records if r.pdf_path)} with PDF)")

    # 2. Auto-tagging
    if skip_tagging:
        tags_map: dict = {}
        log.info("Skipping tagging (--skip-tagging)")
    else:
        log.info("Step 1/2: Auto-tagging via MiniMax...")
        tags_map = await tag_all_records(records)
        log.info(f"Tagging done. {len(tags_map)} papers tagged.")

    # 3. ChromaDB 索引
    log.info("Step 2/2: Indexing to ChromaDB...")
    chroma = ChromaIngestor()
    total = len(records)
    ok = skipped = failed = 0

    for i, rec in enumerate(records):
        if not rec.pdf_path:
            skipped += 1
            continue
        try:
            tags = tags_map.get(rec.doi, [])
            chroma.ingest_record(rec, tags)
            ok += 1
            if (i + 1) % 50 == 0:
                log.info(f"  Progress: {i+1}/{total} | ok={ok} skipped={skipped} failed={failed}")
        except Exception as e:
            failed += 1
            log.warning(f"  Failed [{rec.doi}]: {e}")

    stats = chroma.get_stats()
    log.info(f"=== Done: ok={ok} skipped={skipped} failed={failed} | {stats} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tagging", action="store_true", help="跳过 MiniMax 打标步骤")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 篇（调试用）")
    parser.add_argument("--backfill", action="store_true", help="补填存量分块的元数据 (作者、刊名等)")
    args = parser.parse_args()

    if args.backfill:
        _setup_logging(BACKFILL_LOG_FILE)
        backfill_metadata()
    else:
        _setup_logging(LOG_FILE)
        asyncio.run(main(skip_tagging=args.skip_tagging, limit=args.limit))
