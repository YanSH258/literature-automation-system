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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

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
    args = parser.parse_args()

    asyncio.run(main(skip_tagging=args.skip_tagging, limit=args.limit))
