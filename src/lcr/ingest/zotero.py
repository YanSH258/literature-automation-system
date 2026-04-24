import os
import sqlite3
import getpass
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict

DOI_QUERY = """
SELECT i.itemID, i.key, LOWER(idv.value) AS doi
FROM items i
JOIN itemData id ON i.itemID = id.itemID
JOIN itemDataValues idv ON id.valueID = idv.valueID
JOIN fields f ON id.fieldID = f.fieldID
WHERE f.fieldName = 'DOI'
  AND i.libraryID = 1
  AND i.itemTypeID NOT IN (
      SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment', 'note', 'feed', 'feedItem')
  )
"""

ABSTRACT_QUERY = """
SELECT i.itemID, idv.value AS abstract
FROM items i
JOIN itemData id ON i.itemID = id.itemID
JOIN itemDataValues idv ON id.valueID = idv.valueID
JOIN fields f ON id.fieldID = f.fieldID
WHERE f.fieldName = 'abstractNote'
  AND i.libraryID = 1
  AND i.itemTypeID NOT IN (
      SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment','note', 'feed', 'feedItem')
  )
"""

ITEM_COLLECTIONS_QUERY = """
SELECT ci.itemID, c.collectionID, c.collectionName, c.parentCollectionID
FROM collectionItems ci
JOIN collections c ON ci.collectionID = c.collectionID
"""

COLLECTION_QUERY = """
SELECT c.collectionID, c.collectionName, c.parentCollectionID
FROM collections c
JOIN collectionItems ci ON c.collectionID = ci.collectionID
GROUP BY c.collectionID
"""

ITEM_METADATA_QUERY = """
SELECT
    i.itemID,
    LOWER(doi_val.value) AS doi,
    title_val.value AS title,
    year_val.value AS year,
    abs_val.value AS abstract
FROM items i
JOIN collectionItems ci ON i.itemID = ci.itemID
LEFT JOIN itemData doi_d ON i.itemID = doi_d.itemID
    AND doi_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'DOI')
LEFT JOIN itemDataValues doi_val ON doi_d.valueID = doi_val.valueID
LEFT JOIN itemData title_d ON i.itemID = title_d.itemID
    AND title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
LEFT JOIN itemDataValues title_val ON title_d.valueID = title_val.valueID
LEFT JOIN itemData year_d ON i.itemID = year_d.itemID
    AND year_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
LEFT JOIN itemDataValues year_val ON year_d.valueID = year_val.valueID
LEFT JOIN itemData abs_d ON i.itemID = abs_d.itemID
    AND abs_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'abstractNote')
LEFT JOIN itemDataValues abs_val ON abs_d.valueID = abs_val.valueID
WHERE ci.collectionID = ?
  AND i.libraryID = 1
  AND i.itemTypeID NOT IN (
      SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment','note', 'feed', 'feedItem')
  )
"""

@dataclass
class ZoteroRecord:
    doi: str           # 规范化后（小写，/保留）
    item_id: int
    item_key: str
    pdf_path: Optional[str]   # 绝对路径，找不到为 None
    title: Optional[str] = None
    year: Optional[str] = None
    abstract: Optional[str] = None
    collection_paths: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # Zotero tags (期刊关键词 + 手动标签)

class ZoteroIngestor:
    def __init__(self, zotero_dir: Optional[str] = None):
        """
        zotero_dir: Zotero 数据目录。None 则自动探测。
        探测顺序：
          1. 环境变量 ZOTERO_DIR
          2. /mnt/d/zotero
          3. ~/Zotero
          4. /mnt/c/Users/{当前用户名}/Zotero
        """
        self.zotero_dir = Path(zotero_dir) if zotero_dir else self.find_zotero_dir()
        self.db_path = self.zotero_dir / "zotero.sqlite"
        if not self.db_path.exists():
            raise FileNotFoundError(f"Zotero database not found at {self.db_path}")

    def find_zotero_dir(self) -> Path:
        """探测 Zotero 数据目录，找不到抛 FileNotFoundError。"""
        current_user = getpass.getuser()
        
        # 候选顺序
        paths = []
        
        # 1. 环境变量
        env_path = os.environ.get("ZOTERO_DIR")
        if env_path:
            paths.append(Path(env_path))
            
        # 2. WSL2 D 盘常用路径
        paths.append(Path("/mnt/d/zotero"))
        
        # 3. 默认路径
        paths.extend([
            Path.home() / "Zotero",
            Path(f"/mnt/c/Users/{current_user}/Zotero"),
            Path("/mnt/c/Users/yan/Zotero")
        ])
        
        for p in paths:
            if p.exists() and (p / "zotero.sqlite").exists():
                return p
        raise FileNotFoundError(f"Could not find Zotero data directory. Searched: {paths}")

    def to_manifest_records(self, records: list) -> list[dict]:
        return [
            {"file_location": r.pdf_path, "doi": r.doi, "title": r.title or ""}
            for r in records if r.pdf_path
        ]

    def load_collections_tree(self) -> dict:
        """加载 Zotero 集合树及文献元数据。"""
        try:
            db_uri = f"file:{self.db_path}?mode=ro&immutable=1"
            conn = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(self.db_path))

        try:
            cursor = conn.cursor()
            
            # 2. 获取所有集合
            cursor.execute(COLLECTION_QUERY)
            collections_rows = cursor.fetchall()
            
            # 3. 预加载所有 DOI 对应的 PDF 路径
            cursor.execute(DOI_QUERY)
            doi_rows = cursor.fetchall()

            # 一次性取出所有附件（替代 N+1 的 per-item query）
            cursor.execute("""
                SELECT ia.parentItemID, ci.key AS att_key, ia.path
                FROM items ci
                JOIN itemAttachments ia ON ci.itemID = ia.itemID
                WHERE (ia.contentType = 'application/pdf' OR ia.path LIKE '%.pdf')
            """)
            att_map: dict = {}
            for parent_id, att_key, path in cursor.fetchall():
                if parent_id not in att_map:
                    att_map[parent_id] = (att_key, path)

            item_pdf_map = {}
            for item_id, item_key, doi in doi_rows:
                pdf_path = None
                att_row = att_map.get(item_id)
                if att_row:
                    att_key, path = att_row
                    if path:
                        if path.startswith("storage:"):
                            filename = path.replace("storage:", "")
                            full_path = self.zotero_dir / "storage" / att_key / filename
                            if full_path.exists():
                                pdf_path = str(full_path.absolute())
                        elif os.path.isabs(path):
                            if Path(path).exists():
                                pdf_path = str(Path(path).absolute())
                item_pdf_map[item_id] = pdf_path

            # 4. 一次性获取所有集合的文献元数据，避免 N+1
            ALL_METADATA_QUERY = """
            SELECT
                ci.collectionID,
                i.itemID,
                LOWER(doi_val.value) AS doi,
                title_val.value AS title,
                year_val.value AS year,
                abs_val.value AS abstract
            FROM items i
            JOIN collectionItems ci ON i.itemID = ci.itemID
            LEFT JOIN itemData doi_d ON i.itemID = doi_d.itemID
                AND doi_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'DOI')
            LEFT JOIN itemDataValues doi_val ON doi_d.valueID = doi_val.valueID
            LEFT JOIN itemData title_d ON i.itemID = title_d.itemID
                AND title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
            LEFT JOIN itemDataValues title_val ON title_d.valueID = title_val.valueID
            LEFT JOIN itemData year_d ON i.itemID = year_d.itemID
                AND year_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
            LEFT JOIN itemDataValues year_val ON year_d.valueID = year_val.valueID
            LEFT JOIN itemData abs_d ON i.itemID = abs_d.itemID
                AND abs_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'abstractNote')
            LEFT JOIN itemDataValues abs_val ON abs_d.valueID = abs_val.valueID
            WHERE i.libraryID = 1
              AND i.itemTypeID NOT IN (
                  SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment','note', 'feed', 'feedItem')
              )
            """
            cursor.execute(ALL_METADATA_QUERY)
            col_items_map = {}
            for cid, iid, doi, title, year, abstract in cursor.fetchall():
                if cid not in col_items_map:
                    col_items_map[cid] = []
                col_items_map[cid].append({
                    "doi": doi or f"no_doi_{iid}",
                    "title": title or doi or f"Untitled {iid}",
                    "year": year[:4] if year else "N/A",
                    "abstract": abstract,
                    "has_pdf": item_pdf_map.get(iid) is not None,
                    "pdf_path": item_pdf_map.get(iid)
                })

            # 5. 构造集合树
            collections = []
            for col_id, col_name, parent_id in collections_rows:
                items = col_items_map.get(col_id, [])
                
                collections.append({
                    "id": col_id,
                    "name": col_name,
                    "parent_id": parent_id,
                    "items": items
                })
            
            return {"collections": collections}
        finally:
            conn.close()

    def load_records(self) -> List[ZoteroRecord]:
        """查询所有带 DOI 的文献记录（补充摘要和集合路径）。"""
        records = []
        try:
            db_uri = f"file:{self.db_path}?mode=ro&immutable=1"
            conn = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(self.db_path))
            
        try:
            cursor = conn.cursor()
            
            # 1. 抓取摘要映射
            cursor.execute(ABSTRACT_QUERY)
            abstracts = {row[0]: row[1] for row in cursor.fetchall()}

            # 2. 抓取标题和年份映射 (reuse parts of ITEM_METADATA_QUERY logic)
            # 为了性能，这里我们直接扫一遍 items 表获取元数据
            META_ALL_QUERY = """
            SELECT i.itemID, title_val.value, year_val.value
            FROM items i
            LEFT JOIN itemData title_d ON i.itemID = title_d.itemID
                AND title_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'title')
            LEFT JOIN itemDataValues title_val ON title_d.valueID = title_val.valueID
            LEFT JOIN itemData year_d ON i.itemID = year_d.itemID
                AND year_d.fieldID = (SELECT fieldID FROM fields WHERE fieldName = 'date')
            LEFT JOIN itemDataValues year_val ON year_d.valueID = year_val.valueID
            """
            cursor.execute(META_ALL_QUERY)
            metadata = {row[0]: {"title": row[1], "year": row[2]} for row in cursor.fetchall()}

            # 3. 抓取关键词（Zotero tags：期刊自动导入关键词 + 用户手动标签）
            cursor.execute("""
                SELECT it.itemID, t.name
                FROM itemTags it
                JOIN tags t ON it.tagID = t.tagID
            """)
            item_keywords: Dict[int, List[str]] = {}
            for iid, kw in cursor.fetchall():
                item_keywords.setdefault(iid, []).append(kw)

            # 4. 递归集合路径缓存（优化：只对有 item 的集合构建映射）
            cursor.execute("SELECT collectionID, collectionName, parentCollectionID FROM collections")
            col_info = {row[0]: {"name": row[1], "parent": row[2]} for row in cursor.fetchall()}
            
            cursor.execute("SELECT itemID, collectionID FROM collectionItems")
            item_to_cols = {}
            for iid, cid in cursor.fetchall():
                if iid not in item_to_cols: item_to_cols[iid] = []
                item_to_cols[iid].append(cid)

            def get_full_path(cid):
                parts = []
                curr = cid
                while curr in col_info:
                    parts.append(col_info[curr]["name"])
                    curr = col_info[curr]["parent"]
                return "/".join(reversed(parts))

            # 5. 主查询
            cursor.execute(DOI_QUERY)
            doi_rows = cursor.fetchall()
            
            # 一次性取出所有附件（替代 N+1 的 per-item query）
            cursor.execute("""
                SELECT ia.parentItemID, ci.key AS att_key, ia.path
                FROM items ci
                JOIN itemAttachments ia ON ci.itemID = ia.itemID
                WHERE (ia.contentType = 'application/pdf' OR ia.path LIKE '%.pdf')
            """)
            att_map: dict = {}
            for parent_id, att_key, path in cursor.fetchall():
                if parent_id not in att_map:
                    att_map[parent_id] = (att_key, path)

            for item_id, item_key, doi in doi_rows:
                pdf_path = None
                att_row = att_map.get(item_id)
                if att_row:
                    att_key, path = att_row
                    if path:
                        if path.startswith("storage:"):
                            filename = path.replace("storage:", "")
                            full_path = self.zotero_dir / "storage" / att_key / filename
                            if full_path.exists():
                                pdf_path = str(full_path.absolute())
                        elif os.path.isabs(path):
                            if Path(path).exists():
                                pdf_path = str(Path(path).absolute())

                # 拼接集合路径
                cids = item_to_cols.get(item_id, [])
                paths = [get_full_path(cid) for cid in cids]

                meta = metadata.get(item_id, {})
                records.append(ZoteroRecord(
                    doi=doi,
                    item_id=item_id,
                    item_key=item_key,
                    pdf_path=pdf_path,
                    title=meta.get("title"),
                    year=meta.get("year"),
                    abstract=abstracts.get(item_id),
                    collection_paths=paths,
                    keywords=item_keywords.get(item_id, []),
                ))
        finally:
            conn.close()
        return records
