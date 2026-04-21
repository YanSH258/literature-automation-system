import os
import sqlite3
import getpass
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
from lcr.ingest.normalizer import normalize_doi

DOI_QUERY = """
SELECT i.itemID, i.key, LOWER(idv.value) AS doi
FROM items i
JOIN itemData id ON i.itemID = id.itemID
JOIN itemDataValues idv ON id.valueID = idv.valueID
JOIN fields f ON id.fieldID = f.fieldID
WHERE f.fieldName = 'DOI'
  AND i.itemTypeID NOT IN (
      SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment', 'note')
  )
"""

ATTACHMENT_QUERY = """
SELECT ci.key AS att_key, ia.path, ia.contentType
FROM items ci
JOIN itemAttachments ia ON ci.itemID = ia.itemID
WHERE ia.parentItemID = ?
  AND (ia.contentType = 'application/pdf' OR ia.path LIKE '%.pdf')
LIMIT 1
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
    year_val.value AS year
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
WHERE ci.collectionID = ?
  AND i.itemTypeID NOT IN (
      SELECT itemTypeID FROM itemTypes WHERE typeName IN ('attachment','note')
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

    def load_collections_tree(self) -> dict:
        """加载 Zotero 集合树及文献元数据。"""
        try:
            db_uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(self.db_path))

        try:
            cursor = conn.cursor()
            
            # 2. 获取所有集合
            cursor.execute(COLLECTION_QUERY)
            collections_rows = cursor.fetchall()
            
            # 3. 预加载所有 DOI 对应的 PDF 路径，提高效率
            cursor.execute(DOI_QUERY)
            doi_rows = cursor.fetchall()
            item_pdf_map = {}
            for item_id, item_key, doi in doi_rows:
                cursor.execute(ATTACHMENT_QUERY, (item_id,))
                att_row = cursor.fetchone()
                pdf_path = None
                if att_row:
                    att_key, path, _ = att_row
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

            # 4. 构造集合及文献数据
            collections = []
            for col_id, col_name, parent_id in collections_rows:
                cursor.execute(ITEM_METADATA_QUERY, (col_id,))
                item_rows = cursor.fetchall()
                
                items = []
                for iid, doi, title, year in item_rows:
                    items.append({
                        "doi": doi or f"no_doi_{iid}",
                        "title": title or doi or f"Untitled {iid}",
                        "year": year[:4] if year else "N/A",
                        "has_pdf": item_pdf_map.get(iid) is not None,
                        "pdf_path": item_pdf_map.get(iid)
                    })
                
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
        """查询所有带 DOI 的文献记录（用于后台匹配）。"""
        records = []
        try:
            db_uri = f"file:{self.db_path}?mode=ro"
            conn = sqlite3.connect(db_uri, uri=True)
        except sqlite3.OperationalError:
            conn = sqlite3.connect(str(self.db_path))
            
        try:
            cursor = conn.cursor()
            cursor.execute(DOI_QUERY)
            doi_rows = cursor.fetchall()
            
            for item_id, item_key, doi in doi_rows:
                cursor.execute(ATTACHMENT_QUERY, (item_id,))
                att_row = cursor.fetchone()
                pdf_path = None
                if att_row:
                    att_key = att_row[0]
                    path = att_row[1]
                    if path:
                        if path.startswith("storage:"):
                            filename = path.replace("storage:", "")
                            full_path = self.zotero_dir / "storage" / att_key / filename
                            if full_path.exists():
                                pdf_path = str(full_path.absolute())
                        elif os.path.isabs(path):
                            if Path(path).exists():
                                pdf_path = str(Path(path).absolute())

                records.append(ZoteroRecord(
                    doi=doi,
                    item_id=item_id,
                    item_key=item_key,
                    pdf_path=pdf_path
                ))
        finally:
            conn.close()
        return records

    def to_manifest_records(self, records: List[ZoteroRecord]) -> List[Dict]:
        """把 ZoteroRecord 列表转为 generate_manifest_csv 需要的格式。"""
        manifest_records = []
        for r in records:
            if r.pdf_path:
                manifest_records.append({
                    "file_location": r.pdf_path,
                    "doi": r.doi,
                    "title": r.title or ""
                })
        return manifest_records
