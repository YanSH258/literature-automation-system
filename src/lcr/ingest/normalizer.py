import csv
from pathlib import Path
from typing import List, Dict

def normalize_doi(doi: str) -> str:
    """将 DOI 中的 / 替换为 _ 以便用作文件名/ID。"""
    return doi.replace("/", "_")

def reverse_doi(normalized: str) -> str:
    """还原 normalize_doi 的结果为原始小写 DOI。"""
    return normalized.replace("_", "/").lower()

def generate_manifest_csv(records: List[Dict], path: str) -> None:
    """将记录列表写入 CSV 文件，字段顺序为 file_location, doi, title。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_location", "doi", "title"])
        writer.writeheader()
        writer.writerows(records)
