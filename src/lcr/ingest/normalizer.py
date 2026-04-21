import re
import csv
from pathlib import Path


def normalize_doi(doi: str) -> str:
    if not doi:
        return ""
    doi = str(doi).lower().strip()
    match = re.match(r'^(10\.\d+)[/_](.+)$', doi)
    if match:
        prefix = match.group(1)
        suffix = match.group(2)
        return f"{prefix}/{suffix}".replace("/", "_")
    return doi.replace("/", "_")


def reverse_doi(encoded_doi: str) -> str:
    if not encoded_doi:
        return ""
    match = re.match(r'^(10\.\d+)_(.+)$', str(encoded_doi).lower())
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    return str(encoded_doi).lower()


def generate_manifest_csv(records: list[dict], output_path: str) -> None:
    """
    生成 PaperQA2 支持的 manifest.csv。
    records: [{"file_location": str, "doi": str, "title": str}, ...]
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_location", "doi", "title"])
        writer.writeheader()
        writer.writerows(records)
