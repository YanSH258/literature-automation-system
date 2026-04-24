"""
从 MinerU 预生成的 content_list.json 中读取结构化块并分 chunk。
替代原先对 markdown 字符串做正则切分的 chunk_text()。
"""
import json
import re
from pathlib import Path
from typing import Optional

_REF_HEADER = re.compile(
    r'^(References|Bibliography|参考文献|致谢|Acknowledgements?)',
    re.IGNORECASE,
)
_SKIP_TYPES = {"header", "footer", "page_number"}


class MineruOutputIndex:
    """扫描 MinerU 输出目录，建立 pdf_filename → content_list.json 的映射。

    支持两种目录结构：
      1. uuid 结构 (output/)   : {uuid}/uploads/*.pdf + {uuid}/{stem}/auto/*_content_list.json
      2. stem 结构 (mineru_cache/) : {stem}/auto/*_content_list.json
    """

    def __init__(self, *dirs: Path):
        self._index: dict[str, Path] = {}
        for d in dirs:
            if d.exists():
                self._build(d)

    def _build(self, output_dir: Path) -> None:
        # 自动检测目录结构类型
        for child in output_dir.iterdir():
            if not child.is_dir():
                continue
            if (child / "uploads").exists():
                self._build_uuid_structure(output_dir)
                return
            if (child / "auto").exists():
                self._build_stem_structure(output_dir)
                return

    def _build_uuid_structure(self, output_dir: Path) -> None:
        for uuid_dir in output_dir.iterdir():
            if not uuid_dir.is_dir():
                continue
            uploads = uuid_dir / "uploads"
            if not uploads.exists():
                continue
            for pdf in uploads.glob("*.pdf"):
                cl = self._find_content_list(uuid_dir, pdf.stem)
                if cl:
                    self._index[pdf.name] = cl

    def _build_stem_structure(self, output_dir: Path) -> None:
        for stem_dir in output_dir.iterdir():
            if not stem_dir.is_dir():
                continue
            cl = self._find_content_list(output_dir, stem_dir.name)
            if cl:
                pdf_name = stem_dir.name + ".pdf"
                self._index[pdf_name] = cl

    @staticmethod
    def _find_content_list(uuid_dir: Path, stem: str) -> Optional[Path]:
        paper_dir = uuid_dir / stem / "auto"
        if not paper_dir.exists():
            return None
        candidates = list(paper_dir.glob("*_content_list.json"))
        if not candidates:
            return None
        # 优先选择 v2 版本（更新的格式），否则取第一个
        for candidate in candidates:
            if "v2" in candidate.name:
                return candidate
        return candidates[0]

    def find(self, pdf_path: str) -> Optional[Path]:
        filename = Path(pdf_path).name
        return self._index.get(filename)

    def __len__(self) -> int:
        return len(self._index)


def _block_text(block: dict) -> tuple[Optional[str], str]:
    """返回 (文本内容, chunk_type)。chunk_type 为 'text'/'table'/'equation'/'list'。"""
    btype = block.get("type")
    if btype == "text":
        return block.get("text", "").strip() or None, "text"
    if btype == "table":
        caption = " ".join((block.get("table_caption") or []) + (block.get("table_footnote") or []))
        body = block.get("table_body", "")
        content = (caption + "\n" + body).strip() if body else caption.strip()
        return content or None, "table"
    if btype == "chart":
        parts = block.get("chart_caption") or []
        return " ".join(parts).strip() or None, "text"
    if btype == "equation":
        return block.get("text", "").strip() or None, "equation"
    if btype == "list":
        if block.get("sub_type") == "ref_text":
            return None, "text"
        items = block.get("list_items") or []
        return "\n".join(items).strip() or None, "text"
    return None, "text"


from lcr.config import settings as lcr_settings

def chunk_from_content_list(
    content_list_path: Path,
    chunk_size: int = lcr_settings.CHUNK_SIZE,
    overlap: int = lcr_settings.CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    """返回 (chunk_text, chunk_type) 列表。chunk_type 为各 chunk 中主体内容的类型。"""
    blocks: list[dict] = json.loads(content_list_path.read_text(encoding="utf-8"))

    chunks: list[tuple[str, str]] = []
    current_section = ""
    current_text = ""
    current_type = "text"
    in_refs = False

    def flush():
        nonlocal current_text, current_type
        if current_text.strip():
            chunks.append((current_text.strip(), current_type))
        current_text = ""
        current_type = "text"

    def section_header() -> str:
        return f"## {current_section}\n" if current_section else ""

    for block in blocks:
        btype = block.get("type")

        if btype in _SKIP_TYPES:
            continue

        text_level = block.get("text_level")

        # Detect reference/acknowledgement section → stop ingesting
        if btype == "text" and text_level in (1, 2):
            raw = block.get("text", "").strip()
            if _REF_HEADER.match(raw):
                in_refs = True

        if in_refs:
            continue

        # Section boundary: flush and start new section
        if btype == "text" and text_level in (1, 2):
            flush()
            current_section = block.get("text", "").strip()
            current_text = section_header()
            continue

        content, ctype = _block_text(block)
        if not content:
            continue

        separator = "\n\n" if current_text.strip() else ""
        candidate = current_text + separator + content

        if len(candidate) <= chunk_size:
            current_text = candidate
            # table/equation take precedence over generic text for labelling
            if ctype in ("table", "equation"):
                current_type = ctype
        else:
            flush()
            hdr = section_header()
            if len(content) > chunk_size:
                for i in range(0, len(content), chunk_size - overlap):
                    chunks.append((hdr + content[i : i + chunk_size], ctype))
                current_text = hdr
                current_type = "text"
            else:
                current_text = hdr + content
                current_type = ctype

    flush()
    return chunks
