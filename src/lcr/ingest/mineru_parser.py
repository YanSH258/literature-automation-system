import os, shutil, subprocess
from pathlib import Path
from typing import Optional
_CACHE_DIR = Path(os.getenv("MINERU_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "data" / "mineru_cache")))
def parse_pdf_to_markdown(pdf_path: str) -> Optional[str]:
    """用本地 MinerU pipeline 解析 PDF，Markdown 结果缓存复用。"""
    pdf, stem_dir = Path(pdf_path), _CACHE_DIR / Path(pdf_path).stem
    cached = _find_md(stem_dir)
    if cached: return cached.read_text(encoding="utf-8")
    mineru = shutil.which("mineru")
    if mineru is None:
        print(f"[MinerU] mineru not in PATH, skipping {pdf.name}"); return None
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [mineru, "-p", str(pdf), "-o", str(_CACHE_DIR), "-b", "pipeline", "-m", "auto", "-l", "en"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=os.environ)
    except subprocess.TimeoutExpired:
        print(f"[MinerU] timeout: {pdf.name}"); return None
    except Exception as e:
        print(f"[MinerU] error: {e}"); return None
    if r.returncode != 0:
        print(f"[MinerU] failed for {pdf.name}:\n{r.stderr[-300:]}"); return None
    cached = _find_md(stem_dir)
    if cached: return cached.read_text(encoding="utf-8")
    print(f"[MinerU] no .md output found for {pdf.name}"); return None
def _find_md(stem_dir: Path) -> Optional[Path]:
    if not stem_dir.exists(): return None
    for f in stem_dir.rglob("*.md"):
        if f.stat().st_size > 100: return f
    return None
