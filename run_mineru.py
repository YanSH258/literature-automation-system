import sys, subprocess, time, concurrent.futures, os, threading
from pathlib import Path
sys.path.insert(0, "src")

from lcr.ingest.zotero import ZoteroIngestor

CACHE_DIR      = Path("data/mineru_cache")
API_URL        = os.environ.get("MINERU_API_URL", "http://127.0.0.1:8888")
WORKERS        = 1
_CONDA_BIN     = Path(os.environ.get("MINERU_CONDA_BIN", "/home/yan/.conda/envs/lcr/bin"))
MINERU_BIN     = str(_CONDA_BIN / "mineru")
MINERU_API_BIN = str(_CONDA_BIN / "mineru-api")
_SERVER_ENV    = {**os.environ, "MINERU_MODEL_SOURCE": "local", "HF_HUB_OFFLINE": "1"}

_server: subprocess.Popen | None = None
_server_lock = threading.Lock()


def wait_for_server(timeout=120):
    import urllib.request
    for _ in range(timeout // 2):
        try:
            urllib.request.urlopen(f"{API_URL}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def start_server() -> subprocess.Popen:
    global _server
    log = open("mineru_server.log", "a")
    proc = subprocess.Popen(
        [MINERU_API_BIN, "--host", "127.0.0.1", "--port", "8888"],
        stdout=log, stderr=subprocess.STDOUT,
        env=_SERVER_ENV,
    )
    if not wait_for_server():
        proc.terminate()
        raise RuntimeError("mineru-api failed to start in 120s")
    _server = proc
    return proc


def ensure_server():
    """重启已崩溃的服务器，带锁防止并发重启。"""
    global _server
    with _server_lock:
        if _server is None or _server.poll() is not None:
            print("Server down, restarting...", flush=True)
            if _server:
                _server.terminate()
            start_server()
            print(f"Server restarted (PID={_server.pid})", flush=True)


def find_md(stem_dir):
    if not stem_dir.exists():
        return None
    for f in stem_dir.rglob("*.md"):
        if f.stat().st_size > 100:
            return f
    return None


def process_pdf(args):
    i, total, pdf_path = args
    pdf = Path(pdf_path)
    stem_dir = CACHE_DIR / pdf.stem
    if find_md(stem_dir):
        print(f"[{i}/{total}] CACHED  {pdf.name}", flush=True)
        return True

    cmd = [MINERU_BIN, "-p", str(pdf), "-o", str(CACHE_DIR),
           "--api-url", API_URL, "-b", "pipeline", "-m", "auto", "-l", "en"]

    for attempt in range(2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            ok = r.returncode == 0 and find_md(stem_dir) is not None
            print(f"[{i}/{total}] {'OK' if ok else 'FAIL'} {pdf.name}", flush=True)
            if not ok:
                print(f"  stderr: {r.stderr[-800:]}", flush=True)
            return ok
        except subprocess.TimeoutExpired:
            print(f"[{i}/{total}] TIMEOUT {pdf.name}", flush=True)
            return False
        except Exception as e:
            if attempt == 0 and ("ConnectError" in str(e) or "Connection" in str(e)):
                print(f"[{i}/{total}] Server unreachable, restarting... ({pdf.name})", flush=True)
                ensure_server()
                continue
            print(f"[{i}/{total}] ERR {e}", flush=True)
            return False
    return False


# ── 主流程 ────────────────────────────────────────────────────────────────────
records = ZoteroIngestor().load_records()
pdfs = [r.pdf_path for r in records if r.pdf_path]
print(f"Total PDFs: {len(pdfs)}", flush=True)

CACHE_DIR.mkdir(parents=True, exist_ok=True)
pending = [p for p in pdfs if not find_md(CACHE_DIR / Path(p).stem)]
print(f"Pending: {len(pending)}, Cached: {len(pdfs) - len(pending)}", flush=True)
if not pending:
    print("All PDFs already cached. Nothing to do.", flush=True)
    sys.exit(0)

print("Starting mineru-api server...", flush=True)
start_server()
print(f"Server ready (PID={_server.pid}). Processing with {WORKERS} workers...", flush=True)

try:
    tasks = [(i+1, len(pending), p) for i, p in enumerate(pending)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(process_pdf, tasks))
    ok = sum(results)
    print(f"\nDone: {ok}/{len(pending)} succeeded, {len(pending)-ok} failed", flush=True)
finally:
    if _server:
        _server.terminate()
    print("Server stopped.", flush=True)
