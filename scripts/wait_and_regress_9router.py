from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "workspace/scheduler/l21_first20_supervisor_complete.json"
INDEX = ROOT / "workspace/offline_index_l21_first20_9router"
OUTPUT = ROOT / "workspace/validation/l21_first20_9router_querypack_results"


def ready() -> bool:
    if not REPORT.exists():
        return False
    try:
        value = json.loads(REPORT.read_text(encoding="utf-8"))
        return value.get("status") == "complete" and Path(value["index"]).resolve() == INDEX.resolve()
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main() -> None:
    while not ready():
        time.sleep(30)
    server = subprocess.Popen(
        ["uv", "run", "--project", "environments/siglip2", "python",
         "scripts/serve_offline_query.py", "--host", "127.0.0.1", "--port", "8767",
         "--index", str(INDEX), "--no-warmup"], cwd=ROOT)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:8767/", timeout=2)
                break
            except Exception:
                if server.poll() is not None:
                    raise RuntimeError(f"query server exited with {server.returncode}")
                time.sleep(2)
        else:
            raise TimeoutError("query server did not become ready")
        subprocess.run(
            [sys.executable, "scripts/run_l21_first20_querypack.py",
             "--base-url", "http://127.0.0.1:8767", "--output", str(OUTPUT)],
            cwd=ROOT, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
