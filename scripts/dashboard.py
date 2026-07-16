"""Local dashboard for inspecting and launching training runs.

    python scripts/dashboard.py           # serve on http://localhost:8000
    python scripts/dashboard.py --port 8080

Serves scn1a/dashboard.html plus a small JSON API over results/runs/. Training
is launched as a detached subprocess (scripts/train_run.py) that streams to the
run's train.log and updates its status.json; the page polls until it finishes.
No third-party web dependencies — stdlib http.server only.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from scn1a.config import ROOT
from scn1a.runs import RUNS_DIR, Run, list_runs

HTML = ROOT / "scn1a" / "dashboard.html"
TRAIN = ROOT / "scripts" / "train_run.py"
NUMERIC = {"AA_Position", "True_Label", "Predicted_Label",
           "Pathogenic_Probability", "Confidence", "Uncertainty", "Correct"}


def run_summary(run_id: str) -> dict:
    run = Run(run_id)
    config = run.read_json("config.json", {})
    return {
        "id": run_id,
        "model_type": config.get("model_type", "?"),
        "feature": config.get("feature", ""),
        "imported": config.get("imported", False),
        "metrics": run.read_json("metrics.json", {}),
        "status": run.read_json("status.json", {"state": "unknown"}),
    }


def run_detail(run_id: str) -> dict:
    run = Run(run_id)
    predictions = []
    csv_path = run.dir / "predictions.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                predictions.append({k: (float(v) if k in NUMERIC and v not in ("", "nan")
                                        else v) for k, v in row.items()})
    return {
        "config": run.read_json("config.json", {}),
        "metrics": run.read_json("metrics.json", {}),
        "history": run.read_json("history.json", {"train_loss": [], "eval": []}),
        "status": run.read_json("status.json", {"state": "unknown"}),
        "predictions": predictions,
    }


def start_training(spec: dict) -> str:
    model = spec.get("model_type", "lora")
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{model}"
    run = Run(run_id)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.set_status("queued", "starting")

    cmd = [sys.executable, str(TRAIN), "--model", model, "--run-id", run_id]
    for flag in ("epochs", "lr", "batch_size", "r", "alpha", "dropout"):
        if flag in spec:
            cmd += [f"--{flag.replace('_', '-')}", str(spec[flag])]

    log = (run.dir / "train.log").open("w")
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    return run_id


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(HTML.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/runs":
            return self._json([run_summary(r) for r in list_runs()])
        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            if (RUNS_DIR / run_id).is_dir():
                return self._json(run_detail(run_id))
            return self._json({"error": "not found"}, 404)
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/train":
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", 0))
        spec = json.loads(self.rfile.read(length) or "{}")
        self._json({"run_id": start_training(spec)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"SCN1A run dashboard → {url}  (Ctrl-C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(("localhost", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
