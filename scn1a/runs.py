import json
import time
from pathlib import Path

import pandas as pd

from .config import RESULTS

RUNS_DIR = RESULTS / "runs"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class Run:
    """A single training run: a folder of config, metrics, history, predictions.

    Layout under results/runs/<id>/:
        config.json       run type, hyperparameters, seed, split sizes
        metrics.json      final test metrics
        history.json      {train_loss: [...], eval: [...]}  (empty for baselines)
        predictions.csv   test rows + prediction columns, one row per variant
        status.json       live state for the dashboard (queued/running/done/failed)
        train.log         captured stdout of the training subprocess
    """

    def __init__(self, run_id: str):
        self.id = run_id
        self.dir = RUNS_DIR / run_id

    @classmethod
    def create(cls, model_type: str) -> "Run":
        run = cls(f"{time.strftime('%Y%m%d-%H%M%S')}_{model_type}")
        run.dir.mkdir(parents=True, exist_ok=True)
        return run

    def exists(self) -> bool:
        return self.dir.is_dir()

    def write_json(self, name: str, obj) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(obj, indent=2))
        return path

    def read_json(self, name: str, default=None):
        path = self.dir / name
        return json.loads(path.read_text()) if path.exists() else default

    def set_status(self, state: str, message: str = "", progress: float | None = None):
        self.write_json("status.json", {
            "state": state, "message": message,
            "progress": progress, "updated": _now(),
        })

    def write_predictions(self, df: pd.DataFrame) -> Path:
        path = self.dir / "predictions.csv"
        df.to_csv(path, index=False)
        return path


def list_runs() -> list[str]:
    """Run ids, newest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
