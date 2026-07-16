import json
import time
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def section(title: str):
    """Print a titled rule to delimit a stage of a script."""
    console.rule(f"[bold cyan]{title}")


@contextmanager
def timer(label: str):
    """Log the wall-clock duration of the wrapped block."""
    start = time.perf_counter()
    yield
    console.log(f"{label} — {time.perf_counter() - start:.1f}s")


def metrics_table(metrics: dict, title: str = "Metrics"):
    """Render a {name: value} dict as a rich table."""
    table = Table(title=title, title_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


def save_json(obj, path) -> Path:
    """Write `obj` to `path` as indented JSON and log the location."""
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2))
    console.log(f"wrote {path}")
    return path
