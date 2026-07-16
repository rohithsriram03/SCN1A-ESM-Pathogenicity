from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

SEED = 42
TEST_SIZE = 0.2

ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
MAX_AA_LENGTH = 1000

LABELS = {0: "Benign", 1: "Pathogenic"}
REGION_ORDER = ["Other", "Pore", "Voltage_Sensor", "Inactivation_Gate"]


def results_dir(name: str) -> Path:
    """Return results/<name>/, creating it if needed."""
    path = RESULTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path
