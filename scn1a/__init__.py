"""Shared helpers for the SCN1A ESM project.

`torch_utils` is imported explicitly (not re-exported) so `import scn1a`
stays free of a torch import for the plotting/analysis scripts.
"""
from . import config, data, metrics, obs, plots
from .config import DATA, RESULTS, ROOT, SEED, results_dir
from .data import apply_mutation, read_fasta, sequence_window, stratified_split
from .metrics import classification_metrics, prediction_frame
from .obs import console, metrics_table, save_json, section, timer

__all__ = [
    "config", "data", "metrics", "obs", "plots",
    "ROOT", "DATA", "RESULTS", "SEED", "results_dir",
    "read_fasta", "apply_mutation", "sequence_window", "stratified_split",
    "classification_metrics", "prediction_frame",
    "console", "section", "timer", "metrics_table", "save_json",
]
