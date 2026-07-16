"""Wrap the existing baseline_model/ and lora_esm_classifier/ outputs as runs,
so the dashboard opens on the frozen-ESM-2 vs LoRA comparison with no retraining.

    python scripts/import_existing_runs.py
"""
import json

import numpy as np
import pandas as pd

from scn1a import config
from scn1a.metrics import classification_metrics
from scn1a.obs import console
from scn1a.runs import Run

ID_COLS = ["Variant_ID", "Gene", "AA_Position", "Ref_AA", "Alt_AA",
           "ClinVar_Label", "Mapped_Region", "Mapped_Functional_Category"]
PRED_COLS = ID_COLS + ["True_Label", "Predicted_Label", "Pathogenic_Probability",
                       "Confidence", "Uncertainty", "Correct"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if "True_Label" not in df:
        df["True_Label"] = df["Binary_Label"]
    df["Correct"] = (df["True_Label"] == df["Predicted_Label"]).astype(int)
    return df[PRED_COLS]


def lora_history(state_path) -> dict:
    log = json.loads(state_path.read_text())["log_history"]
    return {
        "train_loss": [{"step": e["step"], "epoch": e["epoch"], "loss": e["loss"]}
                       for e in log if "loss" in e],
        "eval": [{"epoch": e["epoch"], "loss": e["eval_loss"],
                  **{k[5:]: e[k] for k in e if k.startswith("eval_") and k != "eval_loss"}}
                 for e in log if "eval_loss" in e],
    }


def write_run(run_id, model_type, feature, preds, history, params) -> Run:
    run = Run(run_id)
    run.dir.mkdir(parents=True, exist_ok=True)
    preds = normalize(preds)
    metrics = classification_metrics(preds["True_Label"], preds["Pathogenic_Probability"])
    run.write_json("config.json", {
        "run_id": run_id, "model_type": model_type, "feature": feature,
        "seed": config.SEED, "params": params, "n_test": len(preds), "imported": True,
    })
    run.write_json("metrics.json", metrics)
    run.write_json("history.json", history)
    run.write_predictions(preds)
    run.set_status("done", "imported", 1.0)
    console.log(f"imported {run_id}  (auc {metrics['roc_auc']:.3f}, f1 {metrics['f1']:.3f})")
    return run


def main():
    write_run(
        "baseline-frozen-esm2", "baseline", "mutation_site_embedding",
        pd.read_csv(config.RESULTS / "baseline_model" / "baseline_predictions.csv"),
        {"train_loss": [], "eval": []}, {})

    checkpoint = config.RESULTS / "lora_esm_classifier" / "checkpoint-6354" / "trainer_state.json"
    write_run(
        "lora-esm2-r8", "lora", "residue_window",
        pd.read_csv(config.RESULTS / "lora_esm_classifier" / "lora_predictions.csv"),
        lora_history(checkpoint),
        {"epochs": 6, "lr": 5e-5, "batch_size": 2, "r": 8, "alpha": 16, "dropout": 0.1})


if __name__ == "__main__":
    main()
