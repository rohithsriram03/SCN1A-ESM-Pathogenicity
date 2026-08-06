from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmarking"

INPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_with_alphamissense.csv"

OUTPUT_DIR = BENCHMARK_DIR / "alphamissense_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "alphamissense_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "alphamissense_metrics.json"
CATEGORY_SUMMARY_FILE = OUTPUT_DIR / "alphamissense_by_functional_category.csv"
THRESHOLD_SWEEP_FILE = OUTPUT_DIR / "alphamissense_threshold_sweep.csv"


ALPHAMISSENSE_PATHOGENIC_THRESHOLD = 0.564


def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Pathogenic"],
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics


def threshold_sweep(y_true, scores):
    rows = []

    thresholds = np.linspace(0.0, 1.0, 1001)

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    sweep = pd.DataFrame(rows)

    return sweep


def category_summary(df):
    rows = []

    for category, sub in df.groupby("Mapped_Functional_Category"):
        if sub["Binary_Label"].nunique() < 2:
            auc = np.nan
            ap = np.nan
        else:
            auc = roc_auc_score(sub["Binary_Label"], sub["AlphaMissense_Score"])
            ap = average_precision_score(sub["Binary_Label"], sub["AlphaMissense_Score"])

        pred = (sub["AlphaMissense_Score"] >= ALPHAMISSENSE_PATHOGENIC_THRESHOLD).astype(int)

        rows.append({
            "Mapped_Functional_Category": category,
            "n": len(sub),
            "n_benign": int((sub["Binary_Label"] == 0).sum()),
            "n_pathogenic": int((sub["Binary_Label"] == 1).sum()),
            "mean_score": float(sub["AlphaMissense_Score"].mean()),
            "median_score": float(sub["AlphaMissense_Score"].median()),
            "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
            "average_precision": float(ap) if not np.isnan(ap) else np.nan,
            "accuracy_at_0_564": float(accuracy_score(sub["Binary_Label"], pred)),
            "precision_at_0_564": float(precision_score(sub["Binary_Label"], pred, zero_division=0)),
            "recall_at_0_564": float(recall_score(sub["Binary_Label"], pred, zero_division=0)),
            "f1_at_0_564": float(f1_score(sub["Binary_Label"], pred, zero_division=0)),
        })

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def main():
    print("Reading:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nInput rows:", len(df))

    df_eval = df.dropna(subset=["AlphaMissense_Score", "Binary_Label"]).copy()

    df_eval["Binary_Label"] = df_eval["Binary_Label"].astype(int)
    df_eval["AlphaMissense_Score"] = df_eval["AlphaMissense_Score"].astype(float)

    print("Rows with AlphaMissense score:", len(df_eval))
    print("Dropped rows:", len(df) - len(df_eval))

    y_true = df_eval["Binary_Label"].values
    scores = df_eval["AlphaMissense_Score"].values

    official_metrics = compute_metrics(
        y_true,
        scores,
        threshold=ALPHAMISSENSE_PATHOGENIC_THRESHOLD,
    )

    sweep = threshold_sweep(y_true, scores)
    sweep.to_csv(THRESHOLD_SWEEP_FILE, index=False)

    best_row = sweep.sort_values("f1", ascending=False).iloc[0]

    optimized_metrics = compute_metrics(
        y_true,
        scores,
        threshold=float(best_row["threshold"]),
    )

    cat_summary = category_summary(df_eval)
    cat_summary.to_csv(CATEGORY_SUMMARY_FILE, index=False)

    df_eval["AlphaMissense_Pred_OfficialThreshold"] = (
        df_eval["AlphaMissense_Score"] >= ALPHAMISSENSE_PATHOGENIC_THRESHOLD
    ).astype(int)

    df_eval["AlphaMissense_Pred_OptimizedF1Threshold"] = (
        df_eval["AlphaMissense_Score"] >= float(best_row["threshold"])
    ).astype(int)

    df_eval.to_csv(PREDICTIONS_FILE, index=False)

    score_by_label = (
        df_eval
        .groupby("Label_Name")["AlphaMissense_Score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    metrics = {
        "input_file": str(INPUT_FILE),
        "n_total_rows": int(len(df)),
        "n_evaluated_rows": int(len(df_eval)),
        "n_dropped_missing_alphamissense": int(len(df) - len(df_eval)),
        "official_threshold": ALPHAMISSENSE_PATHOGENIC_THRESHOLD,
        "official_threshold_metrics": official_metrics,
        "best_f1_threshold": float(best_row["threshold"]),
        "optimized_f1_threshold_metrics": optimized_metrics,
        "score_by_label": score_by_label.to_dict(orient="records"),
        "output_predictions_file": str(PREDICTIONS_FILE),
        "output_category_summary_file": str(CATEGORY_SUMMARY_FILE),
        "output_threshold_sweep_file": str(THRESHOLD_SWEEP_FILE),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 100)
    print("ALPHAMISSENSE OFFICIAL THRESHOLD METRICS")
    print("=" * 100)
    print(json.dumps(official_metrics, indent=4))

    print("\n" + "=" * 100)
    print("ALPHAMISSENSE OPTIMIZED F1 THRESHOLD METRICS")
    print("=" * 100)
    print(json.dumps(optimized_metrics, indent=4))

    print("\n" + "=" * 100)
    print("SCORE BY LABEL")
    print("=" * 100)
    print(score_by_label.to_string(index=False))

    print("\n" + "=" * 100)
    print("FUNCTIONAL CATEGORY SUMMARY")
    print("=" * 100)
    print(cat_summary.to_string(index=False))

    print("\nSaved metrics:")
    print(METRICS_FILE)


if __name__ == "__main__":
    main()