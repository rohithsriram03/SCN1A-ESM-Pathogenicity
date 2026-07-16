import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Standard binary metrics from pathogenic-class probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def prediction_frame(meta: pd.DataFrame, y_true, y_prob, threshold: float = 0.5) -> pd.DataFrame:
    """Attach labels, probability, confidence, and uncertainty to a metadata frame.

    This is the canonical prediction-CSV schema every model in the repo writes.
    """
    prob = np.asarray(y_prob)
    out = meta.copy()
    out["True_Label"] = np.asarray(y_true)
    out["Predicted_Label"] = (prob >= threshold).astype(int)
    out["Pathogenic_Probability"] = prob
    out["Confidence"] = np.maximum(prob, 1 - prob)
    out["Uncertainty"] = 1 - out["Confidence"]
    return out
