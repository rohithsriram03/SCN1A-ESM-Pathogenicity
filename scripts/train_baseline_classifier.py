from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "results" / "embeddings" / "mutation_site_embeddings.npy"
METADATA_FILE = PROJECT_ROOT / "results" / "embeddings" / "embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_model"
OUTPUT_DIR.mkdir(exist_ok=True)

X = np.load(EMBEDDINGS_FILE)
metadata = pd.read_csv(METADATA_FILE)

y = metadata["Binary_Label"].astype(int).values

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)
print("Class counts:")
print(pd.Series(y).value_counts())

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X,
    y,
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    solver="liblinear",
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_prob),
}

print("\nMetrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

results = meta_test.copy()
results["True_Label"] = y_test
results["Predicted_Label"] = y_pred
results["Pathogenic_Probability"] = y_prob
results["Confidence"] = np.maximum(y_prob, 1 - y_prob)
results["Uncertainty"] = 1 - results["Confidence"]

results.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)

with open(OUTPUT_DIR / "baseline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\nSaved predictions to:", OUTPUT_DIR / "baseline_predictions.csv")
print("Saved metrics to:", OUTPUT_DIR / "baseline_metrics.json")