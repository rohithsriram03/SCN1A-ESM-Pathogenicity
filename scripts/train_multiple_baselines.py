from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = PROJECT_ROOT / "results" / "embeddings" / "mutation_site_embeddings.npy"
METADATA_FILE = PROJECT_ROOT / "results" / "embeddings" / "embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "multi_baseline_models"
OUTPUT_DIR.mkdir(exist_ok=True)

X = np.load(EMBEDDINGS_FILE)
metadata = pd.read_csv(METADATA_FILE)
y = metadata["Binary_Label"].astype(int).values

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X,
    y,
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    #     "xgboost": XGBClassifier(
    #         n_estimators=500,
    #         max_depth=4,
    #         learning_rate=0.03,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         eval_metric="logloss",
    #         random_state=42
    # )
}

metrics_rows = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    metrics_rows.append(metrics)

    results = meta_test.copy()
    results["Model"] = model_name
    results["True_Label"] = y_test
    results["Predicted_Label"] = y_pred
    results["Pathogenic_Probability"] = y_prob
    results["Confidence"] = np.maximum(y_prob, 1 - y_prob)
    results["Uncertainty"] = 1 - results["Confidence"]

    results.to_csv(OUTPUT_DIR / f"{model_name}_predictions.csv", index=False)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

print("\nModel comparison:")
print(metrics_df)