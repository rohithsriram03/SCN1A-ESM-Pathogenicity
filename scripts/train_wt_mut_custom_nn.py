from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import joblib


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "custom_wt_mut_nn"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class VariantDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class WTMutDeltaNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def predict_probs(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return np.array(all_probs), np.array(all_labels)


def calculate_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        metrics = calculate_metrics(y_true, probs, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold

    return best_threshold


def train_one_feature_set(feature_name, X, metadata, device):
    print("\n" + "=" * 80)
    print(f"Training feature set: {feature_name}")
    print("Feature shape:", X.shape)
    print("=" * 80)

    y = metadata["Binary_Label"].astype(int).values
    indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y[train_val_idx]
    )

    X_train = X[train_idx]
    X_val = X[val_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = VariantDataset(X_train, y_train)
    val_dataset = VariantDataset(X_val, y_val)
    test_dataset = VariantDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_counts = np.bincount(y_train)
    total = class_counts.sum()

    class_weights = torch.tensor(
        [
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1])
        ],
        dtype=torch.float32
    ).to(device)

    print("Train class counts:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    model = WTMutDeltaNN(input_dim=X.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_auc = -1
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_probs, val_labels = predict_probs(model, val_loader, device)
        val_metrics = calculate_metrics(val_labels, val_probs, threshold=0.5)

        avg_train_loss = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            **val_metrics
        })

        print(
            f"Epoch {epoch:03d} | "
            f"loss={avg_train_loss:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    model.load_state_dict(best_model_state)

    val_probs, val_labels = predict_probs(model, val_loader, device)
    best_threshold = find_best_threshold(val_labels, val_probs)

    test_probs, test_labels = predict_probs(model, test_loader, device)

    test_metrics_default = calculate_metrics(
        test_labels,
        test_probs,
        threshold=0.5
    )

    test_metrics_best_threshold = calculate_metrics(
        test_labels,
        test_probs,
        threshold=best_threshold
    )

    print("\nTest metrics at threshold 0.5:")
    print(test_metrics_default)

    print("\nBest threshold chosen on validation set:", best_threshold)
    print("Test metrics at validation-optimized threshold:")
    print(test_metrics_best_threshold)

    feature_output_dir = OUTPUT_DIR / feature_name
    feature_output_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), feature_output_dir / "model.pt")
    joblib.dump(scaler, feature_output_dir / "scaler.joblib")

    pd.DataFrame(history).to_csv(
        feature_output_dir / "training_history.csv",
        index=False
    )

    test_metadata = metadata.iloc[test_idx].copy()
    test_metadata["Pathogenic_Probability"] = test_probs
    test_metadata["Predicted_Label_0_5"] = (test_probs >= 0.5).astype(int)
    test_metadata["Predicted_Label_Best_Threshold"] = (
        test_probs >= best_threshold
    ).astype(int)
    test_metadata["Confidence"] = np.maximum(test_probs, 1 - test_probs)
    test_metadata["Uncertainty"] = 1 - test_metadata["Confidence"]

    test_metadata.to_csv(
        feature_output_dir / "custom_nn_predictions.csv",
        index=False
    )

    final_results = {
        "feature_set": feature_name,
        "input_dim": int(X.shape[1]),
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_val_auc),
        "best_validation_threshold_for_f1": float(best_threshold),
        "test_metrics_threshold_0_5": test_metrics_default,
        "test_metrics_validation_optimized_threshold": test_metrics_best_threshold,
    }

    with open(feature_output_dir / "custom_nn_metrics.json", "w") as f:
        json.dump(final_results, f, indent=4)

    return final_results


def main():
    set_seed(RANDOM_STATE)

    device = get_device()
    print("Using device:", device)

    metadata = pd.read_csv(INPUT_DIR / "paired_embedding_metadata.csv")

    site_features = np.load(INPUT_DIR / "site_wt_mut_delta_abs_features.npy")
    mean_features = np.load(INPUT_DIR / "mean_wt_mut_delta_abs_features.npy")

    site_plus_mean_features = np.concatenate(
        [site_features, mean_features],
        axis=1
    )

    feature_sets = {
        "site_wt_mut_delta_abs": site_features,
        "mean_wt_mut_delta_abs": mean_features,
        "site_plus_mean_wt_mut_delta_abs": site_plus_mean_features,
    }

    all_results = []

    for feature_name, X in feature_sets.items():
        result = train_one_feature_set(
            feature_name=feature_name,
            X=X,
            metadata=metadata,
            device=device
        )
        all_results.append(result)

    summary_rows = []

    for result in all_results:
        default_metrics = result["test_metrics_threshold_0_5"]
        optimized_metrics = result["test_metrics_validation_optimized_threshold"]

        summary_rows.append({
            "feature_set": result["feature_set"],
            "input_dim": result["input_dim"],
            "best_epoch": result["best_epoch"],
            "best_validation_roc_auc": result["best_validation_roc_auc"],

            "test_auc": default_metrics["roc_auc"],
            "test_accuracy_0_5": default_metrics["accuracy"],
            "test_precision_0_5": default_metrics["precision"],
            "test_recall_0_5": default_metrics["recall"],
            "test_f1_0_5": default_metrics["f1"],

            "best_validation_threshold": result["best_validation_threshold_for_f1"],
            "test_accuracy_optimized_threshold": optimized_metrics["accuracy"],
            "test_precision_optimized_threshold": optimized_metrics["precision"],
            "test_recall_optimized_threshold": optimized_metrics["recall"],
            "test_f1_optimized_threshold": optimized_metrics["f1"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "custom_nn_summary.csv", index=False)

    print("\n" + "=" * 80)
    print("CUSTOM NN SUMMARY")
    print("=" * 80)
    print(summary_df)

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()