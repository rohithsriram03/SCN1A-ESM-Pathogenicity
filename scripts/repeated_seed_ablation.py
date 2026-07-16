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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "repeated_seed_ablation"
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class VariantDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CustomMutationNN(nn.Module):
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

    return float(best_threshold)


def predict_nn_probs(model, loader, device):
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


def split_data(X, y, seed):
    indices = np.arange(len(y))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=seed,
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_logistic_regression_baseline(X, y, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed
    )

    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_threshold(y_val, val_probs)

    test_probs = model.predict_proba(X_test)[:, 1]

    metrics_0_5 = calculate_metrics(y_test, test_probs, threshold=0.5)
    metrics_optimized = calculate_metrics(y_test, test_probs, threshold=best_threshold)

    return {
        "seed": seed,
        "model": "logistic_regression",
        "feature_set": "mutant_site_baseline",
        "input_dim": int(X.shape[1]),
        "best_validation_threshold": best_threshold,

        "test_auc": metrics_0_5["roc_auc"],
        "test_accuracy_0_5": metrics_0_5["accuracy"],
        "test_precision_0_5": metrics_0_5["precision"],
        "test_recall_0_5": metrics_0_5["recall"],
        "test_f1_0_5": metrics_0_5["f1"],

        "test_accuracy_optimized": metrics_optimized["accuracy"],
        "test_precision_optimized": metrics_optimized["precision"],
        "test_recall_optimized": metrics_optimized["recall"],
        "test_f1_optimized": metrics_optimized["f1"],
    }


def train_custom_nn(X, y, seed, feature_name, device):
    set_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)

    train_dataset = VariantDataset(X_train, y_train)
    val_dataset = VariantDataset(X_val, y_val)
    test_dataset = VariantDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
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

    model = CustomMutationNN(input_dim=X.shape[1]).to(device)

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

        val_probs, val_labels = predict_nn_probs(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_model_state)
    model.to(device)

    val_probs, val_labels = predict_nn_probs(model, val_loader, device)
    best_threshold = find_best_threshold(val_labels, val_probs)

    test_probs, test_labels = predict_nn_probs(model, test_loader, device)

    metrics_0_5 = calculate_metrics(test_labels, test_probs, threshold=0.5)
    metrics_optimized = calculate_metrics(test_labels, test_probs, threshold=best_threshold)

    return {
        "seed": seed,
        "model": "custom_nn",
        "feature_set": feature_name,
        "input_dim": int(X.shape[1]),
        "best_epoch": int(best_epoch),
        "best_validation_auc": float(best_val_auc),
        "best_validation_threshold": best_threshold,

        "test_auc": metrics_0_5["roc_auc"],
        "test_accuracy_0_5": metrics_0_5["accuracy"],
        "test_precision_0_5": metrics_0_5["precision"],
        "test_recall_0_5": metrics_0_5["recall"],
        "test_f1_0_5": metrics_0_5["f1"],

        "test_accuracy_optimized": metrics_optimized["accuracy"],
        "test_precision_optimized": metrics_optimized["precision"],
        "test_recall_optimized": metrics_optimized["recall"],
        "test_f1_optimized": metrics_optimized["f1"],
    }


def main():
    device = get_device()
    print("Using device:", device)

    metadata = pd.read_csv(INPUT_DIR / "paired_embedding_metadata.csv")
    y = metadata["Binary_Label"].astype(int).values

    wt_site = np.load(INPUT_DIR / "wt_site_embeddings.npy")
    mut_site = np.load(INPUT_DIR / "mut_site_embeddings.npy")
    delta_site = np.load(INPUT_DIR / "delta_site_embeddings.npy")
    abs_delta_site = np.load(INPUT_DIR / "abs_delta_site_embeddings.npy")

    wt_mean = np.load(INPUT_DIR / "wt_mean_embeddings.npy")
    mut_mean = np.load(INPUT_DIR / "mut_mean_embeddings.npy")
    delta_mean = np.load(INPUT_DIR / "delta_mean_embeddings.npy")
    abs_delta_mean = np.load(INPUT_DIR / "abs_delta_mean_embeddings.npy")

    site_full = np.concatenate(
        [wt_site, mut_site, delta_site, abs_delta_site],
        axis=1
    )

    mean_full = np.concatenate(
        [wt_mean, mut_mean, delta_mean, abs_delta_mean],
        axis=1
    )

    site_plus_mean_full = np.concatenate(
        [site_full, mean_full],
        axis=1
    )

    feature_sets = {
        "nn_mutant_site_only": mut_site,
        "nn_wildtype_site_only": wt_site,
        "nn_delta_site_only": delta_site,
        "nn_abs_delta_site_only": abs_delta_site,
        "nn_wt_plus_mut_site": np.concatenate([wt_site, mut_site], axis=1),
        "nn_mut_plus_delta_site": np.concatenate([mut_site, delta_site], axis=1),
        "nn_site_full_wt_mut_delta_abs": site_full,
        "nn_mean_full_wt_mut_delta_abs": mean_full,
        "nn_site_plus_mean_full": site_plus_mean_full,
    }

    all_results = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"SEED {seed}")
        print("=" * 90)

        print("Training Logistic Regression baseline...")
        baseline_result = train_logistic_regression_baseline(
            X=mut_site,
            y=y,
            seed=seed
        )
        all_results.append(baseline_result)

        print(
            f"Baseline AUC={baseline_result['test_auc']:.4f}, "
            f"F1={baseline_result['test_f1_0_5']:.4f}"
        )

        for feature_name, X in feature_sets.items():
            print(f"\nTraining custom NN: {feature_name} | shape={X.shape}")

            result = train_custom_nn(
                X=X,
                y=y,
                seed=seed,
                feature_name=feature_name,
                device=device
            )

            all_results.append(result)

            print(
                f"{feature_name} | "
                f"AUC={result['test_auc']:.4f}, "
                f"F1={result['test_f1_0_5']:.4f}, "
                f"Optimized F1={result['test_f1_optimized']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "per_seed_results.csv", index=False)

    aggregate = (
        results_df
        .groupby(["model", "feature_set", "input_dim"])
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),

            test_accuracy_0_5_mean=("test_accuracy_0_5", "mean"),
            test_accuracy_0_5_std=("test_accuracy_0_5", "std"),

            test_f1_0_5_mean=("test_f1_0_5", "mean"),
            test_f1_0_5_std=("test_f1_0_5", "std"),

            test_f1_optimized_mean=("test_f1_optimized", "mean"),
            test_f1_optimized_std=("test_f1_optimized", "std"),

            test_recall_0_5_mean=("test_recall_0_5", "mean"),
            test_precision_0_5_mean=("test_precision_0_5", "mean"),
        )
        .reset_index()
        .sort_values(by="test_auc_mean", ascending=False)
    )

    aggregate.to_csv(OUTPUT_DIR / "aggregate_mean_std_results.csv", index=False)

    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS")
    print("=" * 90)
    print(aggregate.to_string(index=False))

    with open(OUTPUT_DIR / "experiment_config.json", "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            f,
            indent=4
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()