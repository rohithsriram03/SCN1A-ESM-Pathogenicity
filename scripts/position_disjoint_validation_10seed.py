from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
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
OUTPUT_DIR = PROJECT_ROOT / "results" / "position_disjoint_validation_10seed"
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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

            nn.Linear(32, 2),
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


def make_position_disjoint_split(y, groups, seed):
    """
    Outer split:
    train_val vs test, with AA positions disjoint.

    Inner split:
    train vs validation, also with AA positions disjoint.
    """

    indices = np.arange(len(y))

    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    train_val_idx, test_idx = next(
        outer_cv.split(indices, y, groups=groups)
    )

    inner_y = y[train_val_idx]
    inner_groups = groups[train_val_idx]
    inner_indices = np.arange(len(train_val_idx))

    inner_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    inner_train_relative, val_relative = next(
        inner_cv.split(inner_indices, inner_y, groups=inner_groups)
    )

    train_idx = train_val_idx[inner_train_relative]
    val_idx = train_val_idx[val_relative]

    return train_idx, val_idx, test_idx


def verify_position_disjoint(metadata, train_idx, val_idx, test_idx):
    train_positions = set(metadata.iloc[train_idx]["AA_Position"].astype(int))
    val_positions = set(metadata.iloc[val_idx]["AA_Position"].astype(int))
    test_positions = set(metadata.iloc[test_idx]["AA_Position"].astype(int))

    train_test_overlap = train_positions.intersection(test_positions)
    train_val_overlap = train_positions.intersection(val_positions)
    val_test_overlap = val_positions.intersection(test_positions)

    print("Unique train positions:", len(train_positions))
    print("Unique val positions:", len(val_positions))
    print("Unique test positions:", len(test_positions))
    print("Train/test position overlap:", len(train_test_overlap))
    print("Train/val position overlap:", len(train_val_overlap))
    print("Val/test position overlap:", len(val_test_overlap))

    if train_test_overlap or train_val_overlap or val_test_overlap:
        raise ValueError("Position leakage detected between splits.")


def prepare_scaled_data(X, y, train_idx, val_idx, test_idx):
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


def train_logistic_baseline(X, y, train_idx, val_idx, test_idx, seed):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_scaled_data(
        X, y, train_idx, val_idx, test_idx
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
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


def train_custom_nn(X, y, train_idx, val_idx, test_idx, seed, feature_name, device):
    set_seed(seed)

    X_train, X_val, X_test, y_train, y_val, y_test = prepare_scaled_data(
        X, y, train_idx, val_idx, test_idx
    )

    train_dataset = VariantDataset(X_train, y_train)
    val_dataset = VariantDataset(X_val, y_val)
    test_dataset = VariantDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_counts = np.bincount(y_train)
    total = class_counts.sum()

    class_weights = torch.tensor(
        [
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1]),
        ],
        dtype=torch.float32,
    ).to(device)

    model = CustomMutationNN(input_dim=X.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = -1
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

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
    metrics_optimized = calculate_metrics(
        test_labels,
        test_probs,
        threshold=best_threshold,
    )

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
    groups = metadata["AA_Position"].astype(int).values

    wt_site = np.load(INPUT_DIR / "wt_site_embeddings.npy")
    mut_site = np.load(INPUT_DIR / "mut_site_embeddings.npy")
    delta_site = np.load(INPUT_DIR / "delta_site_embeddings.npy")
    abs_delta_site = np.load(INPUT_DIR / "abs_delta_site_embeddings.npy")

    wt_plus_mut_site = np.concatenate([wt_site, mut_site], axis=1)

    site_full = np.concatenate(
        [wt_site, mut_site, delta_site, abs_delta_site],
        axis=1,
    )

    feature_sets = {
    "nn_wt_plus_mut_site": wt_plus_mut_site,
}

    all_results = []
    split_records = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"POSITION-DISJOINT SEED {seed}")
        print("=" * 90)

        train_idx, val_idx, test_idx = make_position_disjoint_split(
            y=y,
            groups=groups,
            seed=seed,
        )

        verify_position_disjoint(metadata, train_idx, val_idx, test_idx)

        split_records.append({
            "seed": seed,
            "n_train_variants": int(len(train_idx)),
            "n_val_variants": int(len(val_idx)),
            "n_test_variants": int(len(test_idx)),
            "n_train_positions": int(metadata.iloc[train_idx]["AA_Position"].nunique()),
            "n_val_positions": int(metadata.iloc[val_idx]["AA_Position"].nunique()),
            "n_test_positions": int(metadata.iloc[test_idx]["AA_Position"].nunique()),
        })

        print("\nTraining Logistic Regression baseline...")
        baseline_result = train_logistic_baseline(
            X=mut_site,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
        )
        all_results.append(baseline_result)

        print(
            f"Baseline | "
            f"AUC={baseline_result['test_auc']:.4f}, "
            f"F1={baseline_result['test_f1_0_5']:.4f}"
        )

        for feature_name, X in feature_sets.items():
            print(f"\nTraining custom NN: {feature_name} | shape={X.shape}")

            result = train_custom_nn(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=seed,
                feature_name=feature_name,
                device=device,
            )

            all_results.append(result)

            print(
                f"{feature_name} | "
                f"AUC={result['test_auc']:.4f}, "
                f"F1={result['test_f1_0_5']:.4f}, "
                f"Optimized F1={result['test_f1_optimized']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "position_disjoint_per_seed_results.csv", index=False)

    split_df = pd.DataFrame(split_records)
    split_df.to_csv(OUTPUT_DIR / "position_disjoint_split_summary.csv", index=False)

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

    aggregate.to_csv(
        OUTPUT_DIR / "position_disjoint_aggregate_results.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("POSITION-DISJOINT AGGREGATE RESULTS")
    print("=" * 90)
    print(aggregate.to_string(index=False))

    with open(OUTPUT_DIR / "position_disjoint_config.json", "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "split_group": "AA_Position",
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            f,
            indent=4,
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()