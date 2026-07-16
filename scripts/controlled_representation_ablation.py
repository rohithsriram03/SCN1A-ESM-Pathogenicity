from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from scipy.stats import ttest_rel, wilcoxon

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
OUTPUT_DIR = PROJECT_ROOT / "results" / "controlled_representation_ablation"
OUTPUT_DIR.mkdir(exist_ok=True)

SEEDS = list(range(10))

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
    if torch.cuda.is_available():
        return torch.device("cuda")
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

    if train_positions.intersection(val_positions):
        raise ValueError("Train/validation position overlap detected.")

    if train_positions.intersection(test_positions):
        raise ValueError("Train/test position overlap detected.")

    if val_positions.intersection(test_positions):
        raise ValueError("Validation/test position overlap detected.")


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


def train_logistic_regression(X, y, train_idx, val_idx, test_idx, seed, feature_name):
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
        "model_family": "logistic_regression",
        "feature_set": feature_name,
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
    metrics_optimized = calculate_metrics(test_labels, test_probs, threshold=best_threshold)

    return {
        "seed": seed,
        "model_family": "custom_nn",
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


def build_biochemical_features(metadata):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    hydropathy = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }

    molecular_weight = {
        "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
        "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
        "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
        "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
    }

    charge = {
        "D": -1, "E": -1,
        "K": 1, "R": 1, "H": 1,
    }

    polar = set(["S", "T", "N", "Q", "C", "Y", "D", "E", "K", "R", "H"])
    aromatic = set(["F", "W", "Y", "H"])
    sulfur = set(["C", "M"])
    special = set(["G", "P"])

    rows = []

    for _, row in metadata.iterrows():
        ref = str(row["Ref_AA"]).strip().upper()
        alt = str(row["Alt_AA"]).strip().upper()

        ref_hydro = hydropathy.get(ref, 0.0)
        alt_hydro = hydropathy.get(alt, 0.0)

        ref_mw = molecular_weight.get(ref, 0.0)
        alt_mw = molecular_weight.get(alt, 0.0)

        ref_charge = charge.get(ref, 0)
        alt_charge = charge.get(alt, 0)

        ref_polar = int(ref in polar)
        alt_polar = int(alt in polar)

        ref_aromatic = int(ref in aromatic)
        alt_aromatic = int(alt in aromatic)

        ref_sulfur = int(ref in sulfur)
        alt_sulfur = int(alt in sulfur)

        ref_special = int(ref in special)
        alt_special = int(alt in special)

        features = {
            "ref_hydropathy": ref_hydro,
            "alt_hydropathy": alt_hydro,
            "delta_hydropathy": alt_hydro - ref_hydro,
            "abs_delta_hydropathy": abs(alt_hydro - ref_hydro),

            "ref_molecular_weight": ref_mw,
            "alt_molecular_weight": alt_mw,
            "delta_molecular_weight": alt_mw - ref_mw,
            "abs_delta_molecular_weight": abs(alt_mw - ref_mw),

            "ref_charge": ref_charge,
            "alt_charge": alt_charge,
            "delta_charge": alt_charge - ref_charge,
            "abs_delta_charge": abs(alt_charge - ref_charge),

            "ref_polar": ref_polar,
            "alt_polar": alt_polar,
            "polarity_changed": int(ref_polar != alt_polar),

            "ref_aromatic": ref_aromatic,
            "alt_aromatic": alt_aromatic,
            "aromaticity_changed": int(ref_aromatic != alt_aromatic),

            "ref_sulfur": ref_sulfur,
            "alt_sulfur": alt_sulfur,
            "sulfur_status_changed": int(ref_sulfur != alt_sulfur),

            "ref_special_gly_pro": ref_special,
            "alt_special_gly_pro": alt_special,
            "special_status_changed": int(ref_special != alt_special),
        }

        for aa in amino_acids:
            features[f"ref_is_{aa}"] = int(ref == aa)
            features[f"alt_is_{aa}"] = int(alt == aa)

        rows.append(features)

    feature_df = pd.DataFrame(rows)

    feature_df.to_csv(
        OUTPUT_DIR / "biochemical_substitution_features.csv",
        index=False,
    )

    return feature_df.values.astype(float), list(feature_df.columns)


def summarize_results(results_df):
    aggregate = (
        results_df
        .groupby(["model_family", "feature_set", "input_dim"])
        .agg(
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),

            test_f1_0_5_mean=("test_f1_0_5", "mean"),
            test_f1_0_5_std=("test_f1_0_5", "std"),

            test_f1_optimized_mean=("test_f1_optimized", "mean"),
            test_f1_optimized_std=("test_f1_optimized", "std"),

            test_accuracy_0_5_mean=("test_accuracy_0_5", "mean"),
            test_accuracy_0_5_std=("test_accuracy_0_5", "std"),

            test_recall_0_5_mean=("test_recall_0_5", "mean"),
            test_recall_0_5_std=("test_recall_0_5", "std"),

            test_precision_0_5_mean=("test_precision_0_5", "mean"),
            test_precision_0_5_std=("test_precision_0_5", "std"),
        )
        .reset_index()
        .sort_values(["model_family", "test_auc_mean"], ascending=[True, False])
    )

    return aggregate


def paired_comparison(results_df, model_family, feature_a, feature_b, metric):
    a = results_df[
        (results_df["model_family"] == model_family)
        & (results_df["feature_set"] == feature_a)
    ][["seed", metric]].rename(columns={metric: "a"})

    b = results_df[
        (results_df["model_family"] == model_family)
        & (results_df["feature_set"] == feature_b)
    ][["seed", metric]].rename(columns={metric: "b"})

    merged = a.merge(b, on="seed", how="inner").sort_values("seed")

    diff = merged["b"] - merged["a"]

    t_stat, t_p = ttest_rel(merged["b"], merged["a"])

    try:
        w_stat, w_p = wilcoxon(merged["b"], merged["a"])
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    return {
        "model_family": model_family,
        "metric": metric,
        "comparison": f"{feature_b} minus {feature_a}",
        "feature_a": feature_a,
        "feature_b": feature_b,
        "feature_a_mean": float(merged["a"].mean()),
        "feature_b_mean": float(merged["b"].mean()),
        "mean_difference_b_minus_a": float(diff.mean()),
        "std_difference": float(diff.std()),
        "paired_t_p_value": float(t_p),
        "wilcoxon_p_value": float(w_p) if not np.isnan(w_p) else np.nan,
        "n_pairs": int(len(merged)),
    }


def main():
    device = get_device()
    print("Using device:", device)

    metadata = pd.read_csv(INPUT_DIR / "paired_embedding_metadata.csv")

    y = metadata["Binary_Label"].astype(int).values
    groups = metadata["AA_Position"].astype(int).values

    wt_site = np.load(INPUT_DIR / "wt_site_embeddings.npy")
    mut_site = np.load(INPUT_DIR / "mut_site_embeddings.npy")

    biochem_features, biochem_columns = build_biochemical_features(metadata)

    print("WT site shape:", wt_site.shape)
    print("Mutant site shape:", mut_site.shape)
    print("Biochemical feature shape:", biochem_features.shape)

    with open(OUTPUT_DIR / "biochemical_feature_columns.json", "w") as f:
        json.dump(biochem_columns, f, indent=4)

    feature_sets = {
        "wt_site_only": wt_site,
        "mutant_site_only": mut_site,
        "wt_plus_mut_site": np.concatenate([wt_site, mut_site], axis=1),
        "biochem_only": biochem_features,
        "wt_plus_mut_plus_biochem": np.concatenate(
            [wt_site, mut_site, biochem_features],
            axis=1,
        ),
    }

    all_results = []

    for seed in SEEDS:
        print("\n" + "=" * 100)
        print(f"POSITION-DISJOINT SEED {seed}")
        print("=" * 100)

        train_idx, val_idx, test_idx = make_position_disjoint_split(
            y=y,
            groups=groups,
            seed=seed,
        )

        verify_position_disjoint(metadata, train_idx, val_idx, test_idx)

        for feature_name, X in feature_sets.items():
            print(f"\nLogistic Regression | {feature_name} | shape={X.shape}")

            lr_result = train_logistic_regression(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=seed,
                feature_name=feature_name,
            )

            all_results.append(lr_result)

            print(
                f"LR {feature_name}: "
                f"AUC={lr_result['test_auc']:.4f}, "
                f"F1={lr_result['test_f1_0_5']:.4f}"
            )

        for feature_name, X in feature_sets.items():
            print(f"\nCustom NN | {feature_name} | shape={X.shape}")

            nn_result = train_custom_nn(
                X=X,
                y=y,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                seed=seed,
                feature_name=feature_name,
                device=device,
            )

            all_results.append(nn_result)

            print(
                f"NN {feature_name}: "
                f"AUC={nn_result['test_auc']:.4f}, "
                f"F1={nn_result['test_f1_0_5']:.4f}, "
                f"optimized F1={nn_result['test_f1_optimized']:.4f}"
            )

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "controlled_ablation_per_seed_results.csv", index=False)

    aggregate = summarize_results(results_df)
    aggregate.to_csv(OUTPUT_DIR / "controlled_ablation_aggregate_results.csv", index=False)

    comparison_rows = []

    comparisons = [
        ("wt_site_only", "wt_plus_mut_site"),
        ("mutant_site_only", "wt_plus_mut_site"),
        ("wt_plus_mut_site", "wt_plus_mut_plus_biochem"),
        ("biochem_only", "wt_plus_mut_plus_biochem"),
    ]

    metrics = [
        "test_auc",
        "test_f1_0_5",
        "test_f1_optimized",
        "test_recall_0_5",
        "test_accuracy_0_5",
    ]

    for model_family in ["logistic_regression", "custom_nn"]:
        for feature_a, feature_b in comparisons:
            for metric in metrics:
                comparison_rows.append(
                    paired_comparison(
                        results_df=results_df,
                        model_family=model_family,
                        feature_a=feature_a,
                        feature_b=feature_b,
                        metric=metric,
                    )
                )

    comparisons_df = pd.DataFrame(comparison_rows)
    comparisons_df.to_csv(
        OUTPUT_DIR / "controlled_ablation_paired_comparisons.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("CONTROLLED REPRESENTATION ABLATION: AGGREGATE RESULTS")
    print("=" * 100)
    print(aggregate.to_string(index=False))

    print("\n" + "=" * 100)
    print("KEY PAIRED COMPARISONS")
    print("=" * 100)
    key = comparisons_df[
        (comparisons_df["metric"].isin(["test_auc", "test_f1_0_5"]))
        & (
            comparisons_df["comparison"].isin([
                "wt_plus_mut_site minus wt_site_only",
                "wt_plus_mut_site minus mutant_site_only",
                "wt_plus_mut_plus_biochem minus wt_plus_mut_site",
            ])
        )
    ]

    print(key.to_string(index=False))

    with open(OUTPUT_DIR / "controlled_ablation_config.json", "w") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "validation": "10 repeated position-disjoint splits grouped by AA_Position",
                "feature_sets": list(feature_sets.keys()),
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "note": "Domain category and AA_Position are intentionally excluded as model inputs to avoid position/domain leakage.",
            },
            f,
            indent=4,
        )

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()