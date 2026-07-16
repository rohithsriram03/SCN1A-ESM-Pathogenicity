from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from scipy.stats import kruskal
import scikit_posthocs as sp

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "custom_nn_domain_uncertainty"
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

    if train_positions.intersection(test_positions):
        raise ValueError("Train/test position leakage detected.")

    if train_positions.intersection(val_positions):
        raise ValueError("Train/validation position leakage detected.")

    if val_positions.intersection(test_positions):
        raise ValueError("Validation/test position leakage detected.")


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


def train_custom_nn_and_predict(X, y, train_idx, val_idx, test_idx, seed, device):
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

        val_probs, val_labels = predict_probs(model, val_loader, device)

        # Calculate validation AUC manually to avoid extra imports here.
        # Import locally.
        from sklearn.metrics import roc_auc_score
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

    test_probs, test_labels = predict_probs(model, test_loader, device)

    return test_probs, test_labels, best_epoch, best_val_auc


def summarize_uncertainty_by_domain(df, output_prefix):
    """
    Runs Kruskal-Wallis and Dunn's post hoc test on uncertainty by functional category.
    """

    required_col = "Mapped_Functional_Category"
    if required_col not in df.columns:
        raise ValueError(f"Missing required column: {required_col}")

    clean_df = df.dropna(
        subset=["Mapped_Functional_Category", "Uncertainty", "Correct"]
    ).copy()

    categories = sorted(clean_df["Mapped_Functional_Category"].unique())

    summary_rows = []

    for category in categories:
        sub = clean_df[clean_df["Mapped_Functional_Category"] == category]

        summary_rows.append({
            "Mapped_Functional_Category": category,
            "n": int(len(sub)),
            "mean_uncertainty": float(sub["Uncertainty"].mean()),
            "median_uncertainty": float(sub["Uncertainty"].median()),
            "std_uncertainty": float(sub["Uncertainty"].std()),
            "mean_confidence": float(sub["Confidence"].mean()),
            "accuracy": float(sub["Correct"].mean()),
            "mean_pathogenic_probability": float(sub["Pathogenic_Probability"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_uncertainty",
        ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / f"{output_prefix}_uncertainty_by_domain_summary.csv",
        index=False,
    )

    groups = [
        clean_df.loc[
            clean_df["Mapped_Functional_Category"] == category,
            "Uncertainty"
        ].values
        for category in categories
    ]

    # Only include groups with at least 2 values.
    valid_categories = []
    valid_groups = []

    for category, values in zip(categories, groups):
        if len(values) >= 2:
            valid_categories.append(category)
            valid_groups.append(values)

    h_stat, p_value = kruskal(*valid_groups)

    kruskal_results = {
        "test": "Kruskal-Wallis",
        "n_categories": len(valid_categories),
        "categories": valid_categories,
        "H_statistic": float(h_stat),
        "p_value": float(p_value),
        "input_file_type": output_prefix,
    }

    with open(OUTPUT_DIR / f"{output_prefix}_kruskal_wallis_results.json", "w") as f:
        json.dump(kruskal_results, f, indent=4)

    dunn = sp.posthoc_dunn(
        clean_df,
        val_col="Uncertainty",
        group_col="Mapped_Functional_Category",
        p_adjust="bonferroni",
    )

    dunn.to_csv(
        OUTPUT_DIR / f"{output_prefix}_dunn_posthoc_bonferroni.csv"
    )

    # Boxplot
    plt.figure(figsize=(8, 5))

    plot_data = [
        clean_df.loc[
            clean_df["Mapped_Functional_Category"] == category,
            "Uncertainty"
        ].values
        for category in valid_categories
    ]

    plt.boxplot(plot_data, labels=valid_categories, showfliers=True)
    plt.ylabel("Prediction uncertainty")
    plt.xlabel("Functional region")
    plt.title("Custom NN uncertainty by SCN1A functional region")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_uncertainty_by_domain_boxplot.png",
        dpi=300,
    )

    plt.close()

    return summary, kruskal_results, dunn


def main():
    device = get_device()
    print("Using device:", device)

    metadata = pd.read_csv(INPUT_DIR / "paired_embedding_metadata.csv")

    y = metadata["Binary_Label"].astype(int).values
    groups = metadata["AA_Position"].astype(int).values

    wt_site = np.load(INPUT_DIR / "wt_site_embeddings.npy")
    mut_site = np.load(INPUT_DIR / "mut_site_embeddings.npy")

    # Final best model features:
    # WT-site embedding + mutant-site embedding
    X = np.concatenate([wt_site, mut_site], axis=1)

    print("Feature matrix shape:", X.shape)
    print("Metadata rows:", len(metadata))

    all_prediction_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"Training custom WT+mutant NN for seed {seed}")
        print("=" * 90)

        train_idx, val_idx, test_idx = make_position_disjoint_split(
            y=y,
            groups=groups,
            seed=seed,
        )

        verify_position_disjoint(metadata, train_idx, val_idx, test_idx)

        test_probs, test_labels, best_epoch, best_val_auc = train_custom_nn_and_predict(
            X=X,
            y=y,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            seed=seed,
            device=device,
        )

        test_metadata = metadata.iloc[test_idx].copy()

        test_metadata["Seed"] = seed
        test_metadata["Best_Epoch"] = best_epoch
        test_metadata["Best_Validation_AUC"] = best_val_auc
        test_metadata["Pathogenic_Probability"] = test_probs
        test_metadata["Predicted_Label"] = (test_probs >= 0.5).astype(int)
        test_metadata["True_Label"] = test_labels
        test_metadata["Confidence"] = np.maximum(test_probs, 1 - test_probs)
        test_metadata["Uncertainty"] = 1 - test_metadata["Confidence"]
        test_metadata["Correct"] = (
            test_metadata["Predicted_Label"] == test_metadata["True_Label"]
        ).astype(int)

        seed_accuracy = accuracy_score(
            test_metadata["True_Label"],
            test_metadata["Predicted_Label"],
        )

        print(f"Seed {seed} test accuracy @ 0.5: {seed_accuracy:.4f}")
        print(f"Seed {seed} best validation AUC: {best_val_auc:.4f}")

        all_prediction_rows.append(test_metadata)

    all_predictions = pd.concat(all_prediction_rows, ignore_index=True)

    all_predictions.to_csv(
        OUTPUT_DIR / "custom_nn_domain_uncertainty_predictions_all_seeds.csv",
        index=False,
    )

    print("\nSaved all seed-level test predictions.")

    # Variant-level average to avoid treating repeated appearances across seeds
    # as fully independent observations.
    id_cols = [
        "Variant_ID",
        "Gene",
        "AA_Position",
        "Ref_AA",
        "Alt_AA",
        "ClinVar_Label",
        "Binary_Label",
        "Mapped_Region",
        "Mapped_Functional_Category",
    ]

    available_id_cols = [col for col in id_cols if col in all_predictions.columns]

    variant_avg = (
        all_predictions
        .groupby(available_id_cols, dropna=False)
        .agg(
            n_test_appearances=("Seed", "count"),
            Pathogenic_Probability=("Pathogenic_Probability", "mean"),
            True_Label=("True_Label", "first"),
        )
        .reset_index()
    )

    variant_avg["Predicted_Label"] = (
        variant_avg["Pathogenic_Probability"] >= 0.5
    ).astype(int)

    variant_avg["Confidence"] = np.maximum(
        variant_avg["Pathogenic_Probability"],
        1 - variant_avg["Pathogenic_Probability"],
    )

    variant_avg["Uncertainty"] = 1 - variant_avg["Confidence"]

    variant_avg["Correct"] = (
        variant_avg["Predicted_Label"] == variant_avg["True_Label"]
    ).astype(int)

    variant_avg.to_csv(
        OUTPUT_DIR / "custom_nn_domain_uncertainty_variant_averaged_predictions.csv",
        index=False,
    )

    print("Saved variant-averaged predictions.")

    print("\nRunning uncertainty-by-domain analysis on variant-averaged predictions...")
    variant_summary, variant_kw, variant_dunn = summarize_uncertainty_by_domain(
        variant_avg,
        output_prefix="variant_averaged",
    )

    print("\nVariant-averaged uncertainty by domain:")
    print(variant_summary.to_string(index=False))

    print("\nVariant-averaged Kruskal-Wallis:")
    print(variant_kw)

    print("\nVariant-averaged Dunn post hoc test:")
    print(variant_dunn)

    print("\nRunning uncertainty-by-domain analysis on all seed-level predictions...")
    all_summary, all_kw, all_dunn = summarize_uncertainty_by_domain(
        all_predictions,
        output_prefix="all_seed_predictions",
    )

    print("\nAll seed-level uncertainty by domain:")
    print(all_summary.to_string(index=False))

    print("\nAll seed-level Kruskal-Wallis:")
    print(all_kw)

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()