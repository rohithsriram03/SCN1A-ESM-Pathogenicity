from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "benchmarking" / "esm2_650M_embeddings"
ZERO_SHOT_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "esm_zero_shot"
    / "esm2_650M_zero_shot_scores.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarking" / "esm2_650M_supervised_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILE = EMBED_DIR / "wt_mut_site_features_650m.npy"
METADATA_FILE = EMBED_DIR / "paired_embedding_metadata_650m.csv"

SEEDS = list(range(10))
N_SPLITS = 5
MAX_EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


class VariantNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
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
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_loader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def evaluate_scores(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
    }


def optimize_threshold_for_f1(y_true, scores):
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))

    best_threshold = 0.5
    best_f1 = -1

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred, zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


def train_one_model(X_train, y_train, X_val, y_val, input_dim, seed, device):
    set_seed(seed)

    model = VariantNN(input_dim=input_dim).to(device)

    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2.0 * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_val_auc = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs = []

        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())

        val_probs = np.array(val_probs)

        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = -1

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    return model, best_val_auc


def predict_model(model, X, device):
    loader = make_loader(X, np.zeros(len(X), dtype=int), BATCH_SIZE, shuffle=False)

    probs = []

    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())

    return np.array(probs)


def summarize_results(results_df):
    metric_cols = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "optimized_f1",
    ]

    rows = []

    for model_name, sub in results_df.groupby("model"):
        row = {"model": model_name}

        for metric in metric_cols:
            row[f"{metric}_mean"] = sub[metric].mean()
            row[f"{metric}_std"] = sub[metric].std()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def main():
    print("Reading features:")
    print(FEATURE_FILE)

    X = np.load(FEATURE_FILE)

    print("Reading metadata:")
    print(METADATA_FILE)

    meta = pd.read_csv(METADATA_FILE)

    print("Feature shape:", X.shape)
    print("Metadata rows:", len(meta))

    if X.shape[0] != len(meta):
        raise ValueError("Feature rows and metadata rows do not match.")

    # Add zero-shot scores if available
    if ZERO_SHOT_FILE.exists():
        zero = pd.read_csv(ZERO_SHOT_FILE)

        keep_cols = [
            "Variant_Key",
            "ESM_ZeroShot_Pathogenic_Score",
        ]

        zero = zero[keep_cols].drop_duplicates(subset=["Variant_Key"])

        meta = meta.merge(zero, on="Variant_Key", how="left")

        print("\nMerged zero-shot scores.")
        print("Zero-shot missing:", meta["ESM_ZeroShot_Pathogenic_Score"].isna().sum())
    else:
        print("\nWARNING: zero-shot file not found:")
        print(ZERO_SHOT_FILE)

    y = meta["Binary_Label"].astype(int).values
    groups = meta["AA_Position"].astype(int).values

    device = choose_device()
    print("\nDevice:", device)

    all_results = []
    all_predictions = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"SEED {seed}")
        print("=" * 80)

        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        train_val_idx, test_idx = next(sgkf.split(X, y, groups))

        X_train_val = X[train_val_idx]
        y_train_val = y[train_val_idx]
        groups_train_val = groups[train_val_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        meta_test = meta.iloc[test_idx].copy()

        # Validation split inside train set.
        # This is not group-disjoint within train_val, but test remains position-disjoint.
        train_idx_local, val_idx_local = train_test_split(
            np.arange(len(train_val_idx)),
            test_size=0.2,
            random_state=seed,
            stratify=y_train_val,
        )

        X_train = X_train_val[train_idx_local]
        y_train = y_train_val[train_idx_local]

        X_val = X_train_val[val_idx_local]
        y_val = y_train_val[val_idx_local]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        model, best_val_auc = train_one_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            input_dim=X.shape[1],
            seed=seed,
            device=device,
        )

        nn_scores = predict_model(model, X_test_scaled, device)

        nn_metrics = evaluate_scores(y_test, nn_scores, threshold=0.5)
        nn_best_threshold, nn_best_f1 = optimize_threshold_for_f1(y_test, nn_scores)

        nn_metrics["optimized_threshold"] = nn_best_threshold
        nn_metrics["optimized_f1"] = nn_best_f1
        nn_metrics["model"] = "ESM2_650M_WTMut_NN"
        nn_metrics["seed"] = seed
        nn_metrics["best_val_auc"] = best_val_auc
        nn_metrics["n_test"] = len(test_idx)

        all_results.append(nn_metrics)

        # AlphaMissense on same test fold
        if "AlphaMissense_Score" in meta_test.columns:
            alpha_test = meta_test.dropna(subset=["AlphaMissense_Score"]).copy()

            if len(alpha_test) > 0 and alpha_test["Binary_Label"].nunique() == 2:
                alpha_scores = alpha_test["AlphaMissense_Score"].astype(float).values
                alpha_y = alpha_test["Binary_Label"].astype(int).values

                alpha_metrics = evaluate_scores(
                    alpha_y,
                    alpha_scores,
                    threshold=0.564,
                )

                alpha_best_threshold, alpha_best_f1 = optimize_threshold_for_f1(
                    alpha_y,
                    alpha_scores,
                )

                alpha_metrics["optimized_threshold"] = alpha_best_threshold
                alpha_metrics["optimized_f1"] = alpha_best_f1
                alpha_metrics["model"] = "AlphaMissense"
                alpha_metrics["seed"] = seed
                alpha_metrics["best_val_auc"] = np.nan
                alpha_metrics["n_test"] = len(alpha_test)

                all_results.append(alpha_metrics)

        # 650M zero-shot on same test fold
        if "ESM_ZeroShot_Pathogenic_Score" in meta_test.columns:
            zero_test = meta_test.dropna(subset=["ESM_ZeroShot_Pathogenic_Score"]).copy()

            if len(zero_test) > 0 and zero_test["Binary_Label"].nunique() == 2:
                zero_scores = zero_test["ESM_ZeroShot_Pathogenic_Score"].astype(float).values
                zero_y = zero_test["Binary_Label"].astype(int).values

                zero_metrics = evaluate_scores(
                    zero_y,
                    zero_scores,
                    threshold=0.0,
                )

                zero_best_threshold, zero_best_f1 = optimize_threshold_for_f1(
                    zero_y,
                    zero_scores,
                )

                zero_metrics["optimized_threshold"] = zero_best_threshold
                zero_metrics["optimized_f1"] = zero_best_f1
                zero_metrics["model"] = "ESM2_650M_ZeroShot"
                zero_metrics["seed"] = seed
                zero_metrics["best_val_auc"] = np.nan
                zero_metrics["n_test"] = len(zero_test)

                all_results.append(zero_metrics)

        fold_pred = meta_test.copy()
        fold_pred["seed"] = seed
        fold_pred["ESM2_650M_WTMut_NN_Score"] = nn_scores
        fold_pred["ESM2_650M_WTMut_NN_Pred"] = (nn_scores >= 0.5).astype(int)

        all_predictions.append(fold_pred)

        print("NN test AUC:", round(nn_metrics["roc_auc"], 4))
        print("NN test F1:", round(nn_metrics["f1"], 4))
        print("NN optimized F1:", round(nn_metrics["optimized_f1"], 4))

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    summary_df = summarize_results(results_df)

    results_df.to_csv(OUTPUT_DIR / "seed_level_results.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "test_fold_predictions_all_seeds.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "summary_results.csv", index=False)

    with open(OUTPUT_DIR / "summary_results.json", "w") as f:
        json.dump(
            {
                "feature_file": str(FEATURE_FILE),
                "metadata_file": str(METADATA_FILE),
                "zero_shot_file": str(ZERO_SHOT_FILE),
                "n_variants": int(len(meta)),
                "feature_shape": list(X.shape),
                "seeds": SEEDS,
                "summary": summary_df.to_dict(orient="records"),
            },
            f,
            indent=4,
        )

    print("\n" + "=" * 100)
    print("SUMMARY RESULTS")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()