from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FEATURE_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
    / "wt_mut_site_features.npy"
)

META_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
    / "scion_embedding_metadata.csv"
)

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "contrastive_projection_esm"


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, proj_dim=128, dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        z = self.net(x)
        z = nn.functional.normalize(z, dim=1)
        return z


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_labels(meta):
    if "Mechanism_Binary" in meta.columns:
        return meta["Mechanism_Binary"].astype(int).to_numpy()

    if "Mechanism_Label" in meta.columns:
        return meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1}).astype(int).to_numpy()

    raise ValueError("Could not find Mechanism_Binary or Mechanism_Label in metadata.")


def supervised_contrastive_loss(z, labels, genes, temperature=0.1):
    """
    Cross-gene supervised contrastive loss.

    Positives:
        same GoF/LoF label, preferably from a different gene.

    Negatives:
        opposite GoF/LoF label.

    ESM itself is frozen. Only the projection head is trained.
    """
    n = z.shape[0]
    device = z.device

    labels = labels.view(-1, 1)
    genes = genes.view(-1, 1)

    same_label = labels.eq(labels.T)
    different_gene = ~genes.eq(genes.T)
    not_self = ~torch.eye(n, dtype=torch.bool, device=device)

    positive_mask = same_label & different_gene & not_self

    # Fallback: if a sample has no cross-gene positive in the batch,
    # allow same-label positives from any gene.
    fallback_mask = same_label & not_self
    no_cross_gene_pos = positive_mask.sum(dim=1) == 0
    positive_mask[no_cross_gene_pos] = fallback_mask[no_cross_gene_pos]

    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = not_self.float()
    exp_logits = torch.exp(logits) * logits_mask

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    positives_per_sample = positive_mask.sum(dim=1)
    valid = positives_per_sample > 0

    loss = -(
        positive_mask.float() * log_prob
    ).sum(dim=1)[valid] / positives_per_sample[valid]

    return loss.mean()


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def evaluate_knn(z, y, genes, query_mask, candidate_mask, k_values=(1, 3, 5)):
    """
    Query variants are evaluated by nearest neighbors from candidate variants.

    For SCN1A testing:
        query_mask = SCN1A variants
        candidate_mask = non-SCN1A variants
    """
    z = l2_normalize(z)

    z_query = z[query_mask]
    y_query = y[query_mask]

    z_candidate = z[candidate_mask]
    y_candidate = y[candidate_mask]
    genes_candidate = genes[candidate_mask]

    sim = np.matmul(z_query, z_candidate.T)

    rows = []

    for k in k_values:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = y_candidate[topk_idx]

        # For k > 1, score is fraction of nearest neighbors labeled GoF.
        scores = topk_labels.mean(axis=1)
        preds = (scores >= 0.5).astype(int)

        row = {
            "k": k,
            "n_query": int(len(y_query)),
            "n_candidate": int(len(y_candidate)),
            "accuracy": accuracy_score(y_query, preds),
            "balanced_accuracy": balanced_accuracy_score(y_query, preds),
            "macro_f1": f1_score(y_query, preds, average="macro", zero_division=0),
            "gof_f1": f1_score(y_query, preds, pos_label=1, zero_division=0),
            "gof_precision": precision_score(y_query, preds, pos_label=1, zero_division=0),
            "gof_recall": recall_score(y_query, preds, pos_label=1, zero_division=0),
        }

        try:
            row["roc_auc"] = roc_auc_score(y_query, scores)
        except ValueError:
            row["roc_auc"] = np.nan

        try:
            row["average_precision"] = average_precision_score(y_query, scores)
        except ValueError:
            row["average_precision"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def project_all(model, scaler, x, device):
    x_scaled = scaler.transform(x).astype(np.float32)

    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))
    return z.cpu().numpy()


def train_one_model(
    x_train,
    y_train,
    genes_train,
    gene_codes_train,
    input_dim,
    seed,
    epochs,
    lr,
    weight_decay,
    temperature,
    hidden_dim,
    proj_dim,
    dropout,
    device,
):
    set_seed(seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype(np.float32)

    x_t = torch.tensor(x_train_scaled, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)
    g_t = torch.tensor(gene_codes_train, dtype=torch.long, device=device)

    model = ProjectionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        z = model(x_t)
        loss = supervised_contrastive_loss(
            z=z,
            labels=y_t,
            genes=g_t,
            temperature=temperature,
        )

        loss.backward()
        optimizer.step()

        losses.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu().item()),
            }
        )

    return model, scaler, pd.DataFrame(losses)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--heldout_gene", type=str, default="SCN1A")
    parser.add_argument("--val_gene", type=str, default="SCN2A")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features:")
    print(FEATURE_FILE)
    x = np.load(FEATURE_FILE)

    print("Loading metadata:")
    print(META_FILE)
    meta = pd.read_csv(META_FILE)

    y = get_labels(meta)
    genes = meta["Gene"].astype(str).to_numpy()

    gene_to_code = {g: i for i, g in enumerate(sorted(set(genes)))}
    gene_codes = np.array([gene_to_code[g] for g in genes])

    input_dim = x.shape[1]

    heldout_gene = args.heldout_gene
    val_gene = args.val_gene

    if heldout_gene not in set(genes):
        raise ValueError(f"Held-out gene {heldout_gene} not found.")

    if val_gene not in set(genes):
        raise ValueError(f"Validation gene {val_gene} not found.")

    heldout_mask = genes == heldout_gene
    non_heldout_mask = genes != heldout_gene

    val_mask = genes == val_gene
    train_sub_mask = (genes != heldout_gene) & (genes != val_gene)

    final_train_mask = genes != heldout_gene

    print("\nDataset split:")
    print(f"Total variants: {len(y)}")
    print(f"Held-out test gene: {heldout_gene}, n={heldout_mask.sum()}")
    print(f"Validation gene for epoch selection: {val_gene}, n={val_mask.sum()}")
    print(f"Training subset excluding heldout and validation gene: n={train_sub_mask.sum()}")
    print(f"Final training set excluding heldout gene only: n={final_train_mask.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    all_seed_metrics = []
    all_seed_val_metrics = []

    for seed in args.seeds:
        print("\n" + "=" * 80)
        print(f"Seed {seed}")
        print("=" * 80)

        seed_dir = OUT_DIR / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Stage 1: train using non-SCN1A minus validation gene.
        # This chooses a reasonable epoch without touching SCN1A.
        # -----------------------------
        print("\nStage 1: validation training")

        val_model, val_scaler, val_losses = train_one_model(
            x_train=x[train_sub_mask],
            y_train=y[train_sub_mask],
            genes_train=genes[train_sub_mask],
            gene_codes_train=gene_codes[train_sub_mask],
            input_dim=input_dim,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            device=device,
        )

        val_z_all = project_all(val_model, val_scaler, x, device)

        val_metrics = evaluate_knn(
            z=val_z_all,
            y=y,
            genes=genes,
            query_mask=val_mask,
            candidate_mask=train_sub_mask,
            k_values=(1, 3, 5),
        )

        val_metrics["seed"] = seed
        val_metrics["query_gene"] = val_gene
        val_metrics["candidate_pool"] = f"non-{heldout_gene}, non-{val_gene}"
        all_seed_val_metrics.append(val_metrics)

        # Since this first version trains for a fixed number of epochs,
        # we keep the same epoch count for final training.
        best_epoch = args.epochs

        # -----------------------------
        # Stage 2: retrain on all non-SCN1A variants.
        # SCN1A is still never used for training.
        # -----------------------------
        print("\nStage 2: final training on all non-heldout variants")

        final_model, final_scaler, final_losses = train_one_model(
            x_train=x[final_train_mask],
            y_train=y[final_train_mask],
            genes_train=genes[final_train_mask],
            gene_codes_train=gene_codes[final_train_mask],
            input_dim=input_dim,
            seed=seed,
            epochs=best_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            hidden_dim=args.hidden_dim,
            proj_dim=args.proj_dim,
            dropout=args.dropout,
            device=device,
        )

        projected_z = project_all(final_model, final_scaler, x, device)

        projected_metrics = evaluate_knn(
            z=projected_z,
            y=y,
            genes=genes,
            query_mask=heldout_mask,
            candidate_mask=non_heldout_mask,
            k_values=(1, 3, 5),
        )

        projected_metrics["seed"] = seed
        projected_metrics["method"] = "contrastive_projection_wt_mut"
        projected_metrics["query_gene"] = heldout_gene
        projected_metrics["candidate_pool"] = f"non-{heldout_gene}"

        # Frozen baseline, using raw WT+mutant features directly.
        frozen_metrics = evaluate_knn(
            z=x,
            y=y,
            genes=genes,
            query_mask=heldout_mask,
            candidate_mask=non_heldout_mask,
            k_values=(1, 3, 5),
        )

        frozen_metrics["seed"] = seed
        frozen_metrics["method"] = "frozen_esm_wt_mut"
        frozen_metrics["query_gene"] = heldout_gene
        frozen_metrics["candidate_pool"] = f"non-{heldout_gene}"

        combined = pd.concat([frozen_metrics, projected_metrics], ignore_index=True)
        all_seed_metrics.append(combined)

        np.save(seed_dir / "projected_embeddings.npy", projected_z)
        final_losses.to_csv(seed_dir / "training_loss.csv", index=False)
        val_losses.to_csv(seed_dir / "validation_training_loss.csv", index=False)
        combined.to_csv(seed_dir / "heldout_scn1a_metrics.csv", index=False)

        config = vars(args).copy()
        config["seed"] = seed
        config["feature_file"] = str(FEATURE_FILE)
        config["metadata_file"] = str(META_FILE)
        config["input_dim"] = int(input_dim)
        config["device"] = str(device)

        with open(seed_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("\nHeld-out SCN1A metrics:")
        print(
            combined[
                [
                    "method",
                    "k",
                    "accuracy",
                    "balanced_accuracy",
                    "roc_auc",
                    "average_precision",
                    "gof_f1",
                ]
            ].to_string(index=False)
        )

    # -----------------------------
    # Aggregate across seeds
    # -----------------------------
    all_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    all_val_metrics = pd.concat(all_seed_val_metrics, ignore_index=True)

    all_metrics.to_csv(OUT_DIR / "all_seed_heldout_scn1a_metrics.csv", index=False)
    all_val_metrics.to_csv(OUT_DIR / "all_seed_validation_metrics.csv", index=False)

    summary = (
        all_metrics
        .groupby(["method", "k"])
        .agg(
            n_seeds=("seed", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
            mean_gof_f1=("gof_f1", "mean"),
            std_gof_f1=("gof_f1", "std"),
        )
        .reset_index()
        .sort_values(["mean_balanced_accuracy", "mean_roc_auc"], ascending=False)
    )

    summary.to_csv(OUT_DIR / "summary_by_method_k.csv", index=False)

    print("\n" + "=" * 80)
    print("Multi-seed summary")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()