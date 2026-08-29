from pathlib import Path
import argparse
import json
import random
import itertools

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

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "contrastive_projection_generalization_sweep"


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
        return nn.functional.normalize(z, dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_labels(meta):
    if "Mechanism_Binary" in meta.columns:
        return meta["Mechanism_Binary"].astype(int).to_numpy()

    if "Mechanism_Label" in meta.columns:
        return meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1}).astype(int).to_numpy()

    raise ValueError("Could not find Mechanism_Binary or Mechanism_Label.")


def supervised_contrastive_loss(z, labels, genes, temperature=0.1):
    n = z.shape[0]
    device = z.device

    labels = labels.view(-1, 1)
    genes = genes.view(-1, 1)

    same_label = labels.eq(labels.T)
    different_gene = ~genes.eq(genes.T)
    not_self = ~torch.eye(n, dtype=torch.bool, device=device)

    positive_mask = same_label & different_gene & not_self

    # Fallback if no cross-gene positive exists inside the batch/full training set.
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


def evaluate_knn(z, y, query_mask, candidate_mask, k_values=(1, 3, 5)):
    z = l2_normalize(z)

    z_query = z[query_mask]
    y_query = y[query_mask]

    z_candidate = z[candidate_mask]
    y_candidate = y[candidate_mask]

    if len(np.unique(y_query)) < 2:
        return pd.DataFrame()

    sim = np.matmul(z_query, z_candidate.T)

    rows = []

    for k in k_values:
        if z_candidate.shape[0] < k:
            continue

        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = y_candidate[topk_idx]

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


def train_projection(
    x_train,
    y_train,
    gene_codes_train,
    input_dim,
    hidden_dim,
    proj_dim,
    dropout,
    epochs,
    lr,
    weight_decay,
    temperature,
    seed,
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

        losses.append(float(loss.detach().cpu().item()))

    return model, scaler, losses


def project_all(model, scaler, x, device):
    x_scaled = scaler.transform(x).astype(np.float32)

    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(x_scaled, dtype=torch.float32, device=device))

    return z.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_genes", nargs="+", default=["SCN1A", "SCN2A"])
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--proj_dims", nargs="+", type=int, default=[64, 128, 256, 512])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.2)

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.load(FEATURE_FILE)
    meta = pd.read_csv(META_FILE)

    y = get_labels(meta)
    genes = meta["Gene"].astype(str).to_numpy()

    gene_to_code = {g: i for i, g in enumerate(sorted(set(genes)))}
    gene_codes = np.array([gene_to_code[g] for g in genes])

    input_dim = x.shape[1]

    print(f"Loaded features: {x.shape}")
    print(f"Input dimension: {input_dim}")
    print(f"Genes: {sorted(set(genes))}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_rows = []

    configs = list(itertools.product(args.target_genes, args.hidden_dims, args.proj_dims, args.seeds))

    for target_gene, hidden_dim, proj_dim, seed in configs:
        print("\n" + "=" * 90)
        print(f"Target={target_gene} | hidden={hidden_dim} | proj={proj_dim} | seed={seed}")
        print("=" * 90)

        target_mask = genes == target_gene
        train_mask = genes != target_gene

        if target_mask.sum() == 0:
            print(f"Skipping {target_gene}: no variants.")
            continue

        if len(np.unique(y[target_mask])) < 2:
            print(f"Skipping {target_gene}: target has only one class.")
            continue

        model, scaler, losses = train_projection(
            x_train=x[train_mask],
            y_train=y[train_mask],
            gene_codes_train=gene_codes[train_mask],
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            seed=seed,
            device=device,
        )

        z_proj = project_all(model, scaler, x, device)

        projected_metrics = evaluate_knn(
            z=z_proj,
            y=y,
            query_mask=target_mask,
            candidate_mask=train_mask,
            k_values=(1, 3, 5),
        )

        projected_metrics["method"] = "contrastive_projection_wt_mut"
        projected_metrics["target_gene"] = target_gene
        projected_metrics["candidate_pool"] = f"non-{target_gene}"
        projected_metrics["hidden_dim"] = hidden_dim
        projected_metrics["proj_dim"] = proj_dim
        projected_metrics["seed"] = seed
        projected_metrics["final_loss"] = losses[-1]

        frozen_metrics = evaluate_knn(
            z=x,
            y=y,
            query_mask=target_mask,
            candidate_mask=train_mask,
            k_values=(1, 3, 5),
        )

        frozen_metrics["method"] = "frozen_esm_wt_mut"
        frozen_metrics["target_gene"] = target_gene
        frozen_metrics["candidate_pool"] = f"non-{target_gene}"
        frozen_metrics["hidden_dim"] = hidden_dim
        frozen_metrics["proj_dim"] = proj_dim
        frozen_metrics["seed"] = seed
        frozen_metrics["final_loss"] = np.nan

        all_rows.append(projected_metrics)
        all_rows.append(frozen_metrics)

        run_dir = OUT_DIR / f"target_{target_gene}_hidden_{hidden_dim}_proj_{proj_dim}_seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        projected_metrics.to_csv(run_dir / "projected_metrics.csv", index=False)
        frozen_metrics.to_csv(run_dir / "frozen_metrics.csv", index=False)
        pd.DataFrame({"epoch": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
            run_dir / "training_loss.csv",
            index=False,
        )

        with open(run_dir / "config.json", "w") as f:
            json.dump(
                {
                    "target_gene": target_gene,
                    "hidden_dim": hidden_dim,
                    "proj_dim": proj_dim,
                    "seed": seed,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "temperature": args.temperature,
                    "dropout": args.dropout,
                    "input_dim": input_dim,
                },
                f,
                indent=2,
            )

    all_metrics = pd.concat(all_rows, ignore_index=True)
    all_metrics.to_csv(OUT_DIR / "all_metrics.csv", index=False)

    summary = (
        all_metrics
        .groupby(["target_gene", "method", "hidden_dim", "proj_dim", "k"])
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
            mean_final_loss=("final_loss", "mean"),
        )
        .reset_index()
        .sort_values(
            ["target_gene", "mean_balanced_accuracy", "mean_roc_auc"],
            ascending=[True, False, False],
        )
    )

    summary.to_csv(OUT_DIR / "summary_by_config.csv", index=False)

    projected_only = summary[summary["method"] == "contrastive_projection_wt_mut"].copy()

    best_per_target = (
        projected_only
        .sort_values(
            ["target_gene", "mean_balanced_accuracy", "mean_roc_auc", "mean_gof_f1"],
            ascending=[True, False, False, False],
        )
        .groupby("target_gene")
        .head(10)
    )

    best_per_target.to_csv(OUT_DIR / "top10_projected_configs_by_target.csv", index=False)

    print("\n" + "=" * 90)
    print("Top projected configurations by target gene")
    print("=" * 90)

    cols = [
        "target_gene",
        "hidden_dim",
        "proj_dim",
        "k",
        "n_seeds",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "mean_roc_auc",
        "mean_gof_f1",
    ]

    print(best_per_target[cols].to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()