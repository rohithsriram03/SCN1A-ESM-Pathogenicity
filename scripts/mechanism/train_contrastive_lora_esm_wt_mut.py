from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "processed"
    / "scion_mechanism_variants_with_sequences.csv"
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
HELDOUT_GENE = "SCN1A"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "mechanism" / "contrastive_lora_esm_wt_mut"),
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_to_binary(x):
    if x == "GOF":
        return 1
    if x == "LOF":
        return 0
    raise ValueError(f"Unknown label: {x}")


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = [
        "Gene",
        "Mechanism_Label",
        "WT_Window",
        "Mutant_Window",
        "Local_Mutation_Index_0Indexed",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["Mechanism_Binary"] = df["Mechanism_Label"].map(label_to_binary)

    train_df = df[df["Gene"] != HELDOUT_GENE].reset_index(drop=True)
    test_df = df[df["Gene"] == HELDOUT_GENE].reset_index(drop=True)

    return df.reset_index(drop=True), train_df, test_df


def sample_triplet(train_df):
    """
    Balanced, cross-gene triplet sampling.

    Anchor: random GoF or LoF variant from non-SCN1A training genes.
    Positive: same mechanism, preferably different gene.
    Negative: opposite mechanism, preferably different gene.
    """

    anchor_label = random.choice([0, 1])
    anchor_pool = train_df[train_df["Mechanism_Binary"] == anchor_label]

    anchor = anchor_pool.sample(n=1).iloc[0]
    anchor_gene = anchor["Gene"]

    positive_pool = train_df[
        (train_df["Mechanism_Binary"] == anchor_label)
        & (train_df["Gene"] != anchor_gene)
    ]

    if len(positive_pool) == 0:
        positive_pool = train_df[
            train_df["Mechanism_Binary"] == anchor_label
        ]

    negative_pool = train_df[
        (train_df["Mechanism_Binary"] != anchor_label)
        & (train_df["Gene"] != anchor_gene)
    ]

    if len(negative_pool) == 0:
        negative_pool = train_df[
            train_df["Mechanism_Binary"] != anchor_label
        ]

    positive = positive_pool.sample(n=1).iloc[0]
    negative = negative_pool.sample(n=1).iloc[0]

    return anchor, positive, negative


def get_wt_mut_representations(model, tokenizer, rows, device):
    """
    For each variant row, compute:

        WT+mutant representation = concat(WT site embedding, mutant site embedding)

    This is the representation used by the best frozen ESM kNN result.
    """

    seqs = []
    mutation_indices = []

    for row in rows:
        idx = int(row["Local_Mutation_Index_0Indexed"])

        seqs.append(row["WT_Window"])
        mutation_indices.append(idx)

        seqs.append(row["Mutant_Window"])
        mutation_indices.append(idx)

    tokens = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    )

    tokens = {k: v.to(device) for k, v in tokens.items()}

    outputs = model(**tokens)
    hidden = outputs.last_hidden_state

    site_embeddings = []

    for i, idx in enumerate(mutation_indices):
        # +1 because ESM has a beginning special token.
        site_embeddings.append(hidden[i, idx + 1, :])

    site_embeddings = torch.stack(site_embeddings, dim=0)

    wt_embeddings = site_embeddings[0::2]
    mutant_embeddings = site_embeddings[1::2]

    wt_mut = torch.cat([wt_embeddings, mutant_embeddings], dim=-1)
    wt_mut = wt_mut.float()
    wt_mut = F.normalize(wt_mut, dim=-1)

    return wt_mut


def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)


def train_lora(model, tokenizer, train_df, args, output_dir, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    triplet_loss = nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance,
        margin=args.margin,
    )

    loss_rows = []

    model.train()

    for step in tqdm(range(1, args.n_steps + 1), desc="Training WT+mutant LoRA"):
        anchors = []
        positives = []
        negatives = []

        for _ in range(args.batch_size):
            a, p, n = sample_triplet(train_df)
            anchors.append(a)
            positives.append(p)
            negatives.append(n)

        all_rows = anchors + positives + negatives

        reps = get_wt_mut_representations(model, tokenizer, all_rows, device)

        b = args.batch_size
        anchor_reps = reps[:b]
        positive_reps = reps[b : 2 * b]
        negative_reps = reps[2 * b : 3 * b]

        loss = triplet_loss(anchor_reps, positive_reps, negative_reps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_rows.append(
            {
                "step": step,
                "triplet_loss": float(loss.detach().cpu()),
                "learning_rate": args.learning_rate,
                "margin": args.margin,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )

    loss_df = pd.DataFrame(loss_rows)
    loss_df.to_csv(output_dir / "training_loss.csv", index=False)

    return loss_df


@torch.no_grad()
def generate_embeddings(model, tokenizer, df, device, batch_size=2):
    model.eval()

    wt_list = []
    mut_list = []

    rows = [row for _, row in df.iterrows()]

    for start in tqdm(range(0, len(rows), batch_size), desc="Generating LoRA embeddings"):
        batch_rows = rows[start : start + batch_size]

        seqs = []
        mutation_indices = []

        for row in batch_rows:
            idx = int(row["Local_Mutation_Index_0Indexed"])

            seqs.append(row["WT_Window"])
            mutation_indices.append(idx)

            seqs.append(row["Mutant_Window"])
            mutation_indices.append(idx)

        tokens = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = model(**tokens)
        hidden = outputs.last_hidden_state

        site_embeddings = []

        for i, idx in enumerate(mutation_indices):
            site_embeddings.append(hidden[i, idx + 1, :].float().cpu())

        site_embeddings = torch.stack(site_embeddings, dim=0)

        wt_batch = site_embeddings[0::2]
        mut_batch = site_embeddings[1::2]

        wt_list.append(wt_batch)
        mut_list.append(mut_batch)

    wt = torch.cat(wt_list, dim=0).numpy()
    mut = torch.cat(mut_list, dim=0).numpy()
    delta = mut - wt
    wt_mut = np.concatenate([wt, mut], axis=1)

    return wt, mut, delta, wt_mut


def cosine_similarity_matrix(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def evaluate_knn(df, features, output_dir):
    rows = []

    y = df["Mechanism_Binary"].values
    genes = df["Gene"].values

    test_mask = genes == HELDOUT_GENE
    candidate_mask = genes != HELDOUT_GENE

    y_true = y[test_mask]
    candidate_labels = y[candidate_mask]
    candidate_genes = genes[candidate_mask]

    for feature_name, X in features.items():
        X_test = X[test_mask]
        X_candidates = X[candidate_mask]

        sims = cosine_similarity_matrix(X_test, X_candidates)

        for k in [1, 3, 5]:
            topk_idx = np.argsort(-sims, axis=1)[:, :k]

            neighbor_labels = candidate_labels[topk_idx]
            neighbor_genes = candidate_genes[topk_idx]

            # Explicit safety check: no SCN1A neighbors.
            assert not np.any(neighbor_genes == HELDOUT_GENE)

            y_score = neighbor_labels.mean(axis=1)
            y_pred = (y_score >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

            row = {
                "analysis": "contrastive_lora_wt_mut_scn1a_knn",
                "feature_set": feature_name,
                "k": k,
                "n": len(y_true),
                "n_lof": int((y_true == 0).sum()),
                "n_gof": int((y_true == 1).sum()),
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "gof_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                "gof_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                "gof_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                "roc_auc": roc_auc_score(y_true, y_score),
                "average_precision": average_precision_score(y_true, y_score),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

            rows.append(row)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "contrastive_lora_scn1a_knn_metrics.csv", index=False)

    return metrics


def main():
    args = parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = 2 * args.lora_r

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, train_df, test_df = load_data()

    print("Dataset:")
    print("rows:", len(df))
    print("gene counts:")
    print(df["Gene"].value_counts())
    print("\nmechanism counts:")
    print(df["Mechanism_Label"].value_counts())

    print("\nTrain/test split:")
    print("train genes:", sorted(train_df["Gene"].unique().tolist()))
    print("test genes:", sorted(test_df["Gene"].unique().tolist()))
    print("train rows:", len(train_df))
    print("test rows:", len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    model.print_trainable_parameters()

    config = {
        "model_name": MODEL_NAME,
        "heldout_gene": HELDOUT_GENE,
        "representation_mode": "wt_mut_site_contrastive_training",
        "seed": args.seed,
        "n_steps": args.n_steps,
        "learning_rate": args.learning_rate,
        "margin": args.margin,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "n_total": len(df),
        "n_train_non_heldout": len(train_df),
        "n_test_heldout": len(test_df),
        "train_genes": sorted(train_df["Gene"].unique().tolist()),
        "test_genes": sorted(test_df["Gene"].unique().tolist()),
        "train_label_counts": train_df["Mechanism_Label"].value_counts().to_dict(),
        "test_label_counts": test_df["Mechanism_Label"].value_counts().to_dict(),
        "device": device,
    }

    with open(output_dir / "contrastive_lora_qc.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\nTraining WT+mutant LoRA...")
    train_lora(model, tokenizer, train_df, args, output_dir, device)

    print("\nSaving LoRA adapter...")
    model.save_pretrained(output_dir / "lora_adapter")

    print("\nGenerating post-trained LoRA embeddings...")
    wt, mut, delta, wt_mut = generate_embeddings(model, tokenizer, df, device)

    np.save(output_dir / "lora_wt_site_embeddings.npy", wt)
    np.save(output_dir / "lora_mutant_site_embeddings.npy", mut)
    np.save(output_dir / "lora_delta_site_embeddings.npy", delta)
    np.save(output_dir / "lora_wt_mut_site_embeddings.npy", wt_mut)

    metadata = df.copy()
    metadata.to_csv(output_dir / "lora_embedding_metadata.csv", index=False)

    features = {
        "wt_site": wt,
        "mutant_site": mut,
        "delta_site": delta,
        "wt_mut_site": wt_mut,
    }

    print("\nEvaluating held-out SCN1A kNN...")
    metrics = evaluate_knn(df, features, output_dir)

    print("\nHeld-out SCN1A kNN after WT+mutant LoRA:")
    print(
        metrics.sort_values("roc_auc", ascending=False)[
            [
                "feature_set",
                "k",
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "average_precision",
                "gof_f1",
                "gof_recall",
                "gof_precision",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(output_dir)


if __name__ == "__main__":
    main()