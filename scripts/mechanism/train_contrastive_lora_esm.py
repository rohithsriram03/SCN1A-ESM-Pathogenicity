from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
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

from transformers import AutoTokenizer, EsmModel
from peft import LoraConfig, get_peft_model


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "processed"
    / "scion_mechanism_variants_with_sequences.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "contrastive_lora_esm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use 650M for the real experiment.
# If your Mac runs out of memory, test the script first with:
# "facebook/esm2_t12_35M_UR50D"
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

HELDOUT_GENE = "SCN1A"

RANDOM_SEED = 42
N_STEPS = 200
LEARNING_RATE = 2e-4
MARGIN = 0.2

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

K_VALUES = [1, 3, 5]


# =============================================================================
# SETUP
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_possible_lora_targets(model):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.append(name)
    return names


def load_data():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "Gene",
        "Mechanism_Label",
        "Mechanism_Binary",
        "WT_Window",
        "Mutant_Window",
        "Local_Mutation_Index_0Indexed",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from input file: {missing}")

    df = df.copy()
    df["Mechanism_Binary"] = df["Mechanism_Binary"].astype(int)
    df["Local_Mutation_Index_0Indexed"] = df["Local_Mutation_Index_0Indexed"].astype(int)

    print("Loaded rows:", len(df))
    print("Mechanism counts:")
    print(df["Mechanism_Label"].value_counts().to_string())
    print("\nGene counts:")
    print(df["Gene"].value_counts().to_string())

    return df


# =============================================================================
# MODEL + EMBEDDING FUNCTIONS
# =============================================================================

def build_lora_esm_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = EsmModel.from_pretrained(MODEL_NAME)

    print("\nLoaded base model:", MODEL_NAME)

    # ESM attention modules usually include query/key/value linear layers.
    # We target query and value first because this is a standard lightweight LoRA setup.
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )

    try:
        model = get_peft_model(base_model, lora_config)
    except ValueError as e:
        print("\nLoRA target module error.")
        print("Here are possible Linear module names in this ESM model:")
        for name in find_possible_lora_targets(base_model)[:100]:
            print(name)
        raise e

    model.to(device)
    model.train()

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    return tokenizer, model


def get_site_embedding(model, tokenizer, sequence, local_mutation_index, device):
    """
    Returns the ESM embedding at the mutation site.

    ESM tokenization adds a special token at the beginning, so amino-acid index i
    maps to token position i + 1.
    """
    encoded = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    outputs = model(**encoded)
    hidden = outputs.last_hidden_state

    token_position = int(local_mutation_index) + 1

    if token_position >= hidden.shape[1]:
        raise ValueError(
            f"Token position {token_position} outside hidden-state length {hidden.shape[1]}"
        )

    emb = hidden[:, token_position, :]
    emb = F.normalize(emb, dim=-1)

    return emb


def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)


# =============================================================================
# TRIPLET SAMPLING
# =============================================================================

def sample_triplet(train_df):
    """
    Anchor and positive share the same GoF/LoF label.
    Negative has the opposite GoF/LoF label.

    Whenever possible, positive and negative examples are sampled from different genes
    than the anchor to encourage cross-paralog mechanism transfer.
    """
    anchor_idx = random.choice(train_df.index.tolist())
    anchor = train_df.loc[anchor_idx]
    anchor_label = int(anchor["Mechanism_Binary"])
    anchor_gene = anchor["Gene"]

    same_label = train_df[
        (train_df["Mechanism_Binary"] == anchor_label)
        & (train_df.index != anchor_idx)
    ]

    same_label_diff_gene = same_label[same_label["Gene"] != anchor_gene]
    if len(same_label_diff_gene) > 0:
        positive = same_label_diff_gene.sample(1, random_state=random.randint(0, 10**9)).iloc[0]
    else:
        positive = same_label.sample(1, random_state=random.randint(0, 10**9)).iloc[0]

    opposite_label = train_df[train_df["Mechanism_Binary"] != anchor_label]
    opposite_label_diff_gene = opposite_label[opposite_label["Gene"] != anchor_gene]
    if len(opposite_label_diff_gene) > 0:
        negative = opposite_label_diff_gene.sample(1, random_state=random.randint(0, 10**9)).iloc[0]
    else:
        negative = opposite_label.sample(1, random_state=random.randint(0, 10**9)).iloc[0]

    return anchor, positive, negative


# =============================================================================
# TRAINING
# =============================================================================

def train_contrastive_lora(df, tokenizer, model, device):
    train_df = df[df["Gene"] != HELDOUT_GENE].copy()

    print("\nTraining LoRA on non-held-out genes only.")
    print("Held-out gene:", HELDOUT_GENE)
    print("Training rows:", len(train_df))
    print("Training mechanism counts:")
    print(train_df["Mechanism_Label"].value_counts().to_string())

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    loss_fn = nn.TripletMarginWithDistanceLoss(
        distance_function=cosine_distance,
        margin=MARGIN,
        reduction="mean",
    )

    losses = []

    for step in range(1, N_STEPS + 1):
        model.train()
        optimizer.zero_grad()

        anchor, positive, negative = sample_triplet(train_df)

        anchor_emb = get_site_embedding(
            model,
            tokenizer,
            anchor["Mutant_Window"],
            anchor["Local_Mutation_Index_0Indexed"],
            device,
        )

        positive_emb = get_site_embedding(
            model,
            tokenizer,
            positive["Mutant_Window"],
            positive["Local_Mutation_Index_0Indexed"],
            device,
        )

        negative_emb = get_site_embedding(
            model,
            tokenizer,
            negative["Mutant_Window"],
            negative["Local_Mutation_Index_0Indexed"],
            device,
        )

        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        if step % 10 == 0 or step == 1:
            recent = np.mean(losses[-10:])
            print(f"Step {step:04d}/{N_STEPS} | recent triplet loss = {recent:.4f}")

    loss_df = pd.DataFrame(
        {
            "step": np.arange(1, len(losses) + 1),
            "triplet_loss": losses,
        }
    )
    loss_df.to_csv(OUTPUT_DIR / "training_loss.csv", index=False)

    adapter_dir = OUTPUT_DIR / "lora_adapter"
    model.save_pretrained(adapter_dir)

    print("\nSaved LoRA adapter to:")
    print(adapter_dir)

    return model


# =============================================================================
# EMBEDDING GENERATION
# =============================================================================

@torch.no_grad()
def generate_adapted_embeddings(df, tokenizer, model, device):
    model.eval()

    wt_embeddings = []
    mutant_embeddings = []

    for i, row in df.iterrows():
        if i % 25 == 0:
            print(f"Embedding row {i}/{len(df)}")

        wt_emb = get_site_embedding(
            model,
            tokenizer,
            row["WT_Window"],
            row["Local_Mutation_Index_0Indexed"],
            device,
        )

        mut_emb = get_site_embedding(
            model,
            tokenizer,
            row["Mutant_Window"],
            row["Local_Mutation_Index_0Indexed"],
            device,
        )

        wt_embeddings.append(wt_emb.squeeze(0).detach().cpu().numpy())
        mutant_embeddings.append(mut_emb.squeeze(0).detach().cpu().numpy())

    wt_embeddings = np.vstack(wt_embeddings)
    mutant_embeddings = np.vstack(mutant_embeddings)

    delta_embeddings = mutant_embeddings - wt_embeddings
    wt_mut_embeddings = np.concatenate([wt_embeddings, mutant_embeddings], axis=1)

    np.save(OUTPUT_DIR / "lora_wt_site_embeddings.npy", wt_embeddings)
    np.save(OUTPUT_DIR / "lora_mutant_site_embeddings.npy", mutant_embeddings)
    np.save(OUTPUT_DIR / "lora_delta_site_embeddings.npy", delta_embeddings)
    np.save(OUTPUT_DIR / "lora_wt_mut_site_embeddings.npy", wt_mut_embeddings)

    meta_out = df.copy()
    meta_out.to_csv(OUTPUT_DIR / "lora_embedding_metadata.csv", index=False)

    print("\nSaved adapted embeddings to:")
    print(OUTPUT_DIR)

    return {
        "wt_site": wt_embeddings,
        "mutant_site": mutant_embeddings,
        "delta_site": delta_embeddings,
        "wt_mut_site": wt_mut_embeddings,
    }


# =============================================================================
# KNN EVALUATION
# =============================================================================

def normalize_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def evaluate_scn1a_knn(df, features, feature_name, k):
    X = normalize_matrix(features)

    y = df["Mechanism_Binary"].astype(int).values
    genes = df["Gene"].astype(str).values

    query_idx = np.where(genes == HELDOUT_GENE)[0]
    candidate_idx = np.where(genes != HELDOUT_GENE)[0]

    y_true = []
    y_pred = []
    y_score = []

    for i in query_idx:
        sims = X[[i]] @ X[candidate_idx].T
        sims = sims.ravel()

        order = np.argsort(-sims)
        top_idx = candidate_idx[order[:k]]

        top_labels = y[top_idx]
        score = float(np.mean(top_labels == 1))
        pred = 1 if score >= 0.5 else 0

        y_true.append(int(y[i]))
        y_pred.append(pred)
        y_score.append(score)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "method": "contrastive_lora_esm_knn",
        "heldout_gene": HELDOUT_GENE,
        "feature_set": feature_name,
        "k": int(k),
        "n": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "gof_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "gof_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "gof_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def run_knn_eval(df, feature_dict):
    rows = []

    for feature_name, features in feature_dict.items():
        for k in K_VALUES:
            rows.append(evaluate_scn1a_knn(df, features, feature_name, k))

    out = pd.DataFrame(rows)
    out = out.sort_values("roc_auc", ascending=False)

    out.to_csv(OUTPUT_DIR / "contrastive_lora_scn1a_knn_metrics.csv", index=False)

    print("\nHeld-out SCN1A kNN after contrastive LoRA:")
    print(out.to_string(index=False))

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_seed(RANDOM_SEED)
    device = get_device()

    print("Device:", device)
    print("Input file:", INPUT_FILE)
    print("Output dir:", OUTPUT_DIR)

    df = load_data()

    tokenizer, model = build_lora_esm_model(device)

    model = train_contrastive_lora(df, tokenizer, model, device)

    feature_dict = generate_adapted_embeddings(df, tokenizer, model, device)

    metrics = run_knn_eval(df, feature_dict)

    qc = {
        "model_name": MODEL_NAME,
        "heldout_gene": HELDOUT_GENE,
        "n_rows": int(len(df)),
        "n_train_non_heldout": int((df["Gene"] != HELDOUT_GENE).sum()),
        "n_test_heldout": int((df["Gene"] == HELDOUT_GENE).sum()),
        "n_steps": int(N_STEPS),
        "learning_rate": LEARNING_RATE,
        "margin": MARGIN,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "k_values": K_VALUES,
        "best_row_by_auc": metrics.iloc[0].to_dict(),
    }

    with open(OUTPUT_DIR / "contrastive_lora_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\nSaved QC:")
    print(json.dumps(qc, indent=4))


if __name__ == "__main__":
    main()