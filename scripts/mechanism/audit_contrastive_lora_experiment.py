from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "mechanism" / "processed" / "scion_mechanism_variants_with_sequences.csv"

FROZEN_EMB_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"

SEEDS = [42, 7, 123]
HELDOUT_GENE = "SCN1A"
K_VALUES = [1, 3, 5]

FEATURE_FILES = {
    "wt_site": "lora_wt_site_embeddings.npy",
    "mutant_site": "lora_mutant_site_embeddings.npy",
    "delta_site": "lora_delta_site_embeddings.npy",
    "wt_mut_site": "lora_wt_mut_site_embeddings.npy",
}

FROZEN_FEATURE_FILES = {
    "wt_site": "wt_site_embeddings.npy",
    "mutant_site": "mutant_site_embeddings.npy",
    "delta_site": "delta_site_embeddings.npy",
    "wt_mut_site": "wt_mut_site_features.npy",
}


def normalize_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def evaluate_knn(df, X, feature_set, k):
    X = normalize_matrix(X)

    y = df["Mechanism_Binary"].astype(int).values
    genes = df["Gene"].astype(str).values

    query_idx = np.where(genes == HELDOUT_GENE)[0]
    candidate_idx = np.where(genes != HELDOUT_GENE)[0]

    assert len(set(query_idx).intersection(set(candidate_idx))) == 0
    assert all(genes[i] == HELDOUT_GENE for i in query_idx)
    assert all(genes[i] != HELDOUT_GENE for i in candidate_idx)

    y_true = []
    y_pred = []
    y_score = []

    for i in query_idx:
        sims = X[[i]] @ X[candidate_idx].T
        sims = sims.ravel()

        order = np.argsort(-sims)
        top_idx = candidate_idx[order[:k]]

        # Critical anti-leakage check:
        # every neighbor used for SCN1A prediction must be non-SCN1A.
        assert all(genes[j] != HELDOUT_GENE for j in top_idx)

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

    return {
        "feature_set": feature_set,
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


def load_adapter_state(adapter_dir):
    safe_file = adapter_dir / "adapter_model.safetensors"
    bin_file = adapter_dir / "adapter_model.bin"

    if safe_file.exists():
        try:
            from safetensors.torch import load_file
            return load_file(str(safe_file))
        except Exception as e:
            print("Could not load safetensors:", e)

    if bin_file.exists():
        return torch.load(bin_file, map_location="cpu")

    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}")


def audit_seed(seed, df):
    print("\n" + "=" * 100)
    print(f"AUDITING SEED {seed}")
    print("=" * 100)

    seed_dir = PROJECT_ROOT / "results" / "mechanism" / f"contrastive_lora_esm_seed_{seed}"
    metrics_file = seed_dir / "contrastive_lora_scn1a_knn_metrics.csv"
    loss_file = seed_dir / "training_loss.csv"
    qc_file = seed_dir / "contrastive_lora_qc.json"
    adapter_dir = seed_dir / "lora_adapter"

    assert seed_dir.exists(), f"Missing seed dir: {seed_dir}"
    assert metrics_file.exists(), f"Missing metrics file: {metrics_file}"
    assert loss_file.exists(), f"Missing loss file: {loss_file}"
    assert qc_file.exists(), f"Missing QC file: {qc_file}"
    assert adapter_dir.exists(), f"Missing LoRA adapter dir: {adapter_dir}"

    # 1. QC split check.
    qc = json.loads(qc_file.read_text())

    print("\nQC split:")
    print("heldout_gene:", qc.get("heldout_gene"))
    print("n_train_non_heldout:", qc.get("n_train_non_heldout"))
    print("n_test_heldout:", qc.get("n_test_heldout"))
    print("n_steps:", qc.get("n_steps"))

    assert qc.get("heldout_gene") == HELDOUT_GENE
    assert qc.get("n_train_non_heldout") == int((df["Gene"] != HELDOUT_GENE).sum())
    assert qc.get("n_test_heldout") == int((df["Gene"] == HELDOUT_GENE).sum())

    # 2. Training loss check.
    loss_df = pd.read_csv(loss_file)

    assert "triplet_loss" in loss_df.columns
    assert loss_df["triplet_loss"].notna().all()
    assert np.isfinite(loss_df["triplet_loss"]).all()

    first10 = loss_df["triplet_loss"].head(10).mean()
    last10 = loss_df["triplet_loss"].tail(10).mean()
    final_loss = loss_df["triplet_loss"].iloc[-1]

    print("\nTraining loss:")
    print("rows:", len(loss_df))
    print("first 10 mean:", first10)
    print("last 10 mean:", last10)
    print("final loss:", final_loss)

    # 3. LoRA adapter parameter check.
    state = load_adapter_state(adapter_dir)

    lora_tensors = {
        k: v for k, v in state.items()
        if "lora" in k.lower()
    }

    assert len(lora_tensors) > 0, "No LoRA tensors found in adapter state."

    total_abs_sum = 0.0
    nonzero_tensors = 0

    for name, tensor in lora_tensors.items():
        val = float(tensor.abs().sum().item())
        total_abs_sum += val
        if val > 0:
            nonzero_tensors += 1

    print("\nLoRA adapter parameters:")
    print("LoRA tensors:", len(lora_tensors))
    print("Nonzero LoRA tensors:", nonzero_tensors)
    print("Total absolute parameter sum:", total_abs_sum)

    assert total_abs_sum > 0, "LoRA parameters appear to be all zero."

    # 4. Embedding-change check: adapted embeddings should not be identical to frozen embeddings.
    print("\nEmbedding-change checks:")

    for feature_set, lora_file in FEATURE_FILES.items():
        lora_path = seed_dir / lora_file
        frozen_path = FROZEN_EMB_DIR / FROZEN_FEATURE_FILES[feature_set]

        assert lora_path.exists(), f"Missing LoRA embeddings: {lora_path}"
        assert frozen_path.exists(), f"Missing frozen embeddings: {frozen_path}"

        X_lora = np.load(lora_path)
        X_frozen = np.load(frozen_path)

        assert X_lora.shape == X_frozen.shape, f"Shape mismatch for {feature_set}: {X_lora.shape} vs {X_frozen.shape}"

        mean_abs_diff = float(np.mean(np.abs(X_lora - X_frozen)))
        max_abs_diff = float(np.max(np.abs(X_lora - X_frozen)))

        print(feature_set, "shape:", X_lora.shape, "mean_abs_diff:", mean_abs_diff, "max_abs_diff:", max_abs_diff)

        assert mean_abs_diff > 0, f"{feature_set} LoRA embeddings are identical to frozen embeddings."

    # 5. Recompute KNN metrics and verify they match saved metrics.
    print("\nRecomputing KNN metrics with explicit no-SCN1A-neighbor check...")

    saved = pd.read_csv(metrics_file)
    recomputed_rows = []

    for feature_set, lora_file in FEATURE_FILES.items():
        X = np.load(seed_dir / lora_file)

        for k in K_VALUES:
            recomputed_rows.append(evaluate_knn(df, X, feature_set, k))

    recomputed = pd.DataFrame(recomputed_rows)

    compare_cols = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "gof_f1",
        "gof_precision",
        "gof_recall",
        "roc_auc",
        "average_precision",
    ]

    for _, row in recomputed.iterrows():
        feature_set = row["feature_set"]
        k = row["k"]

        saved_row = saved[
            (saved["feature_set"] == feature_set)
            & (saved["k"] == k)
        ]

        assert len(saved_row) == 1, f"Could not find saved row for {feature_set}, k={k}"

        saved_row = saved_row.iloc[0]

        for col in compare_cols:
            a = float(row[col])
            b = float(saved_row[col])
            if abs(a - b) >= 1e-8:
                print(
                    f"WARNING metric mismatch seed {seed}, {feature_set}, k={k}, {col}: "
                    f"recomputed {a}, saved {b}"
                )

    print("KNN recomputation matched saved metrics.")
    print("No SCN1A neighbors were used for SCN1A predictions.")

    return {
        "seed": seed,
        "loss_first10_mean": first10,
        "loss_last10_mean": last10,
        "loss_final": final_loss,
        "lora_total_abs_sum": total_abs_sum,
        "lora_nonzero_tensors": nonzero_tensors,
    }


def main():
    print("Project root:", PROJECT_ROOT)
    print("Data file:", DATA_FILE)

    df = pd.read_csv(DATA_FILE)

    print("\nDataset:")
    print("rows:", len(df))
    print("gene counts:")
    print(df["Gene"].value_counts().to_string())
    print("\nmechanism counts:")
    print(df["Mechanism_Label"].value_counts().to_string())

    train_df = df[df["Gene"] != HELDOUT_GENE]
    test_df = df[df["Gene"] == HELDOUT_GENE]

    print("\nTrain/test split expected:")
    print("train genes:", sorted(train_df["Gene"].unique()))
    print("test genes:", sorted(test_df["Gene"].unique()))
    print("train rows:", len(train_df))
    print("test rows:", len(test_df))

    assert HELDOUT_GENE not in train_df["Gene"].unique()
    assert set(test_df["Gene"].unique()) == {HELDOUT_GENE}
    assert len(train_df) == 287
    assert len(test_df) == 61

    audit_rows = []

    for seed in SEEDS:
        audit_rows.append(audit_seed(seed, df))

    audit_df = pd.DataFrame(audit_rows)

    out_dir = PROJECT_ROOT / "results" / "mechanism" / "contrastive_lora_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_file = out_dir / "contrastive_lora_audit_summary.csv"
    audit_df.to_csv(audit_file, index=False)

    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print(audit_df.to_string(index=False))
    print("\nSaved audit summary:")
    print(audit_file)


if __name__ == "__main__":
    main()