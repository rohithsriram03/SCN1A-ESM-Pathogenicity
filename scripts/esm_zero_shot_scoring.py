from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmarking"
INPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_with_alphamissense.csv"

FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"

OUTPUT_DIR = BENCHMARK_DIR / "esm_zero_shot"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MODEL_SHORT_NAME = "esm2_8M"

WINDOW_SIZE = 1000
BATCH_SIZE = 8

OUTPUT_FILE = OUTPUT_DIR / f"{MODEL_SHORT_NAME}_zero_shot_scores.csv"
METRICS_FILE = OUTPUT_DIR / f"{MODEL_SHORT_NAME}_zero_shot_metrics.json"
CATEGORY_SUMMARY_FILE = OUTPUT_DIR / f"{MODEL_SHORT_NAME}_zero_shot_by_functional_category.csv"
THRESHOLD_SWEEP_FILE = OUTPUT_DIR / f"{MODEL_SHORT_NAME}_zero_shot_threshold_sweep.csv"


VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            lines.append(line.strip())

    return "".join(lines)


def make_centered_window(sequence, position_1indexed, window_size):
    pos0 = int(position_1indexed) - 1

    half = window_size // 2
    start = max(0, pos0 - half)
    end = min(len(sequence), start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = sequence[start:end]
    local_idx = pos0 - start

    return window, local_idx, start, end


def prepare_masked_sequences(df, wt_sequence, tokenizer):
    records = []

    for idx, row in df.iterrows():
        ref = str(row["Ref_AA"]).strip().upper()
        alt = str(row["Alt_AA"]).strip().upper()
        position = int(row["AA_Position"])

        status = "OK"
        masked_sequence = None
        local_idx = None
        window_start = None
        window_end = None

        if ref not in VALID_AAS or alt not in VALID_AAS:
            status = "Invalid amino acid"
        else:
            window, local_idx, window_start, window_end = make_centered_window(
                wt_sequence,
                position,
                WINDOW_SIZE,
            )

            observed_ref = window[local_idx]

            if observed_ref != ref:
                status = f"Reference mismatch: expected {ref}, observed {observed_ref}"
            else:
                masked_sequence = (
                    window[:local_idx]
                    + tokenizer.mask_token
                    + window[local_idx + 1:]
                )

        records.append({
            "row_index": idx,
            "Variant_Key": row.get("Variant_Key", ""),
            "Protein_Change_Parsed": row.get("Protein_Change_Parsed", ""),
            "AA_Position": position,
            "Ref_AA": ref,
            "Alt_AA": alt,
            "masked_sequence": masked_sequence,
            "local_idx": local_idx,
            "window_start_0indexed": window_start,
            "window_end_0indexed": window_end,
            "ZeroShot_Status": status,
        })

    return pd.DataFrame(records)


def compute_zero_shot_scores(prepared, tokenizer, model, device):
    scored_rows = []

    ok_df = prepared[prepared["ZeroShot_Status"] == "OK"].copy()

    for start in tqdm(range(0, len(ok_df), BATCH_SIZE), desc="Scoring variants"):
        batch = ok_df.iloc[start:start + BATCH_SIZE].copy()

        sequences = batch["masked_sequence"].tolist()

        encoded = tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

        mask_positions = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=False)

        if mask_positions.shape[0] != len(batch):
            raise RuntimeError(
                f"Expected one mask per sequence, but found {mask_positions.shape[0]} masks "
                f"for batch size {len(batch)}."
            )

        for batch_i, (_, row) in enumerate(batch.iterrows()):
            mask_row = mask_positions[batch_i]
            sequence_index = int(mask_row[0].item())
            mask_index = int(mask_row[1].item())

            ref = row["Ref_AA"]
            alt = row["Alt_AA"]

            ref_token_id = tokenizer.convert_tokens_to_ids(ref)
            alt_token_id = tokenizer.convert_tokens_to_ids(alt)

            logp_ref = float(log_probs[sequence_index, mask_index, ref_token_id].cpu().item())
            logp_alt = float(log_probs[sequence_index, mask_index, alt_token_id].cpu().item())

            llr_alt_minus_ref = logp_alt - logp_ref

            # Higher means more damaging/pathogenic-like:
            pathogenic_score = logp_ref - logp_alt

            scored_rows.append({
                "row_index": int(row["row_index"]),
                "ESM_Model": MODEL_NAME,
                "ESM_ZeroShot_LogP_Ref": logp_ref,
                "ESM_ZeroShot_LogP_Alt": logp_alt,
                "ESM_ZeroShot_LLR_AltMinusRef": llr_alt_minus_ref,
                "ESM_ZeroShot_Pathogenic_Score": pathogenic_score,
            })

    return pd.DataFrame(scored_rows)


def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["Benign", "Pathogenic"],
            output_dict=True,
            zero_division=0,
        ),
    }


def threshold_sweep(y_true, scores):
    rows = []

    # Use score quantiles instead of fixed 0-1 because zero-shot scores are not probabilities.
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 1001)))

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    return pd.DataFrame(rows)


def category_summary(df):
    rows = []

    for category, sub in df.groupby("Mapped_Functional_Category"):
        sub = sub.dropna(subset=["ESM_ZeroShot_Pathogenic_Score"]).copy()

        if len(sub) == 0:
            continue

        if sub["Binary_Label"].nunique() < 2:
            auc = np.nan
            ap = np.nan
        else:
            auc = roc_auc_score(
                sub["Binary_Label"],
                sub["ESM_ZeroShot_Pathogenic_Score"],
            )
            ap = average_precision_score(
                sub["Binary_Label"],
                sub["ESM_ZeroShot_Pathogenic_Score"],
            )

        rows.append({
            "Mapped_Functional_Category": category,
            "n": len(sub),
            "n_benign": int((sub["Binary_Label"] == 0).sum()),
            "n_pathogenic": int((sub["Binary_Label"] == 1).sum()),
            "mean_zero_shot_score": float(sub["ESM_ZeroShot_Pathogenic_Score"].mean()),
            "median_zero_shot_score": float(sub["ESM_ZeroShot_Pathogenic_Score"].median()),
            "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
            "average_precision": float(ap) if not np.isnan(ap) else np.nan,
        })

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def main():
    print("Reading benchmark:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nReading WT sequence:")
    print(FASTA_FILE)

    wt_sequence = read_fasta(FASTA_FILE)

    print("WT length:", len(wt_sequence))

    print("\nLoading tokenizer/model:")
    print(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    prepared = prepare_masked_sequences(df, wt_sequence, tokenizer)

    print("\nPrepared variants:")
    print(prepared["ZeroShot_Status"].value_counts().to_string())

    scored = compute_zero_shot_scores(prepared, tokenizer, model, device)

    merged = df.copy()
    merged["row_index"] = merged.index

    merged = merged.merge(
        prepared.drop(columns=["masked_sequence"]),
        on=[
            "row_index",
            "Variant_Key",
            "Protein_Change_Parsed",
            "AA_Position",
            "Ref_AA",
            "Alt_AA",
        ],
        how="left",
    )

    merged = merged.merge(
        scored,
        on="row_index",
        how="left",
    )

    merged.to_csv(OUTPUT_FILE, index=False)

    df_eval = merged.dropna(
        subset=["ESM_ZeroShot_Pathogenic_Score", "Binary_Label"]
    ).copy()

    df_eval["Binary_Label"] = df_eval["Binary_Label"].astype(int)

    y_true = df_eval["Binary_Label"].values
    scores = df_eval["ESM_ZeroShot_Pathogenic_Score"].values

    natural_threshold = 0.0

    natural_metrics = compute_metrics(
        y_true,
        scores,
        threshold=natural_threshold,
    )

    sweep = threshold_sweep(y_true, scores)
    sweep.to_csv(THRESHOLD_SWEEP_FILE, index=False)

    best_row = sweep.sort_values("f1", ascending=False).iloc[0]

    optimized_metrics = compute_metrics(
        y_true,
        scores,
        threshold=float(best_row["threshold"]),
    )

    cat_summary = category_summary(df_eval)
    cat_summary.to_csv(CATEGORY_SUMMARY_FILE, index=False)

    score_by_label = (
        df_eval
        .groupby("Label_Name")["ESM_ZeroShot_Pathogenic_Score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    metrics = {
        "model_name": MODEL_NAME,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "input_file": str(INPUT_FILE),
        "n_total_rows": int(len(df)),
        "n_scored_rows": int(len(df_eval)),
        "n_unscored_rows": int(len(df) - len(df_eval)),
        "status_counts": prepared["ZeroShot_Status"].value_counts().to_dict(),
        "natural_threshold_zero_metrics": natural_metrics,
        "best_f1_threshold": float(best_row["threshold"]),
        "optimized_f1_threshold_metrics": optimized_metrics,
        "score_by_label": score_by_label.to_dict(orient="records"),
        "output_scores_file": str(OUTPUT_FILE),
        "output_category_summary_file": str(CATEGORY_SUMMARY_FILE),
        "output_threshold_sweep_file": str(THRESHOLD_SWEEP_FILE),
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 100)
    print("ESM ZERO-SHOT NATURAL THRESHOLD METRICS")
    print("=" * 100)
    print(json.dumps(natural_metrics, indent=4))

    print("\n" + "=" * 100)
    print("ESM ZERO-SHOT OPTIMIZED F1 THRESHOLD METRICS")
    print("=" * 100)
    print(json.dumps(optimized_metrics, indent=4))

    print("\n" + "=" * 100)
    print("SCORE BY LABEL")
    print("=" * 100)
    print(score_by_label.to_string(index=False))

    print("\n" + "=" * 100)
    print("FUNCTIONAL CATEGORY SUMMARY")
    print("=" * 100)
    print(cat_summary.to_string(index=False))

    print("\nSaved scores:")
    print(OUTPUT_FILE)

    print("\nSaved metrics:")
    print(METRICS_FILE)


if __name__ == "__main__":
    main()