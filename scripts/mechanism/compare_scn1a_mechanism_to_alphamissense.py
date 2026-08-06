from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCN1A_PRED_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scn1a_heldout_validation"
    / "scn1a_heldout_predictions.csv"
)

ALPHAMISSENSE_FILE = (
    PROJECT_ROOT
    / "data"
    / "alphamissense"
    / "AlphaMissense-Search-P35498.tsv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scn1a_alphamissense_mechanism_comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_variant_text(x):
    text = str(x).strip()

    text = text.replace("SCN1A:p.", "")
    text = text.replace("p.", "")
    text = text.replace(" ", "")

    match = re.search(r"([A-Z])(\d+)([A-Z])", text)

    if match is None:
        return None

    return f"{match.group(1)}{match.group(2)}{match.group(3)}"


def find_col(columns, candidates):
    lower_map = {c.lower(): c for c in columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for col in columns:
        col_lower = col.lower()
        for cand in candidates:
            if cand.lower() in col_lower:
                return col

    return None


def read_alphamissense(path):
    print("Reading AlphaMissense file:")
    print(path)

    try:
        alpha = pd.read_csv(path, sep="\t")
        if alpha.shape[1] == 1:
            alpha = pd.read_csv(path)
    except Exception:
        alpha = pd.read_csv(path)

    print("\nAlphaMissense shape:", alpha.shape)
    print("Columns:")
    for col in alpha.columns:
        print(" -", col)

    required_cols = ["a.a.1", "position", "a.a.2", "pathogenicity score"]

    missing = [col for col in required_cols if col not in alpha.columns]

    if missing:
        raise ValueError(
            f"Missing expected AlphaMissense columns: {missing}\n"
            f"Available columns: {list(alpha.columns)}"
        )

    out = pd.DataFrame()

    out["Ref_AA"] = alpha["a.a.1"].astype(str).str.strip().str.upper()
    out["AA_Position"] = pd.to_numeric(alpha["position"], errors="coerce")
    out["Alt_AA"] = alpha["a.a.2"].astype(str).str.strip().str.upper()

    out["AlphaMissense_Score"] = pd.to_numeric(
        alpha["pathogenicity score"],
        errors="coerce",
    )

    if "pathogenicity class" in alpha.columns:
        out["AlphaMissense_Class"] = alpha["pathogenicity class"].astype(str)
    else:
        out["AlphaMissense_Class"] = ""

    out["AlphaMissense_Protein_Change_Raw"] = alpha["protein variant"].astype(str)

    out = out.dropna(
        subset=[
            "Ref_AA",
            "AA_Position",
            "Alt_AA",
            "AlphaMissense_Score",
        ]
    ).copy()

    out["AA_Position"] = out["AA_Position"].astype(int)

    out["Protein_Change"] = (
        out["Ref_AA"]
        + out["AA_Position"].astype(str)
        + out["Alt_AA"]
    )

    out = out.drop_duplicates(subset=["Protein_Change"], keep="first")

    print("\nParsed AlphaMissense rows:", len(out))
    print("Parsed preview:")
    print(
        out[
            [
                "Protein_Change",
                "AlphaMissense_Score",
                "AlphaMissense_Class",
                "AlphaMissense_Protein_Change_Raw",
            ]
        ].head(10).to_string(index=False)
    )

    return out


def evaluate_scores(y_true, scores, threshold=0.5):
    y_true = np.array(y_true)
    scores = np.array(scores)

    preds = (scores >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y_true, preds, zero_division=0),
    }


def bootstrap_auc_ci(y, scores, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(y)
    scores = np.array(scores)

    aucs = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))

        if len(np.unique(y[idx])) < 2:
            continue

        aucs.append(roc_auc_score(y[idx], scores[idx]))

    aucs = np.array(aucs)

    return {
        "auc_boot_mean": float(np.mean(aucs)),
        "auc_ci_low": float(np.percentile(aucs, 2.5)),
        "auc_ci_high": float(np.percentile(aucs, 97.5)),
    }


def bootstrap_auc_difference(y, score_a, score_b, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    y = np.array(y)
    score_a = np.array(score_a)
    score_b = np.array(score_b)

    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))

        if len(np.unique(y[idx])) < 2:
            continue

        auc_a = roc_auc_score(y[idx], score_a[idx])
        auc_b = roc_auc_score(y[idx], score_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)

    return {
        "auc_difference_mean": float(np.mean(diffs)),
        "auc_difference_ci_low": float(np.percentile(diffs, 2.5)),
        "auc_difference_ci_high": float(np.percentile(diffs, 97.5)),
        "fraction_difference_leq_0": float(np.mean(diffs <= 0)),
    }


def permutation_auc_pvalue(y, scores, n_perm=10000, seed=2):
    rng = np.random.default_rng(seed)

    y = np.array(y)
    scores = np.array(scores)

    observed = roc_auc_score(y, scores)

    null = []

    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        null.append(roc_auc_score(y_perm, scores))

    null = np.array(null)

    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def main():
    print("Reading SCN1A held-out prediction table:")
    print(SCN1A_PRED_FILE)

    pred = pd.read_csv(SCN1A_PRED_FILE)
    pred = pred[pred["Gene"] == "SCN1A"].copy()

    pred["Protein_Change"] = pred["Protein_Change"].apply(normalize_variant_text)

    print("\nSCN1A rows:", len(pred))
    print("SCN1A mechanism counts:")
    print(pred["Mechanism_Label"].value_counts().to_string())

    print("\nSCION SCN1A variants:")
    print(pred[["Variant_Key", "Protein_Change", "Mechanism_Label"]].head(20).to_string(index=False))

    alpha = read_alphamissense(ALPHAMISSENSE_FILE)

    merged = pred.merge(
        alpha,
        on="Protein_Change",
        how="left",
    )

    missing = merged[merged["AlphaMissense_Score"].isna()].copy()

    print("\nMerged rows:", len(merged))
    print("AlphaMissense matched:", int(merged["AlphaMissense_Score"].notna().sum()))
    print("AlphaMissense missing:", int(merged["AlphaMissense_Score"].isna().sum()))

    if len(missing) > 0:
        print("\nUnmatched SCION variants:")
        print(missing[["Variant_Key", "Protein_Change", "Mechanism_Label"]].to_string(index=False))
        missing.to_csv(OUTPUT_DIR / "unmatched_scion_scn1a_to_alphamissense.csv", index=False)

    eval_df = merged.dropna(subset=["AlphaMissense_Score"]).copy()

    if len(eval_df) == 0:
        print("\nERROR: 0 variants matched AlphaMissense.")
        print("This means the AlphaMissense parsing still does not match your file format.")
        print("Paste the printed AlphaMissense columns + parsed preview.")
        return

    y = eval_df["Mechanism_Binary"].astype(int).values

    score_columns = {
        "AlphaMissense_raw_pathogenicity_score": eval_df["AlphaMissense_Score"].values,
        "AlphaMissense_reversed_score": -eval_df["AlphaMissense_Score"].values,
        "ESM_mutant_site_mechanism_model": eval_df["mutant_site_gof_score"].values,
        "ESM_wt_mut_site_mechanism_model": eval_df["wt_mut_site_gof_score"].values,
        "ESM_wt_site_mechanism_model": eval_df["wt_site_gof_score"].values,
        "biochem_only_model": eval_df["biochem_only_gof_score"].values,
    }

    rows = []

    for name, scores in score_columns.items():
        metrics = evaluate_scores(y, scores)
        metrics.update(bootstrap_auc_ci(y, scores))
        metrics["permutation_p_auc_greater_than_random"] = permutation_auc_pvalue(y, scores)
        metrics["score_name"] = name
        rows.append(metrics)

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    esm_scores = score_columns["ESM_mutant_site_mechanism_model"]

    diff_rows = []

    for comp_name in [
        "AlphaMissense_raw_pathogenicity_score",
        "AlphaMissense_reversed_score",
        "biochem_only_model",
    ]:
        diff = bootstrap_auc_difference(
            y,
            esm_scores,
            score_columns[comp_name],
        )
        diff["comparison"] = f"ESM_mutant_site_minus_{comp_name}"
        diff_rows.append(diff)

    diff_df = pd.DataFrame(diff_rows)

    score_by_label = (
        eval_df.groupby("Mechanism_Label")["AlphaMissense_Score"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )

    merged.to_csv(OUTPUT_DIR / "scn1a_mechanism_with_alphamissense.csv", index=False)
    results.to_csv(OUTPUT_DIR / "scn1a_gof_lof_score_comparison.csv", index=False)
    diff_df.to_csv(OUTPUT_DIR / "auc_difference_tests.csv", index=False)
    score_by_label.to_csv(OUTPUT_DIR / "alphamissense_score_by_mechanism.csv", index=False)

    qc = {
        "n_scn1a_rows": int(len(pred)),
        "n_with_alphamissense": int(len(eval_df)),
        "n_missing_alphamissense": int(merged["AlphaMissense_Score"].isna().sum()),
        "mechanism_counts_evaluated": eval_df["Mechanism_Label"].value_counts().to_dict(),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("QC")
    print("=" * 100)
    print(json.dumps(qc, indent=4))

    print("\n" + "=" * 100)
    print("ALPHAMISSENSE SCORE BY MECHANISM")
    print("=" * 100)
    print(score_by_label.to_string(index=False))

    print("\n" + "=" * 100)
    print("SCN1A GOF-vs-LOF MECHANISM COMPARISON")
    print("=" * 100)
    print(results.to_string(index=False))

    print("\n" + "=" * 100)
    print("AUC DIFFERENCE TESTS")
    print("=" * 100)
    print(diff_df.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()