from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
)

METADATA_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_mechanism_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))
N_SPLITS = 5


FEATURE_FILES = {
    "biochem_only": None,
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
    "full_site": EMBED_DIR / "full_site_features.npy",
}


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

MOLECULAR_WEIGHT = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G": 75.1, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

CHARGE = {
    "D": -1, "E": -1,
    "K": 1, "R": 1, "H": 0.5,
}

POLAR = set(["D", "E", "K", "R", "H", "N", "Q", "S", "T", "Y", "C"])
AROMATIC = set(["F", "W", "Y", "H"])


def aa_one_hot(aa):
    return [1 if aa == x else 0 for x in AA_LIST]


def build_biochem_features(meta):
    rows = []

    for _, row in meta.iterrows():
        ref = row["Ref_AA"]
        alt = row["Alt_AA"]

        ref_h = HYDROPATHY.get(ref, 0)
        alt_h = HYDROPATHY.get(alt, 0)

        ref_w = MOLECULAR_WEIGHT.get(ref, 0)
        alt_w = MOLECULAR_WEIGHT.get(alt, 0)

        ref_c = CHARGE.get(ref, 0)
        alt_c = CHARGE.get(alt, 0)

        features = []

        # Basic biochemical deltas
        features.extend([
            ref_h,
            alt_h,
            alt_h - ref_h,
            abs(alt_h - ref_h),
            ref_w,
            alt_w,
            alt_w - ref_w,
            abs(alt_w - ref_w),
            ref_c,
            alt_c,
            alt_c - ref_c,
            abs(alt_c - ref_c),
            int(ref in POLAR),
            int(alt in POLAR),
            int(ref in POLAR) != int(alt in POLAR),
            int(ref in AROMATIC),
            int(alt in AROMATIC),
            int(ref in AROMATIC) != int(alt in AROMATIC),
            int(ref == "G"),
            int(alt == "G"),
            int(ref == "P"),
            int(alt == "P"),
        ])

        # Ref and alt amino-acid identity
        features.extend(aa_one_hot(ref))
        features.extend(aa_one_hot(alt))

        rows.append(features)

    return np.array(rows, dtype=float)


def make_model(X_train, seed):
    # For ESM features, reduce dimensionality.
    # For small biochem features, skip PCA.
    steps = [("scaler", StandardScaler())]

    if X_train.shape[1] > 80:
        n_components = min(50, X_train.shape[0] - 2, X_train.shape[1])
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=seed,
                    svd_solver="randomized",
                ),
            )
        )

    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            ),
        )
    )

    return Pipeline(steps)


def evaluate(y_true, scores, threshold=0.5):
    preds = (scores >= threshold).astype(int)

    metrics = {
        "n_test": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "precision_gof": precision_score(y_true, preds, zero_division=0),
        "recall_gof": recall_score(y_true, preds, zero_division=0),
        "f1_gof": f1_score(y_true, preds, zero_division=0),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["average_precision"] = average_precision_score(y_true, scores)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def fit_predict(X, y, train_idx, test_idx, seed):
    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    if len(np.unique(y_train)) < 2:
        return None

    model = make_model(X_train, seed)
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]

    return y_test, scores


def run_random_cv(feature_name, X, y):
    rows = []

    for seed in SEEDS:
        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            result = fit_predict(X, y, train_idx, test_idx, seed)

            if result is None:
                continue

            y_test, scores = result

            metrics = evaluate(y_test, scores)
            metrics.update({
                "experiment": "random_stratified_cv",
                "feature_set": feature_name,
                "seed": seed,
                "fold": fold,
                "heldout_gene": "NA",
            })

            rows.append(metrics)

    return rows


def run_family_position_group_cv(feature_name, X, y, groups):
    rows = []

    for seed in SEEDS:
        sgkf = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
            result = fit_predict(X, y, train_idx, test_idx, seed)

            if result is None:
                continue

            y_test, scores = result

            metrics = evaluate(y_test, scores)
            metrics.update({
                "experiment": "family_alignment_cid_group_cv",
                "feature_set": feature_name,
                "seed": seed,
                "fold": fold,
                "heldout_gene": "NA",
            })

            rows.append(metrics)

    return rows


def run_leave_one_gene_out(feature_name, X, y, meta):
    rows = []

    genes = sorted(meta["Gene"].unique())

    for gene in genes:
        test_idx = np.where(meta["Gene"].values == gene)[0]
        train_idx = np.where(meta["Gene"].values != gene)[0]

        if len(test_idx) < 5:
            continue

        if len(np.unique(y[test_idx])) < 2:
            continue

        result = fit_predict(X, y, train_idx, test_idx, seed=0)

        if result is None:
            continue

        y_test, scores = result

        metrics = evaluate(y_test, scores)
        metrics.update({
            "experiment": "leave_one_gene_out",
            "feature_set": feature_name,
            "seed": 0,
            "fold": 0,
            "heldout_gene": gene,
        })

        rows.append(metrics)

    return rows


def summarize(results):
    df = pd.DataFrame(results)

    metric_cols = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision_gof",
        "recall_gof",
        "f1_gof",
        "macro_f1",
    ]

    summary = (
        df.groupby(["experiment", "feature_set"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary


def main():
    print("Reading metadata:")
    print(METADATA_FILE)

    meta = pd.read_csv(METADATA_FILE)

    y = meta["Mechanism_Binary"].astype(int).values
    groups = meta["Family_Alignment_CID"].astype(int).values

    print("\nDataset:")
    print("Rows:", len(meta))
    print("Mechanism counts:")
    print(meta["Mechanism_Label"].value_counts().to_string())
    print("\nGene × mechanism:")
    print(pd.crosstab(meta["Gene"], meta["Mechanism_Label"]).to_string())

    feature_arrays = {}

    for name, path in FEATURE_FILES.items():
        if name == "biochem_only":
            X = build_biochem_features(meta)
        else:
            X = np.load(path)

        feature_arrays[name] = X
        print(f"\n{name}: {X.shape}")

    all_results = []

    for feature_name, X in feature_arrays.items():
        print("\n" + "=" * 100)
        print("Running feature set:", feature_name)
        print("=" * 100)

        all_results.extend(run_random_cv(feature_name, X, y))
        all_results.extend(run_family_position_group_cv(feature_name, X, y, groups))
        all_results.extend(run_leave_one_gene_out(feature_name, X, y, meta))

    results_df = pd.DataFrame(all_results)
    summary_df = summarize(all_results)

    results_file = OUTPUT_DIR / "mechanism_benchmark_all_results.csv"
    summary_file = OUTPUT_DIR / "mechanism_benchmark_summary.csv"
    gene_file = OUTPUT_DIR / "mechanism_leave_one_gene_out_results.csv"

    results_df.to_csv(results_file, index=False)
    summary_df.to_csv(summary_file, index=False)

    logo = results_df[results_df["experiment"] == "leave_one_gene_out"].copy()
    logo.to_csv(gene_file, index=False)

    qc = {
        "n_variants": int(len(meta)),
        "mechanism_counts": meta["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": meta["Gene"].value_counts().to_dict(),
        "feature_sets": {k: list(v.shape) for k, v in feature_arrays.items()},
        "results_file": str(results_file),
        "summary_file": str(summary_file),
        "leave_one_gene_out_file": str(gene_file),
    }

    with open(OUTPUT_DIR / "mechanism_benchmark_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("SUMMARY BY EXPERIMENT AND FEATURE")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("LEAVE-ONE-GENE-OUT RESULTS")
    print("=" * 100)
    print(logo.sort_values(["heldout_gene", "roc_auc"], ascending=[True, False]).to_string(index=False))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()