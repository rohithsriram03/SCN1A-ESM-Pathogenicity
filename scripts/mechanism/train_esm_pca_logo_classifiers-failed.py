from pathlib import Path
import argparse
import itertools
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"

FEATURE_FILES = {
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}

META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "esm_pca_logo_classifiers"


def get_labels(meta: pd.DataFrame) -> np.ndarray:
    if "Mechanism_Binary" in meta.columns:
        return meta["Mechanism_Binary"].astype(int).to_numpy()

    if "Mechanism_Label" in meta.columns:
        return meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1}).astype(int).to_numpy()

    raise ValueError("Could not find Mechanism_Binary or Mechanism_Label.")


def safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, scores)
    except ValueError:
        return np.nan


def safe_ap(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, scores)
    except ValueError:
        return np.nan


def evaluate_predictions(y_true, preds, scores):
    return {
        "n_test": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "gof_precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "gof_recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "roc_auc": safe_auc(y_true, scores),
        "average_precision": safe_ap(y_true, scores),
    }


def make_classifier(model_name, C=1.0, gamma="scale"):
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=10000,
            solver="liblinear",
        )

    if model_name == "linear_svm":
        return SVC(
            kernel="linear",
            C=C,
            class_weight="balanced",
            probability=False,
        )

    if model_name == "rbf_svm":
        return SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",
            probability=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name, pca_dim, C, gamma, n_train, n_features):
    actual_pca_dim = min(pca_dim, n_train - 1, n_features)

    if actual_pca_dim < 2:
        return None, None

    clf = make_classifier(model_name=model_name, C=C, gamma=gamma)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=actual_pca_dim, random_state=0)),
            ("clf", clf),
        ]
    )

    return pipe, actual_pca_dim


def predict_scores(pipe, X):
    clf = pipe.named_steps["clf"]

    preds = pipe.predict(X)

    if hasattr(clf, "predict_proba"):
        scores = pipe.predict_proba(X)[:, 1]
    else:
        scores = pipe.decision_function(X)

    return preds, scores


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def evaluate_frozen_knn(X, y, genes, target_gene, k_values=(1, 3, 5)):
    X = l2_normalize(X)

    query_mask = genes == target_gene
    candidate_mask = genes != target_gene

    X_query = X[query_mask]
    y_query = y[query_mask]

    X_candidate = X[candidate_mask]
    y_candidate = y[candidate_mask]

    sim = X_query @ X_candidate.T

    rows = []

    for k in k_values:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labels = y_candidate[topk_idx]

        scores = topk_labels.mean(axis=1)
        preds = (scores >= 0.5).astype(int)

        row = evaluate_predictions(y_query, preds, scores)
        row["k"] = k
        rows.append(row)

    return pd.DataFrame(rows)


def get_evaluable_target_genes(genes, y):
    out = []

    for gene in sorted(set(genes)):
        mask = genes == gene
        labels = y[mask]

        # Need both classes to compute balanced accuracy and AUC meaningfully.
        if len(np.unique(labels)) == 2:
            out.append(gene)

    return out


def make_config_grid(feature_sets, pca_dims, models):
    configs = []

    for feature_set in feature_sets:
        for pca_dim in pca_dims:
            for model_name in models:
                if model_name in ["logistic_regression", "linear_svm"]:
                    for C in [0.01, 0.1, 1.0, 10.0]:
                        configs.append(
                            {
                                "feature_set": feature_set,
                                "pca_dim": pca_dim,
                                "model_name": model_name,
                                "C": C,
                                "gamma": "none",
                            }
                        )

                elif model_name == "rbf_svm":
                    for C in [0.1, 1.0, 10.0]:
                        for gamma in ["scale", 0.01, 0.1]:
                            configs.append(
                                {
                                    "feature_set": feature_set,
                                    "pca_dim": pca_dim,
                                    "model_name": model_name,
                                    "C": C,
                                    "gamma": gamma,
                                }
                            )

    return configs


def inner_gene_cv_select_config(X_by_feature, y, genes, outer_target_gene, configs):
    """
    For one outer held-out gene, choose model settings using only non-target genes.
    This avoids tuning on the final test gene.
    """
    outer_train_mask = genes != outer_target_gene

    validation_genes = []

    for gene in sorted(set(genes[outer_train_mask])):
        val_mask = genes == gene
        if len(np.unique(y[val_mask])) == 2:
            validation_genes.append(gene)

    all_inner_rows = []

    for config in configs:
        feature_set = config["feature_set"]
        X = X_by_feature[feature_set]

        val_metrics = []

        for val_gene in validation_genes:
            train_mask = (genes != outer_target_gene) & (genes != val_gene)
            val_mask = genes == val_gene

            if len(np.unique(y[train_mask])) < 2:
                continue

            if len(np.unique(y[val_mask])) < 2:
                continue

            pipe, actual_pca_dim = build_pipeline(
                model_name=config["model_name"],
                pca_dim=config["pca_dim"],
                C=config["C"],
                gamma=config["gamma"],
                n_train=int(train_mask.sum()),
                n_features=X.shape[1],
            )

            if pipe is None:
                continue

            pipe.fit(X[train_mask], y[train_mask])

            preds, scores = predict_scores(pipe, X[val_mask])
            metrics = evaluate_predictions(y[val_mask], preds, scores)

            row = {
                "outer_target_gene": outer_target_gene,
                "inner_validation_gene": val_gene,
                "actual_pca_dim": actual_pca_dim,
                **config,
                **metrics,
            }

            all_inner_rows.append(row)
            val_metrics.append(metrics)

        if len(val_metrics) == 0:
            continue

    inner_df = pd.DataFrame(all_inner_rows)

    group_cols = ["feature_set", "pca_dim", "model_name", "C", "gamma"]

    summary = (
        inner_df.groupby(group_cols)
        .agg(
            n_inner_genes=("inner_validation_gene", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_gof_f1=("gof_f1", "mean"),
            mean_average_precision=("average_precision", "mean"),
        )
        .reset_index()
        .sort_values(
            ["mean_balanced_accuracy", "mean_roc_auc", "mean_gof_f1"],
            ascending=False,
        )
    )

    best_config = summary.iloc[0].to_dict()

    return best_config, inner_df, summary


def train_final_and_test(X_by_feature, y, genes, target_gene, best_config):
    feature_set = best_config["feature_set"]
    X = X_by_feature[feature_set]

    train_mask = genes != target_gene
    test_mask = genes == target_gene

    pipe, actual_pca_dim = build_pipeline(
        model_name=best_config["model_name"],
        pca_dim=int(best_config["pca_dim"]),
        C=float(best_config["C"]),
        gamma=best_config["gamma"],
        n_train=int(train_mask.sum()),
        n_features=X.shape[1],
    )

    pipe.fit(X[train_mask], y[train_mask])

    preds, scores = predict_scores(pipe, X[test_mask])
    metrics = evaluate_predictions(y[test_mask], preds, scores)

    row = {
        "target_gene": target_gene,
        "method": "trained_esm_pca_classifier",
        "feature_set": feature_set,
        "model_name": best_config["model_name"],
        "pca_dim_requested": int(best_config["pca_dim"]),
        "pca_dim_actual": int(actual_pca_dim),
        "C": best_config["C"],
        "gamma": best_config["gamma"],
        **metrics,
    }

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=["wt_mut_site"],
        choices=list(FEATURE_FILES.keys()),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "linear_svm", "rbf_svm"],
        choices=["logistic_regression", "linear_svm", "rbf_svm"],
    )

    parser.add_argument(
        "--pca_dims",
        nargs="+",
        type=int,
        default=[8, 16, 32, 64, 128],
    )

    parser.add_argument(
        "--target_genes",
        nargs="+",
        default=["auto"],
    )

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata:")
    print(META_FILE)

    meta = pd.read_csv(META_FILE)
    y = get_labels(meta)
    genes = meta["Gene"].astype(str).to_numpy()

    print("\nLoading feature matrices:")
    X_by_feature = {}

    for feature_set in args.feature_sets:
        p = FEATURE_FILES[feature_set]
        print(f"{feature_set}: {p}")
        X_by_feature[feature_set] = np.load(p)

    if args.target_genes == ["auto"]:
        target_genes = get_evaluable_target_genes(genes, y)
    else:
        target_genes = args.target_genes

    print("\nTarget genes:")
    print(target_genes)

    print("\nLabel counts by gene:")
    label_table = (
        pd.DataFrame({"Gene": genes, "Label": y})
        .assign(Mechanism=lambda d: d["Label"].map({0: "LOF", 1: "GOF"}))
        .pivot_table(index="Gene", columns="Mechanism", values="Label", aggfunc="count", fill_value=0)
        .reset_index()
    )
    print(label_table.to_string(index=False))

    configs = make_config_grid(
        feature_sets=args.feature_sets,
        pca_dims=args.pca_dims,
        models=args.models,
    )

    print(f"\nNumber of configs to test per target gene: {len(configs)}")

    all_inner_rows = []
    all_inner_summaries = []
    final_test_rows = []
    frozen_rows = []

    for target_gene in target_genes:
        print("\n" + "=" * 90)
        print(f"OUTER HELD-OUT TARGET GENE: {target_gene}")
        print("=" * 90)

        target_mask = genes == target_gene

        if len(np.unique(y[target_mask])) < 2:
            print(f"Skipping {target_gene}: target gene does not contain both GoF and LoF.")
            continue

        best_config, inner_df, inner_summary = inner_gene_cv_select_config(
            X_by_feature=X_by_feature,
            y=y,
            genes=genes,
            outer_target_gene=target_gene,
            configs=configs,
        )

        inner_df["outer_target_gene"] = target_gene
        inner_summary["outer_target_gene"] = target_gene

        all_inner_rows.append(inner_df)
        all_inner_summaries.append(inner_summary)

        print("\nBest config from inner leave-one-gene validation:")
        print(json.dumps(best_config, indent=2, default=str))

        final_row = train_final_and_test(
            X_by_feature=X_by_feature,
            y=y,
            genes=genes,
            target_gene=target_gene,
            best_config=best_config,
        )

        final_test_rows.append(final_row)

        print("\nFinal held-out test result:")
        print(pd.DataFrame([final_row]).to_string(index=False))

        for feature_set in args.feature_sets:
            frozen_df = evaluate_frozen_knn(
                X=X_by_feature[feature_set],
                y=y,
                genes=genes,
                target_gene=target_gene,
                k_values=(1, 3, 5),
            )

            frozen_df["target_gene"] = target_gene
            frozen_df["method"] = "frozen_esm_knn"
            frozen_df["feature_set"] = feature_set
            frozen_rows.append(frozen_df)

    final_test_df = pd.DataFrame(final_test_rows)
    frozen_df = pd.concat(frozen_rows, ignore_index=True)

    inner_all_df = pd.concat(all_inner_rows, ignore_index=True)
    inner_summary_all_df = pd.concat(all_inner_summaries, ignore_index=True)

    final_test_df.to_csv(OUT_DIR / "trained_classifier_final_logo_results.csv", index=False)
    frozen_df.to_csv(OUT_DIR / "frozen_esm_knn_logo_results.csv", index=False)
    inner_all_df.to_csv(OUT_DIR / "inner_cv_all_results.csv", index=False)
    inner_summary_all_df.to_csv(OUT_DIR / "inner_cv_summary_by_config.csv", index=False)

    # Choose best frozen kNN row per target gene for fair comparison.
    best_frozen = (
        frozen_df.sort_values(
            ["target_gene", "balanced_accuracy", "roc_auc", "gof_f1"],
            ascending=[True, False, False, False],
        )
        .groupby("target_gene")
        .head(1)
        .copy()
    )

    comparison_rows = []

    for target_gene in sorted(final_test_df["target_gene"].unique()):
        trained_row = final_test_df[final_test_df["target_gene"] == target_gene].iloc[0]
        frozen_row = best_frozen[best_frozen["target_gene"] == target_gene].iloc[0]

        comparison_rows.append(
            {
                "target_gene": target_gene,
                "trained_feature_set": trained_row["feature_set"],
                "trained_model": trained_row["model_name"],
                "trained_pca_dim": trained_row["pca_dim_actual"],
                "trained_balanced_accuracy": trained_row["balanced_accuracy"],
                "trained_roc_auc": trained_row["roc_auc"],
                "trained_gof_f1": trained_row["gof_f1"],
                "best_frozen_feature_set": frozen_row["feature_set"],
                "best_frozen_k": frozen_row["k"],
                "best_frozen_balanced_accuracy": frozen_row["balanced_accuracy"],
                "best_frozen_roc_auc": frozen_row["roc_auc"],
                "best_frozen_gof_f1": frozen_row["gof_f1"],
                "delta_balanced_accuracy": trained_row["balanced_accuracy"] - frozen_row["balanced_accuracy"],
                "delta_roc_auc": trained_row["roc_auc"] - frozen_row["roc_auc"],
                "delta_gof_f1": trained_row["gof_f1"] - frozen_row["gof_f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "trained_vs_best_frozen_by_target_gene.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "method": "trained_esm_pca_classifier",
                "n_target_genes": final_test_df["target_gene"].nunique(),
                "mean_balanced_accuracy": final_test_df["balanced_accuracy"].mean(),
                "mean_roc_auc": final_test_df["roc_auc"].mean(),
                "mean_gof_f1": final_test_df["gof_f1"].mean(),
                "mean_accuracy": final_test_df["accuracy"].mean(),
            },
            {
                "method": "best_frozen_esm_knn_per_target",
                "n_target_genes": best_frozen["target_gene"].nunique(),
                "mean_balanced_accuracy": best_frozen["balanced_accuracy"].mean(),
                "mean_roc_auc": best_frozen["roc_auc"].mean(),
                "mean_gof_f1": best_frozen["gof_f1"].mean(),
                "mean_accuracy": best_frozen["accuracy"].mean(),
            },
        ]
    )

    overall.to_csv(OUT_DIR / "overall_trained_vs_frozen_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("TRAINED VS BEST FROZEN BY TARGET GENE")
    print("=" * 90)

    cols = [
        "target_gene",
        "trained_model",
        "trained_pca_dim",
        "trained_balanced_accuracy",
        "best_frozen_k",
        "best_frozen_balanced_accuracy",
        "delta_balanced_accuracy",
        "trained_roc_auc",
        "best_frozen_roc_auc",
        "delta_roc_auc",
    ]

    print(comparison_df[cols].to_string(index=False))

    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)
    print(overall.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()