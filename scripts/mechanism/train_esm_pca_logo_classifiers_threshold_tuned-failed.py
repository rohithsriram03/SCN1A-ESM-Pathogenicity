from pathlib import Path
import argparse
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

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "esm_pca_logo_classifiers_threshold_tuned"


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


def threshold_predictions(scores, threshold):
    return (scores >= threshold).astype(int)


def get_candidate_thresholds(scores):
    scores = np.asarray(scores)
    unique_scores = np.unique(scores)

    if len(unique_scores) == 1:
        return np.array([unique_scores[0]])

    mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0

    thresholds = np.concatenate(
        [
            [unique_scores[0] - 1e-9],
            mids,
            [unique_scores[-1] + 1e-9],
        ]
    )

    return thresholds


def choose_best_threshold(y_true, scores):
    """
    Choose the threshold that maximizes balanced accuracy.
    Tie-breakers:
      1. higher GoF F1
      2. higher macro F1
      3. threshold closest to default zero
    """
    best = None

    for threshold in get_candidate_thresholds(scores):
        preds = threshold_predictions(scores, threshold)

        bal = balanced_accuracy_score(y_true, preds)
        gof_f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        row = {
            "threshold": float(threshold),
            "balanced_accuracy": float(bal),
            "gof_f1": float(gof_f1),
            "macro_f1": float(macro_f1),
            "distance_from_zero": abs(float(threshold)),
        }

        if best is None:
            best = row
            continue

        current_key = (
            row["balanced_accuracy"],
            row["gof_f1"],
            row["macro_f1"],
            -row["distance_from_zero"],
        )

        best_key = (
            best["balanced_accuracy"],
            best["gof_f1"],
            best["macro_f1"],
            -best["distance_from_zero"],
        )

        if current_key > best_key:
            best = row

    return best


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

    default_preds = pipe.predict(X)

    if hasattr(clf, "predict_proba"):
        scores = pipe.predict_proba(X)[:, 1]
    else:
        scores = pipe.decision_function(X)

    return default_preds, scores


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


def inner_gene_cv_select_config_and_threshold(X_by_feature, y, genes, outer_target_gene, configs):
    """
    For one outer held-out gene:
      - use only non-target genes
      - run inner leave-one-gene validation
      - choose best model config
      - choose best threshold using inner validation scores only
    """
    outer_train_mask = genes != outer_target_gene

    validation_genes = []

    for gene in sorted(set(genes[outer_train_mask])):
        val_mask = genes == gene
        if len(np.unique(y[val_mask])) == 2:
            validation_genes.append(gene)

    all_config_summaries = []
    all_inner_prediction_rows = []

    for config in configs:
        feature_set = config["feature_set"]
        X = X_by_feature[feature_set]

        pooled_y = []
        pooled_scores = []

        fold_rows = []

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

            default_preds, scores = predict_scores(pipe, X[val_mask])

            # Default-threshold metrics.
            default_metrics = evaluate_predictions(y[val_mask], default_preds, scores)

            fold_row = {
                "outer_target_gene": outer_target_gene,
                "inner_validation_gene": val_gene,
                "actual_pca_dim": actual_pca_dim,
                **config,
                "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                "default_roc_auc": default_metrics["roc_auc"],
                "default_gof_f1": default_metrics["gof_f1"],
            }

            fold_rows.append(fold_row)

            pooled_y.append(y[val_mask])
            pooled_scores.append(scores)

            for true_label, score, default_pred in zip(y[val_mask], scores, default_preds):
                all_inner_prediction_rows.append(
                    {
                        "outer_target_gene": outer_target_gene,
                        "inner_validation_gene": val_gene,
                        **config,
                        "actual_pca_dim": actual_pca_dim,
                        "true_label": int(true_label),
                        "score": float(score),
                        "default_pred": int(default_pred),
                    }
                )

        if len(pooled_y) == 0:
            continue

        pooled_y = np.concatenate(pooled_y)
        pooled_scores = np.concatenate(pooled_scores)

        threshold_info = choose_best_threshold(pooled_y, pooled_scores)
        tuned_preds = threshold_predictions(pooled_scores, threshold_info["threshold"])
        tuned_metrics = evaluate_predictions(pooled_y, tuned_preds, pooled_scores)

        default_preds_pooled = np.array([r["default_pred"] for r in all_inner_prediction_rows if r["outer_target_gene"] == outer_target_gene and r["feature_set"] == config["feature_set"] and r["pca_dim"] == config["pca_dim"] and r["model_name"] == config["model_name"] and r["C"] == config["C"] and r["gamma"] == config["gamma"]])
        default_metrics_pooled = evaluate_predictions(pooled_y, default_preds_pooled, pooled_scores)

        summary_row = {
            "outer_target_gene": outer_target_gene,
            **config,
            "n_inner_genes": len(validation_genes),
            "threshold": threshold_info["threshold"],
            "inner_tuned_accuracy": tuned_metrics["accuracy"],
            "inner_tuned_balanced_accuracy": tuned_metrics["balanced_accuracy"],
            "inner_tuned_roc_auc": tuned_metrics["roc_auc"],
            "inner_tuned_gof_f1": tuned_metrics["gof_f1"],
            "inner_tuned_macro_f1": tuned_metrics["macro_f1"],
            "inner_default_accuracy": default_metrics_pooled["accuracy"],
            "inner_default_balanced_accuracy": default_metrics_pooled["balanced_accuracy"],
            "inner_default_roc_auc": default_metrics_pooled["roc_auc"],
            "inner_default_gof_f1": default_metrics_pooled["gof_f1"],
        }

        all_config_summaries.append(summary_row)

    summary = pd.DataFrame(all_config_summaries)

    summary = summary.sort_values(
        ["inner_tuned_balanced_accuracy", "inner_tuned_roc_auc", "inner_tuned_gof_f1"],
        ascending=False,
    )

    best_config = summary.iloc[0].to_dict()

    inner_predictions = pd.DataFrame(all_inner_prediction_rows)

    return best_config, inner_predictions, summary


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

    default_preds, scores = predict_scores(pipe, X[test_mask])

    threshold = float(best_config["threshold"])
    tuned_preds = threshold_predictions(scores, threshold)

    default_metrics = evaluate_predictions(y[test_mask], default_preds, scores)
    tuned_metrics = evaluate_predictions(y[test_mask], tuned_preds, scores)

    tuned_row = {
        "target_gene": target_gene,
        "method": "trained_esm_pca_classifier_threshold_tuned",
        "feature_set": feature_set,
        "model_name": best_config["model_name"],
        "pca_dim_requested": int(best_config["pca_dim"]),
        "pca_dim_actual": int(actual_pca_dim),
        "C": best_config["C"],
        "gamma": best_config["gamma"],
        "threshold": threshold,
        **tuned_metrics,
    }

    default_row = {
        "target_gene": target_gene,
        "method": "trained_esm_pca_classifier_default_threshold",
        "feature_set": feature_set,
        "model_name": best_config["model_name"],
        "pca_dim_requested": int(best_config["pca_dim"]),
        "pca_dim_actual": int(actual_pca_dim),
        "C": best_config["C"],
        "gamma": best_config["gamma"],
        "threshold": 0.0,
        **default_metrics,
    }

    predictions_df = pd.DataFrame(
        {
            "target_gene": target_gene,
            "true_label": y[test_mask],
            "score": scores,
            "default_pred": default_preds,
            "threshold_tuned_pred": tuned_preds,
        }
    )

    return tuned_row, default_row, predictions_df


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

    all_inner_predictions = []
    all_inner_summaries = []
    tuned_test_rows = []
    default_test_rows = []
    all_test_predictions = []
    frozen_rows = []

    for target_gene in target_genes:
        print("\n" + "=" * 90)
        print(f"OUTER HELD-OUT TARGET GENE: {target_gene}")
        print("=" * 90)

        target_mask = genes == target_gene

        if len(np.unique(y[target_mask])) < 2:
            print(f"Skipping {target_gene}: target gene does not contain both GoF and LoF.")
            continue

        best_config, inner_predictions, inner_summary = inner_gene_cv_select_config_and_threshold(
            X_by_feature=X_by_feature,
            y=y,
            genes=genes,
            outer_target_gene=target_gene,
            configs=configs,
        )

        all_inner_predictions.append(inner_predictions)
        all_inner_summaries.append(inner_summary)

        print("\nBest config + threshold from inner leave-one-gene validation:")
        print(json.dumps(best_config, indent=2, default=str))

        tuned_row, default_row, test_predictions = train_final_and_test(
            X_by_feature=X_by_feature,
            y=y,
            genes=genes,
            target_gene=target_gene,
            best_config=best_config,
        )

        tuned_test_rows.append(tuned_row)
        default_test_rows.append(default_row)
        all_test_predictions.append(test_predictions)

        print("\nFinal held-out test result with threshold tuning:")
        print(pd.DataFrame([tuned_row]).to_string(index=False))

        print("\nSame final model with default threshold:")
        print(pd.DataFrame([default_row]).to_string(index=False))

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

    tuned_df = pd.DataFrame(tuned_test_rows)
    default_df = pd.DataFrame(default_test_rows)
    frozen_df = pd.concat(frozen_rows, ignore_index=True)
    inner_predictions_df = pd.concat(all_inner_predictions, ignore_index=True)
    inner_summary_df = pd.concat(all_inner_summaries, ignore_index=True)
    test_predictions_df = pd.concat(all_test_predictions, ignore_index=True)

    tuned_df.to_csv(OUT_DIR / "threshold_tuned_final_logo_results.csv", index=False)
    default_df.to_csv(OUT_DIR / "default_threshold_final_logo_results.csv", index=False)
    frozen_df.to_csv(OUT_DIR / "frozen_esm_knn_logo_results.csv", index=False)
    inner_predictions_df.to_csv(OUT_DIR / "inner_cv_prediction_scores.csv", index=False)
    inner_summary_df.to_csv(OUT_DIR / "inner_cv_summary_by_config_threshold_tuned.csv", index=False)
    test_predictions_df.to_csv(OUT_DIR / "heldout_test_prediction_scores.csv", index=False)

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

    for target_gene in sorted(tuned_df["target_gene"].unique()):
        tuned_row = tuned_df[tuned_df["target_gene"] == target_gene].iloc[0]
        default_row = default_df[default_df["target_gene"] == target_gene].iloc[0]
        frozen_row = best_frozen[best_frozen["target_gene"] == target_gene].iloc[0]

        comparison_rows.append(
            {
                "target_gene": target_gene,
                "feature_set": tuned_row["feature_set"],
                "model": tuned_row["model_name"],
                "pca_dim": tuned_row["pca_dim_actual"],
                "threshold": tuned_row["threshold"],
                "tuned_balanced_accuracy": tuned_row["balanced_accuracy"],
                "default_balanced_accuracy": default_row["balanced_accuracy"],
                "best_frozen_balanced_accuracy": frozen_row["balanced_accuracy"],
                "delta_tuned_vs_default_balanced_accuracy": tuned_row["balanced_accuracy"] - default_row["balanced_accuracy"],
                "delta_tuned_vs_frozen_balanced_accuracy": tuned_row["balanced_accuracy"] - frozen_row["balanced_accuracy"],
                "tuned_roc_auc": tuned_row["roc_auc"],
                "default_roc_auc": default_row["roc_auc"],
                "best_frozen_roc_auc": frozen_row["roc_auc"],
                "delta_tuned_vs_frozen_roc_auc": tuned_row["roc_auc"] - frozen_row["roc_auc"],
                "tuned_gof_f1": tuned_row["gof_f1"],
                "default_gof_f1": default_row["gof_f1"],
                "best_frozen_gof_f1": frozen_row["gof_f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "threshold_tuned_vs_default_vs_frozen_by_target_gene.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "method": "threshold_tuned_trained_classifier",
                "n_target_genes": tuned_df["target_gene"].nunique(),
                "mean_balanced_accuracy": tuned_df["balanced_accuracy"].mean(),
                "mean_roc_auc": tuned_df["roc_auc"].mean(),
                "mean_gof_f1": tuned_df["gof_f1"].mean(),
                "mean_accuracy": tuned_df["accuracy"].mean(),
            },
            {
                "method": "default_threshold_trained_classifier",
                "n_target_genes": default_df["target_gene"].nunique(),
                "mean_balanced_accuracy": default_df["balanced_accuracy"].mean(),
                "mean_roc_auc": default_df["roc_auc"].mean(),
                "mean_gof_f1": default_df["gof_f1"].mean(),
                "mean_accuracy": default_df["accuracy"].mean(),
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

    overall.to_csv(OUT_DIR / "overall_threshold_tuned_vs_default_vs_frozen_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("THRESHOLD-TUNED VS DEFAULT VS BEST FROZEN BY TARGET GENE")
    print("=" * 90)

    cols = [
        "target_gene",
        "feature_set",
        "model",
        "pca_dim",
        "threshold",
        "tuned_balanced_accuracy",
        "default_balanced_accuracy",
        "best_frozen_balanced_accuracy",
        "delta_tuned_vs_default_balanced_accuracy",
        "delta_tuned_vs_frozen_balanced_accuracy",
        "tuned_roc_auc",
        "best_frozen_roc_auc",
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