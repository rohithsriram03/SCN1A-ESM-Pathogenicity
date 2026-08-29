from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "esm_knn_score_fusion_logo"


def get_labels(meta: pd.DataFrame) -> np.ndarray:
    if "Mechanism_Binary" in meta.columns:
        return meta["Mechanism_Binary"].astype(int).to_numpy()

    if "Mechanism_Label" in meta.columns:
        return meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1}).astype(int).to_numpy()

    raise ValueError("Could not find Mechanism_Binary or Mechanism_Label.")


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


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


def evaluate_from_scores(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)

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


def candidate_thresholds(scores):
    scores = np.asarray(scores)
    unique_scores = np.unique(scores)

    if len(unique_scores) == 1:
        return np.array([unique_scores[0]])

    mids = (unique_scores[:-1] + unique_scores[1:]) / 2.0

    return np.concatenate(
        [
            [unique_scores[0] - 1e-9],
            mids,
            [unique_scores[-1] + 1e-9],
        ]
    )


def choose_best_threshold(y_true, scores):
    best = None

    for threshold in candidate_thresholds(scores):
        metrics = evaluate_from_scores(y_true, scores, threshold)

        row = {
            "threshold": float(threshold),
            "balanced_accuracy": metrics["balanced_accuracy"],
            "gof_f1": metrics["gof_f1"],
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        }

        if best is None:
            best = row
            continue

        key = (
            row["balanced_accuracy"],
            row["gof_f1"],
            row["macro_f1"],
            row["accuracy"],
            -abs(row["threshold"] - 0.5),
        )

        best_key = (
            best["balanced_accuracy"],
            best["gof_f1"],
            best["macro_f1"],
            best["accuracy"],
            -abs(best["threshold"] - 0.5),
        )

        if key > best_key:
            best = row

    return best


def make_knn_score_features(
    X_norm_by_feature,
    y,
    genes,
    query_genes,
    candidate_gene_pool,
    k_values,
):
    """
    Builds one feature row per query variant.

    For each query gene:
      - query variants come from query_genes
      - neighbor candidates come from candidate_gene_pool
      - same-gene candidates are excluded
    """
    all_rows = []
    all_indices = []

    candidate_gene_pool = set(candidate_gene_pool)

    for query_gene in query_genes:
        query_idx = np.where(genes == query_gene)[0]
        candidate_idx = np.array(
            [
                i
                for i, g in enumerate(genes)
                if (g in candidate_gene_pool) and (g != query_gene)
            ],
            dtype=int,
        )

        if len(query_idx) == 0:
            continue

        if len(candidate_idx) == 0:
            raise ValueError(f"No candidate neighbors available for query gene {query_gene}")

        feature_block = {}

        for feature_set, Xn in X_norm_by_feature.items():
            Xq = Xn[query_idx]
            Xc = Xn[candidate_idx]
            yc = y[candidate_idx]

            sim = Xq @ Xc.T
            order = np.argsort(-sim, axis=1)

            top1_idx = order[:, 0]
            top1_sim = sim[np.arange(sim.shape[0]), top1_idx]
            top1_label = yc[top1_idx]

            feature_block[f"{feature_set}__top1_sim"] = top1_sim
            feature_block[f"{feature_set}__top1_gof_label"] = top1_label

            if sim.shape[1] >= 2:
                top2_idx = order[:, 1]
                top2_sim = sim[np.arange(sim.shape[0]), top2_idx]
                feature_block[f"{feature_set}__top1_top2_sim_gap"] = top1_sim - top2_sim
            else:
                feature_block[f"{feature_set}__top1_top2_sim_gap"] = np.zeros_like(top1_sim)

            for k in k_values:
                k_eff = min(k, sim.shape[1])
                topk = order[:, :k_eff]

                topk_labels = yc[topk]
                topk_sims = np.take_along_axis(sim, topk, axis=1)

                gof_score = topk_labels.mean(axis=1)
                mean_sim = topk_sims.mean(axis=1)
                min_sim = topk_sims.min(axis=1)
                confidence = np.abs(gof_score - 0.5)

                feature_block[f"{feature_set}__k{k}__gof_neighbor_score"] = gof_score
                feature_block[f"{feature_set}__k{k}__mean_sim"] = mean_sim
                feature_block[f"{feature_set}__k{k}__min_sim"] = min_sim
                feature_block[f"{feature_set}__k{k}__confidence"] = confidence

        df = pd.DataFrame(feature_block)
        df["query_gene"] = query_gene
        df["variant_index"] = query_idx

        all_rows.append(df)
        all_indices.extend(query_idx.tolist())

    out = pd.concat(all_rows, ignore_index=True)

    # Keep features only, but preserve variant order.
    y_out = y[out["variant_index"].to_numpy()]
    genes_out = genes[out["variant_index"].to_numpy()]
    idx_out = out["variant_index"].to_numpy()

    feature_df = out.drop(columns=["query_gene", "variant_index"])

    return feature_df, y_out, genes_out, idx_out


def make_model(config):
    model_name = config["model_name"]

    if model_name == "logistic_regression":
        clf = LogisticRegression(
            C=float(config["C"]),
            class_weight="balanced",
            max_iter=10000,
            solver="liblinear",
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if model_name == "linear_svm":
        clf = SVC(
            kernel="linear",
            C=float(config["C"]),
            class_weight="balanced",
            probability=False,
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if model_name == "rbf_svm":
        clf = SVC(
            kernel="rbf",
            C=float(config["C"]),
            gamma=config["gamma"],
            class_weight="balanced",
            probability=False,
        )

        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )

    if model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=config["max_depth"],
            min_samples_leaf=int(config["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        )

        return clf

    raise ValueError(f"Unknown model: {model_name}")


def predict_model_scores(model, X):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        default_threshold = 0.5
    else:
        scores = model.decision_function(X)
        default_threshold = 0.0

    return scores, default_threshold


def make_config_grid(models):
    configs = []

    if "logistic_regression" in models:
        for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
            configs.append(
                {
                    "model_name": "logistic_regression",
                    "C": C,
                    "gamma": "none",
                    "n_estimators": 0,
                    "max_depth": "none",
                    "min_samples_leaf": 0,
                }
            )

    if "linear_svm" in models:
        for C in [0.01, 0.1, 1.0, 10.0]:
            configs.append(
                {
                    "model_name": "linear_svm",
                    "C": C,
                    "gamma": "none",
                    "n_estimators": 0,
                    "max_depth": "none",
                    "min_samples_leaf": 0,
                }
            )

    if "rbf_svm" in models:
        for C in [0.1, 1.0, 10.0]:
            for gamma in ["scale", 0.01, 0.1]:
                configs.append(
                    {
                        "model_name": "rbf_svm",
                        "C": C,
                        "gamma": gamma,
                        "n_estimators": 0,
                        "max_depth": "none",
                        "min_samples_leaf": 0,
                    }
                )

    if "random_forest" in models:
        for max_depth in [2, 3, 5, None]:
            for min_samples_leaf in [1, 3, 5, 10]:
                configs.append(
                    {
                        "model_name": "random_forest",
                        "C": 0.0,
                        "gamma": "none",
                        "n_estimators": 300,
                        "max_depth": max_depth,
                        "min_samples_leaf": min_samples_leaf,
                    }
                )

    return configs


def get_evaluable_target_genes(genes, y):
    target_genes = []

    for gene in sorted(set(genes)):
        labels = y[genes == gene]

        if len(np.unique(labels)) == 2:
            target_genes.append(gene)

    return target_genes


def inner_select_fusion_model(
    X_norm_by_feature,
    y,
    genes,
    outer_target_gene,
    k_values,
    configs,
):
    outer_train_genes = sorted([g for g in set(genes) if g != outer_target_gene])

    inner_val_genes = []

    for g in outer_train_genes:
        labels = y[genes == g]
        if len(np.unique(labels)) == 2:
            inner_val_genes.append(g)

    summary_rows = []
    inner_prediction_rows = []

    for config_id, config in enumerate(configs):
        pooled_y = []
        pooled_scores = []

        fold_rows = []

        for val_gene in inner_val_genes:
            inner_train_genes = [g for g in outer_train_genes if g != val_gene]

            X_train_df, y_train, _, _ = make_knn_score_features(
                X_norm_by_feature=X_norm_by_feature,
                y=y,
                genes=genes,
                query_genes=inner_train_genes,
                candidate_gene_pool=inner_train_genes,
                k_values=k_values,
            )

            X_val_df, y_val, _, val_indices = make_knn_score_features(
                X_norm_by_feature=X_norm_by_feature,
                y=y,
                genes=genes,
                query_genes=[val_gene],
                candidate_gene_pool=inner_train_genes,
                k_values=k_values,
            )

            model = make_model(config)
            model.fit(X_train_df, y_train)

            scores, default_threshold = predict_model_scores(model, X_val_df)

            default_metrics = evaluate_from_scores(y_val, scores, default_threshold)

            pooled_y.append(y_val)
            pooled_scores.append(scores)

            fold_row = {
                "config_id": config_id,
                "outer_target_gene": outer_target_gene,
                "inner_validation_gene": val_gene,
                "default_threshold": default_threshold,
                **config,
                "default_accuracy": default_metrics["accuracy"],
                "default_balanced_accuracy": default_metrics["balanced_accuracy"],
                "default_roc_auc": default_metrics["roc_auc"],
                "default_gof_f1": default_metrics["gof_f1"],
            }

            fold_rows.append(fold_row)

            for idx, true_label, score in zip(val_indices, y_val, scores):
                inner_prediction_rows.append(
                    {
                        "config_id": config_id,
                        "outer_target_gene": outer_target_gene,
                        "inner_validation_gene": val_gene,
                        "variant_index": int(idx),
                        "true_label": int(true_label),
                        "score": float(score),
                        **config,
                    }
                )

        if len(pooled_y) == 0:
            continue

        pooled_y = np.concatenate(pooled_y)
        pooled_scores = np.concatenate(pooled_scores)

        threshold_info = choose_best_threshold(pooled_y, pooled_scores)
        tuned_metrics = evaluate_from_scores(
            pooled_y,
            pooled_scores,
            threshold_info["threshold"],
        )

        fold_df = pd.DataFrame(fold_rows)

        summary_row = {
            "config_id": config_id,
            "outer_target_gene": outer_target_gene,
            **config,
            "n_inner_validation_genes": len(inner_val_genes),
            "threshold": threshold_info["threshold"],
            "inner_tuned_accuracy": tuned_metrics["accuracy"],
            "inner_tuned_balanced_accuracy": tuned_metrics["balanced_accuracy"],
            "inner_tuned_roc_auc": tuned_metrics["roc_auc"],
            "inner_tuned_gof_f1": tuned_metrics["gof_f1"],
            "inner_default_accuracy_mean": fold_df["default_accuracy"].mean(),
            "inner_default_balanced_accuracy_mean": fold_df["default_balanced_accuracy"].mean(),
            "inner_default_roc_auc_mean": fold_df["default_roc_auc"].mean(),
            "inner_default_gof_f1_mean": fold_df["default_gof_f1"].mean(),
        }

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df.sort_values(
        ["inner_tuned_balanced_accuracy", "inner_tuned_roc_auc", "inner_tuned_gof_f1"],
        ascending=False,
    )

    best_config = summary_df.iloc[0].to_dict()
    inner_prediction_df = pd.DataFrame(inner_prediction_rows)

    return best_config, summary_df, inner_prediction_df


def final_train_and_test_fusion(
    X_norm_by_feature,
    y,
    genes,
    target_gene,
    k_values,
    best_config,
):
    outer_train_genes = sorted([g for g in set(genes) if g != target_gene])

    X_train_df, y_train, _, train_indices = make_knn_score_features(
        X_norm_by_feature=X_norm_by_feature,
        y=y,
        genes=genes,
        query_genes=outer_train_genes,
        candidate_gene_pool=outer_train_genes,
        k_values=k_values,
    )

    X_test_df, y_test, _, test_indices = make_knn_score_features(
        X_norm_by_feature=X_norm_by_feature,
        y=y,
        genes=genes,
        query_genes=[target_gene],
        candidate_gene_pool=outer_train_genes,
        k_values=k_values,
    )

    config = {
        "model_name": best_config["model_name"],
        "C": best_config["C"],
        "gamma": best_config["gamma"],
        "n_estimators": best_config["n_estimators"],
        "max_depth": None if best_config["max_depth"] == "none" else best_config["max_depth"],
        "min_samples_leaf": best_config["min_samples_leaf"],
    }

    model = make_model(config)
    model.fit(X_train_df, y_train)

    scores, default_threshold = predict_model_scores(model, X_test_df)

    tuned_threshold = float(best_config["threshold"])

    tuned_metrics = evaluate_from_scores(y_test, scores, tuned_threshold)
    default_metrics = evaluate_from_scores(y_test, scores, default_threshold)

    tuned_row = {
        "target_gene": target_gene,
        "method": "esm_knn_score_fusion_threshold_tuned",
        "threshold": tuned_threshold,
        **config,
        **tuned_metrics,
    }

    default_row = {
        "target_gene": target_gene,
        "method": "esm_knn_score_fusion_default_threshold",
        "threshold": default_threshold,
        **config,
        **default_metrics,
    }

    predictions_df = pd.DataFrame(
        {
            "target_gene": target_gene,
            "variant_index": test_indices,
            "true_label": y_test,
            "fusion_score": scores,
            "tuned_pred": (scores >= tuned_threshold).astype(int),
            "default_pred": (scores >= default_threshold).astype(int),
        }
    )

    return tuned_row, default_row, predictions_df


def evaluate_frozen_knn_all_features(
    X_norm_by_feature,
    y,
    genes,
    target_gene,
    k_values,
):
    query_idx = np.where(genes == target_gene)[0]
    candidate_idx = np.where(genes != target_gene)[0]

    rows = []

    for feature_set, Xn in X_norm_by_feature.items():
        Xq = Xn[query_idx]
        Xc = Xn[candidate_idx]
        yc = y[candidate_idx]
        yq = y[query_idx]

        sim = Xq @ Xc.T
        order = np.argsort(-sim, axis=1)

        for k in k_values:
            k_eff = min(k, sim.shape[1])
            topk = order[:, :k_eff]
            scores = yc[topk].mean(axis=1)

            metrics = evaluate_from_scores(yq, scores, threshold=0.5)

            row = {
                "target_gene": target_gene,
                "method": "frozen_esm_knn",
                "feature_set": feature_set,
                "k": k,
                **metrics,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature_sets",
        nargs="+",
        default=["wt_site", "mutant_site", "delta_site", "wt_mut_site"],
        choices=list(FEATURE_FILES.keys()),
    )

    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "linear_svm", "rbf_svm", "random_forest"],
        choices=["logistic_regression", "linear_svm", "rbf_svm", "random_forest"],
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

    print("\nLoading and normalizing ESM feature matrices:")
    X_norm_by_feature = {}

    for feature_set in args.feature_sets:
        path = FEATURE_FILES[feature_set]
        X = np.load(path)
        X_norm_by_feature[feature_set] = l2_normalize(X)
        print(f"{feature_set}: {X.shape}")

    if args.target_genes == ["auto"]:
        target_genes = get_evaluable_target_genes(genes, y)
    else:
        target_genes = args.target_genes

    print("\nTarget genes:")
    print(target_genes)

    label_table = (
        pd.DataFrame({"Gene": genes, "Label": y})
        .assign(Mechanism=lambda d: d["Label"].map({0: "LOF", 1: "GOF"}))
        .pivot_table(index="Gene", columns="Mechanism", values="Label", aggfunc="count", fill_value=0)
        .reset_index()
    )

    print("\nLabel counts by gene:")
    print(label_table.to_string(index=False))

    configs = make_config_grid(args.models)
    print(f"\nNumber of fusion model configs: {len(configs)}")

    tuned_rows = []
    default_rows = []
    frozen_rows = []
    all_inner_summaries = []
    all_inner_predictions = []
    all_test_predictions = []

    for target_gene in target_genes:
        print("\n" + "=" * 90)
        print(f"OUTER HELD-OUT TARGET GENE: {target_gene}")
        print("=" * 90)

        best_config, inner_summary, inner_predictions = inner_select_fusion_model(
            X_norm_by_feature=X_norm_by_feature,
            y=y,
            genes=genes,
            outer_target_gene=target_gene,
            k_values=args.k_values,
            configs=configs,
        )

        all_inner_summaries.append(inner_summary)
        all_inner_predictions.append(inner_predictions)

        print("\nBest fusion config from inner validation:")
        print(json.dumps(best_config, indent=2, default=str))

        tuned_row, default_row, test_predictions = final_train_and_test_fusion(
            X_norm_by_feature=X_norm_by_feature,
            y=y,
            genes=genes,
            target_gene=target_gene,
            k_values=args.k_values,
            best_config=best_config,
        )

        tuned_rows.append(tuned_row)
        default_rows.append(default_row)
        all_test_predictions.append(test_predictions)

        print("\nFinal held-out threshold-tuned fusion result:")
        print(pd.DataFrame([tuned_row]).to_string(index=False))

        print("\nFinal held-out default-threshold fusion result:")
        print(pd.DataFrame([default_row]).to_string(index=False))

        frozen_df = evaluate_frozen_knn_all_features(
            X_norm_by_feature=X_norm_by_feature,
            y=y,
            genes=genes,
            target_gene=target_gene,
            k_values=args.k_values,
        )

        frozen_rows.append(frozen_df)

    tuned_df = pd.DataFrame(tuned_rows)
    default_df = pd.DataFrame(default_rows)
    frozen_df = pd.concat(frozen_rows, ignore_index=True)
    inner_summary_df = pd.concat(all_inner_summaries, ignore_index=True)
    inner_predictions_df = pd.concat(all_inner_predictions, ignore_index=True)
    test_predictions_df = pd.concat(all_test_predictions, ignore_index=True)

    tuned_df.to_csv(OUT_DIR / "fusion_threshold_tuned_final_logo_results.csv", index=False)
    default_df.to_csv(OUT_DIR / "fusion_default_threshold_final_logo_results.csv", index=False)
    frozen_df.to_csv(OUT_DIR / "frozen_esm_knn_logo_results.csv", index=False)
    inner_summary_df.to_csv(OUT_DIR / "inner_fusion_summary_by_config.csv", index=False)
    inner_predictions_df.to_csv(OUT_DIR / "inner_fusion_prediction_scores.csv", index=False)
    test_predictions_df.to_csv(OUT_DIR / "heldout_fusion_prediction_scores.csv", index=False)

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
                "fusion_model": tuned_row["model_name"],
                "fusion_threshold": tuned_row["threshold"],
                "fusion_tuned_balanced_accuracy": tuned_row["balanced_accuracy"],
                "fusion_default_balanced_accuracy": default_row["balanced_accuracy"],
                "best_frozen_feature_set": frozen_row["feature_set"],
                "best_frozen_k": frozen_row["k"],
                "best_frozen_balanced_accuracy": frozen_row["balanced_accuracy"],
                "delta_tuned_vs_default_balanced_accuracy": tuned_row["balanced_accuracy"] - default_row["balanced_accuracy"],
                "delta_tuned_vs_best_frozen_balanced_accuracy": tuned_row["balanced_accuracy"] - frozen_row["balanced_accuracy"],
                "fusion_tuned_roc_auc": tuned_row["roc_auc"],
                "best_frozen_roc_auc": frozen_row["roc_auc"],
                "delta_roc_auc": tuned_row["roc_auc"] - frozen_row["roc_auc"],
                "fusion_tuned_gof_f1": tuned_row["gof_f1"],
                "best_frozen_gof_f1": frozen_row["gof_f1"],
                "delta_gof_f1": tuned_row["gof_f1"] - frozen_row["gof_f1"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUT_DIR / "fusion_vs_default_vs_best_frozen_by_target_gene.csv", index=False)

    overall = pd.DataFrame(
        [
            {
                "method": "fusion_threshold_tuned",
                "n_target_genes": tuned_df["target_gene"].nunique(),
                "mean_balanced_accuracy": tuned_df["balanced_accuracy"].mean(),
                "mean_roc_auc": tuned_df["roc_auc"].mean(),
                "mean_gof_f1": tuned_df["gof_f1"].mean(),
                "mean_accuracy": tuned_df["accuracy"].mean(),
            },
            {
                "method": "fusion_default_threshold",
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

    overall.to_csv(OUT_DIR / "overall_fusion_vs_default_vs_frozen_summary.csv", index=False)

    print("\n" + "=" * 90)
    print("FUSION VS DEFAULT VS BEST FROZEN BY TARGET GENE")
    print("=" * 90)

    cols = [
        "target_gene",
        "fusion_model",
        "fusion_tuned_balanced_accuracy",
        "fusion_default_balanced_accuracy",
        "best_frozen_feature_set",
        "best_frozen_k",
        "best_frozen_balanced_accuracy",
        "delta_tuned_vs_best_frozen_balanced_accuracy",
        "fusion_tuned_roc_auc",
        "best_frozen_roc_auc",
        "delta_roc_auc",
        "fusion_tuned_gof_f1",
        "best_frozen_gof_f1",
        "delta_gof_f1",
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