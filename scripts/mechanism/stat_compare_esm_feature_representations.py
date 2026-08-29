from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "esm_feature_statistical_comparisons"

META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

BASE_FEATURE_FILES = {
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}


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
    except Exception:
        return np.nan


def safe_ap(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return average_precision_score(y_true, scores)
    except Exception:
        return np.nan


def compute_metrics(y_true, scores, preds):
    return {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "roc_auc": safe_auc(y_true, scores),
        "average_precision": safe_ap(y_true, scores),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y_true, preds, pos_label=1, zero_division=0),
    }


def build_feature_matrices():
    wt = np.load(BASE_FEATURE_FILES["wt_site"])
    mut = np.load(BASE_FEATURE_FILES["mutant_site"])
    delta = np.load(BASE_FEATURE_FILES["delta_site"])
    wt_mut = np.load(BASE_FEATURE_FILES["wt_mut_site"])

    abs_delta = np.abs(delta)

    features = {
        "wt_site": wt,
        "mutant_site": mut,
        "delta_site": delta,
        "abs_delta_site": abs_delta,
        "wt_mut_site": wt_mut,

        # Extra combined setups to test whether mutation-specific info helps.
        "wt_delta": np.concatenate([wt, delta], axis=1),
        "mutant_delta": np.concatenate([mut, delta], axis=1),
        "wt_mut_delta": np.concatenate([wt, mut, delta], axis=1),
        "wt_mut_delta_absdelta": np.concatenate([wt, mut, delta, abs_delta], axis=1),
    }

    return {name: l2_normalize(x) for name, x in features.items()}


def cross_gene_knn_scores(X, y, genes, query_indices, k_values, fixed_candidate_gene=None):
    """
    For each query variant, use only cross-gene candidates.

    If fixed_candidate_gene is None:
        each query excludes its own gene.
    If fixed_candidate_gene is given, e.g. "SCN1A":
        candidates exclude that held-out gene.
    """
    rows = []

    for q_idx in query_indices:
        query_gene = genes[q_idx]

        if fixed_candidate_gene is None:
            candidate_indices = np.where(genes != query_gene)[0]
        else:
            candidate_indices = np.where(genes != fixed_candidate_gene)[0]

        sim = X[q_idx] @ X[candidate_indices].T
        order = np.argsort(-sim)

        for k in k_values:
            k_eff = min(k, len(candidate_indices))
            topk = order[:k_eff]
            topk_candidate_indices = candidate_indices[topk]
            topk_labels = y[topk_candidate_indices]

            gof_score = topk_labels.mean()
            pred = int(gof_score >= 0.5)

            rows.append(
                {
                    "variant_index": int(q_idx),
                    "query_gene": query_gene,
                    "k": k,
                    "true_label": int(y[q_idx]),
                    "gof_score": float(gof_score),
                    "pred_label": int(pred),
                    "correct": int(pred == y[q_idx]),
                }
            )

    return pd.DataFrame(rows)


def bootstrap_metric_difference(y, scores_a, preds_a, scores_b, preds_b, metric_name, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)

    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        ya = y[idx]
        sa = scores_a[idx]
        pa = preds_a[idx]
        sb = scores_b[idx]
        pb = preds_b[idx]

        ma = compute_metrics(ya, sa, pa)[metric_name]
        mb = compute_metrics(ya, sb, pb)[metric_name]

        if not np.isnan(ma) and not np.isnan(mb):
            diffs.append(ma - mb)

    diffs = np.array(diffs)

    if len(diffs) == 0:
        return np.nan, np.nan

    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def paired_permutation_metric_test(y, scores_a, preds_a, scores_b, preds_b, metric_name, n_perm=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)

    obs_a = compute_metrics(y, scores_a, preds_a)[metric_name]
    obs_b = compute_metrics(y, scores_b, preds_b)[metric_name]
    obs_diff = obs_a - obs_b

    perm_diffs = []

    for _ in range(n_perm):
        swap = rng.random(n) < 0.5

        perm_scores_a = scores_a.copy()
        perm_scores_b = scores_b.copy()
        perm_preds_a = preds_a.copy()
        perm_preds_b = preds_b.copy()

        perm_scores_a[swap], perm_scores_b[swap] = scores_b[swap], scores_a[swap]
        perm_preds_a[swap], perm_preds_b[swap] = preds_b[swap], preds_a[swap]

        ma = compute_metrics(y, perm_scores_a, perm_preds_a)[metric_name]
        mb = compute_metrics(y, perm_scores_b, perm_preds_b)[metric_name]

        if not np.isnan(ma) and not np.isnan(mb):
            perm_diffs.append(ma - mb)

    perm_diffs = np.array(perm_diffs)

    if len(perm_diffs) == 0 or np.isnan(obs_diff):
        return obs_diff, np.nan, np.nan

    p_two_sided = (np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (len(perm_diffs) + 1)

    if obs_diff >= 0:
        p_one_sided = (np.sum(perm_diffs >= obs_diff) + 1) / (len(perm_diffs) + 1)
    else:
        p_one_sided = (np.sum(perm_diffs <= obs_diff) + 1) / (len(perm_diffs) + 1)

    return obs_diff, p_two_sided, p_one_sided


def compare_feature_pair(predictions, feature_a, feature_b, setting_name, metrics_to_compare, n_boot, n_perm, seed):
    rows = []

    df_a = predictions[predictions["feature_set"] == feature_a].copy()
    df_b = predictions[predictions["feature_set"] == feature_b].copy()

    merge_cols = ["variant_index", "query_gene", "k", "true_label"]

    merged = df_a.merge(
        df_b,
        on=merge_cols,
        suffixes=("_a", "_b"),
    )

    for k, sub in merged.groupby("k"):
        y = sub["true_label"].to_numpy()

        scores_a = sub["gof_score_a"].to_numpy()
        preds_a = sub["pred_label_a"].to_numpy()

        scores_b = sub["gof_score_b"].to_numpy()
        preds_b = sub["pred_label_b"].to_numpy()

        metrics_a = compute_metrics(y, scores_a, preds_a)
        metrics_b = compute_metrics(y, scores_b, preds_b)

        for metric_name in metrics_to_compare:
            ci_low, ci_high = bootstrap_metric_difference(
                y=y,
                scores_a=scores_a,
                preds_a=preds_a,
                scores_b=scores_b,
                preds_b=preds_b,
                metric_name=metric_name,
                n_boot=n_boot,
                seed=seed,
            )

            obs_diff, p_two, p_one = paired_permutation_metric_test(
                y=y,
                scores_a=scores_a,
                preds_a=preds_a,
                scores_b=scores_b,
                preds_b=preds_b,
                metric_name=metric_name,
                n_perm=n_perm,
                seed=seed,
            )

            rows.append(
                {
                    "setting": setting_name,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "k": int(k),
                    "metric": metric_name,
                    "n_queries": int(len(sub)),
                    "value_a": metrics_a[metric_name],
                    "value_b": metrics_b[metric_name],
                    "diff_a_minus_b": obs_diff,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "paired_permutation_p_two_sided": p_two,
                    "paired_permutation_p_one_sided": p_one,
                }
            )

    return pd.DataFrame(rows)


def summarize_predictions(predictions, setting_name):
    rows = []

    for (feature_set, k), sub in predictions.groupby(["feature_set", "k"]):
        y = sub["true_label"].to_numpy()
        scores = sub["gof_score"].to_numpy()
        preds = sub["pred_label"].to_numpy()

        m = compute_metrics(y, scores, preds)

        rows.append(
            {
                "setting": setting_name,
                "feature_set": feature_set,
                "k": int(k),
                "n_queries": int(len(sub)),
                **m,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["setting", "balanced_accuracy", "roc_auc"],
        ascending=[True, False, False],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--heldout_gene", type=str, default="SCN1A")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_FILE)
    y = get_labels(meta)
    genes = meta["Gene"].astype(str).to_numpy()

    features = build_feature_matrices()

    print("Loaded features:")
    for name, x in features.items():
        print(f"{name}: {x.shape}")

    print("\nLabel counts:")
    print(pd.Series(y).map({0: "LOF", 1: "GOF"}).value_counts().to_string())

    all_prediction_frames = []
    all_metric_summaries = []
    all_comparisons = []

    # Setting 1: global all 348 variants as queries.
    global_query_indices = np.arange(len(meta))

    for feature_set, X in features.items():
        pred = cross_gene_knn_scores(
            X=X,
            y=y,
            genes=genes,
            query_indices=global_query_indices,
            k_values=args.k_values,
            fixed_candidate_gene=None,
        )
        pred["setting"] = "global_all_348_cross_gene"
        pred["feature_set"] = feature_set
        all_prediction_frames.append(pred)

    global_predictions = pd.concat(
        [p for p in all_prediction_frames if p["setting"].iloc[0] == "global_all_348_cross_gene"],
        ignore_index=True,
    )

    global_summary = summarize_predictions(global_predictions, "global_all_348_cross_gene")
    all_metric_summaries.append(global_summary)

    # Setting 2: held-out SCN1A only.
    heldout_query_indices = np.where(genes == args.heldout_gene)[0]

    heldout_frames = []

    for feature_set, X in features.items():
        pred = cross_gene_knn_scores(
            X=X,
            y=y,
            genes=genes,
            query_indices=heldout_query_indices,
            k_values=args.k_values,
            fixed_candidate_gene=args.heldout_gene,
        )
        pred["setting"] = f"heldout_{args.heldout_gene}"
        pred["feature_set"] = feature_set
        heldout_frames.append(pred)
        all_prediction_frames.append(pred)

    heldout_predictions = pd.concat(heldout_frames, ignore_index=True)

    heldout_summary = summarize_predictions(heldout_predictions, f"heldout_{args.heldout_gene}")
    all_metric_summaries.append(heldout_summary)

    metrics_to_compare = [
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "gof_f1",
    ]

    comparison_features = [
        "wt_site",
        "mutant_site",
        "delta_site",
        "abs_delta_site",
        "wt_delta",
        "mutant_delta",
        "wt_mut_delta",
        "wt_mut_delta_absdelta",
    ]

    # Compare WT+mutant against each other feature.
    for setting_name, predictions in [
        ("global_all_348_cross_gene", global_predictions),
        (f"heldout_{args.heldout_gene}", heldout_predictions),
    ]:
        for feature_b in comparison_features:
            comp = compare_feature_pair(
                predictions=predictions,
                feature_a="wt_mut_site",
                feature_b=feature_b,
                setting_name=setting_name,
                metrics_to_compare=metrics_to_compare,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=args.seed,
            )
            all_comparisons.append(comp)

    predictions_all = pd.concat(all_prediction_frames, ignore_index=True)
    metric_summary = pd.concat(all_metric_summaries, ignore_index=True)
    comparison_df = pd.concat(all_comparisons, ignore_index=True)

    predictions_all.to_csv(OUT_DIR / "all_feature_knn_predictions.csv", index=False)
    metric_summary.to_csv(OUT_DIR / "feature_metric_summary.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "paired_feature_comparisons.csv", index=False)

    # Key rows: WT+mutant vs WT site.
    key = comparison_df[
        (comparison_df["feature_a"] == "wt_mut_site")
        & (comparison_df["feature_b"] == "wt_site")
        & (comparison_df["metric"].isin(["accuracy", "balanced_accuracy", "roc_auc"]))
    ].copy()

    key.to_csv(OUT_DIR / "key_wtmut_vs_wt_comparison.csv", index=False)

    print("\n" + "=" * 90)
    print("FEATURE METRIC SUMMARY")
    print("=" * 90)
    print(
        metric_summary[
            [
                "setting",
                "feature_set",
                "k",
                "n_queries",
                "accuracy",
                "balanced_accuracy",
                "roc_auc",
                "gof_f1",
            ]
        ]
        .sort_values(["setting", "balanced_accuracy", "roc_auc"], ascending=[True, False, False])
        .to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("KEY COMPARISON: WT+MUTANT VS WT SITE")
    print("=" * 90)
    print(
        key[
            [
                "setting",
                "k",
                "metric",
                "value_a",
                "value_b",
                "diff_a_minus_b",
                "bootstrap_95ci_low",
                "bootstrap_95ci_high",
                "paired_permutation_p_two_sided",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 90)
    print("ALL WT+MUTANT PAIRWISE COMPARISONS SAVED TO:")
    print(OUT_DIR / "paired_feature_comparisons.csv")
    print("=" * 90)


if __name__ == "__main__":
    main()