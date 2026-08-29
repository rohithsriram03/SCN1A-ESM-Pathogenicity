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


def find_project_root():
    start = Path(__file__).resolve()
    for p in [start] + list(start.parents):
        target = p / "results" / "mechanism" / "scion_esm2_650M_embeddings" / "scion_embedding_metadata.csv"
        if target.exists():
            return p
    raise FileNotFoundError("Could not locate SCN1A_ESM_Project root from this script path.")


PROJECT_ROOT = find_project_root()
EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "fast_feature_stat_comparisons"

META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

FEATURE_FILES = {
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}


def get_labels(meta):
    if "Mechanism_Binary" in meta.columns:
        return meta["Mechanism_Binary"].astype(int).to_numpy()

    if "Mechanism_Label" in meta.columns:
        return meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1}).astype(int).to_numpy()

    raise ValueError("Could not find Mechanism_Binary or Mechanism_Label.")


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)


def safe_auc(y, scores):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return roc_auc_score(y, scores)
    except Exception:
        return np.nan


def safe_ap(y, scores):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return average_precision_score(y, scores)
    except Exception:
        return np.nan


def metric_value(metric, y, scores, preds):
    if metric == "accuracy":
        return accuracy_score(y, preds)

    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y, preds)

    if metric == "roc_auc":
        return safe_auc(y, scores)

    if metric == "gof_f1":
        return f1_score(y, preds, pos_label=1, zero_division=0)

    raise ValueError(f"Unknown metric: {metric}")


def all_metrics(y, scores, preds):
    return {
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "roc_auc": safe_auc(y, scores),
        "average_precision": safe_ap(y, scores),
        "gof_f1": f1_score(y, preds, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
    }


def knn_scores(X, y, genes, query_indices, k_values, heldout_gene=None):
    rows = []

    for q_idx in query_indices:
        query_gene = genes[q_idx]

        if heldout_gene is None:
            candidate_indices = np.where(genes != query_gene)[0]
        else:
            candidate_indices = np.where(genes != heldout_gene)[0]

        sim = X[q_idx] @ X[candidate_indices].T
        order = np.argsort(-sim)

        for k in k_values:
            k_eff = min(k, len(candidate_indices))
            topk = order[:k_eff]
            topk_labels = y[candidate_indices[topk]]

            score = float(topk_labels.mean())
            pred = int(score >= 0.5)

            rows.append(
                {
                    "variant_index": int(q_idx),
                    "query_gene": query_gene,
                    "k": int(k),
                    "true_label": int(y[q_idx]),
                    "gof_score": score,
                    "pred_label": pred,
                    "correct": int(pred == y[q_idx]),
                }
            )

    return pd.DataFrame(rows)


def paired_bootstrap_diff(y, scores_a, preds_a, scores_b, preds_b, metric, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        ma = metric_value(metric, y[idx], scores_a[idx], preds_a[idx])
        mb = metric_value(metric, y[idx], scores_b[idx], preds_b[idx])

        if not np.isnan(ma) and not np.isnan(mb):
            diffs.append(ma - mb)

    diffs = np.array(diffs)

    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def paired_permutation_diff(y, scores_a, preds_a, scores_b, preds_b, metric, n_perm, seed):
    rng = np.random.default_rng(seed)
    n = len(y)

    obs_a = metric_value(metric, y, scores_a, preds_a)
    obs_b = metric_value(metric, y, scores_b, preds_b)
    obs_diff = obs_a - obs_b

    perm_diffs = []

    for _ in range(n_perm):
        swap = rng.random(n) < 0.5

        pa_scores = scores_a.copy()
        pb_scores = scores_b.copy()
        pa_preds = preds_a.copy()
        pb_preds = preds_b.copy()

        pa_scores[swap] = scores_b[swap]
        pb_scores[swap] = scores_a[swap]
        pa_preds[swap] = preds_b[swap]
        pb_preds[swap] = preds_a[swap]

        ma = metric_value(metric, y, pa_scores, pa_preds)
        mb = metric_value(metric, y, pb_scores, pb_preds)

        if not np.isnan(ma) and not np.isnan(mb):
            perm_diffs.append(ma - mb)

    perm_diffs = np.array(perm_diffs)

    p_two = (np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (len(perm_diffs) + 1)

    return float(obs_diff), float(p_two)


def compare_pair(predictions, feature_a, feature_b, setting, n_boot, n_perm, seed):
    a = predictions[predictions["feature_set"] == feature_a].copy()
    b = predictions[predictions["feature_set"] == feature_b].copy()

    merged = a.merge(
        b,
        on=["variant_index", "query_gene", "k", "true_label"],
        suffixes=("_a", "_b"),
    )

    rows = []

    for k, sub in merged.groupby("k"):
        y = sub["true_label"].to_numpy()

        scores_a = sub["gof_score_a"].to_numpy()
        preds_a = sub["pred_label_a"].to_numpy()

        scores_b = sub["gof_score_b"].to_numpy()
        preds_b = sub["pred_label_b"].to_numpy()

        for metric in ["accuracy", "balanced_accuracy", "roc_auc", "gof_f1"]:
            ci_low, ci_high = paired_bootstrap_diff(
                y, scores_a, preds_a, scores_b, preds_b, metric, n_boot, seed
            )

            diff, p_two = paired_permutation_diff(
                y, scores_a, preds_a, scores_b, preds_b, metric, n_perm, seed
            )

            rows.append(
                {
                    "setting": setting,
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "k": int(k),
                    "metric": metric,
                    "n_queries": int(len(sub)),
                    "value_a": metric_value(metric, y, scores_a, preds_a),
                    "value_b": metric_value(metric, y, scores_b, preds_b),
                    "diff_a_minus_b": diff,
                    "bootstrap_95ci_low": ci_low,
                    "bootstrap_95ci_high": ci_high,
                    "paired_permutation_p_two_sided": p_two,
                }
            )

    return pd.DataFrame(rows)


def summarize(predictions, setting):
    rows = []

    for (feature, k), sub in predictions.groupby(["feature_set", "k"]):
        y = sub["true_label"].to_numpy()
        scores = sub["gof_score"].to_numpy()
        preds = sub["pred_label"].to_numpy()

        rows.append(
            {
                "setting": setting,
                "feature_set": feature,
                "k": int(k),
                "n_queries": int(len(sub)),
                **all_metrics(y, scores, preds),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--heldout_gene", type=str, default="SCN1A")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--n_perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_FILE)
    y = get_labels(meta)
    genes = meta["Gene"].astype(str).to_numpy()

    features = {}
    for name, path in FEATURE_FILES.items():
        features[name] = l2_normalize(np.load(path))

    all_predictions = []
    all_summaries = []
    all_comparisons = []

    settings = {
        "global_all_348_cross_gene": np.arange(len(meta)),
        f"heldout_{args.heldout_gene}": np.where(genes == args.heldout_gene)[0],
    }

    for setting, query_indices in settings.items():
        setting_preds = []

        for feature_name, X in features.items():
            heldout_gene = args.heldout_gene if setting.startswith("heldout_") else None

            pred = knn_scores(
                X=X,
                y=y,
                genes=genes,
                query_indices=query_indices,
                k_values=args.k_values,
                heldout_gene=heldout_gene,
            )

            pred["setting"] = setting
            pred["feature_set"] = feature_name
            setting_preds.append(pred)

        setting_predictions = pd.concat(setting_preds, ignore_index=True)
        all_predictions.append(setting_predictions)

        all_summaries.append(summarize(setting_predictions, setting))

        for feature_b in ["wt_site", "mutant_site", "delta_site"]:
            comp = compare_pair(
                predictions=setting_predictions,
                feature_a="wt_mut_site",
                feature_b=feature_b,
                setting=setting,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
                seed=args.seed,
            )
            all_comparisons.append(comp)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    comparison_df = pd.concat(all_comparisons, ignore_index=True)

    predictions_df.to_csv(OUT_DIR / "feature_predictions.csv", index=False)
    summary_df.to_csv(OUT_DIR / "feature_metric_summary.csv", index=False)
    comparison_df.to_csv(OUT_DIR / "paired_feature_comparisons.csv", index=False)

    key = comparison_df[
        (comparison_df["feature_a"] == "wt_mut_site")
        & (comparison_df["feature_b"] == "wt_site")
    ].copy()

    key.to_csv(OUT_DIR / "key_wtmut_vs_wt_site.csv", index=False)

    print("\n" + "=" * 90)
    print("FEATURE METRIC SUMMARY")
    print("=" * 90)
    print(
        summary_df[
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
        .sort_values(["setting", "k", "balanced_accuracy"], ascending=[True, True, False])
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
        ]
        .to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()