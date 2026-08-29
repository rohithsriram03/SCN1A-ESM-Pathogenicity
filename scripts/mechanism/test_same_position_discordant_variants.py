from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"

FEATURE_FILES = {
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}

META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "same_position_discordant_test"


def get_labels(meta):
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
            return roc_auc_score(y_true, scores)
    except ValueError:
        pass
    return np.nan


def make_exact_position_group(meta):
    return (
        meta["Gene"].astype(str)
        + "|pos="
        + meta["AA_Position"].astype(str)
        + "|ref="
        + meta["Ref_AA"].astype(str)
    )


def make_cid_group(meta):
    if "Family_Alignment_CID" not in meta.columns:
        raise ValueError("Family_Alignment_CID column not found.")

    return "CID=" + meta["Family_Alignment_CID"].astype(str)


def identify_discordant_groups(meta, labels, group_ids, group_type):
    rows = []

    for group_id, idx in pd.Series(np.arange(len(meta))).groupby(group_ids):
        idx = idx.to_numpy()
        group_labels = labels[idx]

        n_lof = int((group_labels == 0).sum())
        n_gof = int((group_labels == 1).sum())

        if n_lof > 0 and n_gof > 0:
            genes = ",".join(sorted(meta.iloc[idx]["Gene"].astype(str).unique()))
            positions = ",".join(sorted(meta.iloc[idx]["AA_Position"].astype(str).unique()))

            rows.append(
                {
                    "group_type": group_type,
                    "group_id": group_id,
                    "n_variants": len(idx),
                    "n_lof": n_lof,
                    "n_gof": n_gof,
                    "genes": genes,
                    "positions": positions,
                    "variant_indices": ";".join(map(str, idx)),
                }
            )

    return pd.DataFrame(rows)


def compute_knn_scores_for_queries(X, labels, query_indices, candidate_indices, k):
    Xq = X[query_indices]
    Xc = X[candidate_indices]
    yc = labels[candidate_indices]

    sim = Xq @ Xc.T
    order = np.argsort(-sim, axis=1)

    k_eff = min(k, sim.shape[1])
    topk = order[:, :k_eff]

    topk_labels = yc[topk]
    topk_sims = np.take_along_axis(sim, topk, axis=1)

    gof_scores = topk_labels.mean(axis=1)
    mean_sims = topk_sims.mean(axis=1)

    return gof_scores, mean_sims


def run_discordant_pair_test(meta, labels, X_by_feature, group_df, group_ids, group_type, k_values):
    query_rows = []
    pair_rows = []

    for _, group in group_df.iterrows():
        group_id = group["group_id"]
        idx = np.array([int(x) for x in group["variant_indices"].split(";")])

        lof_idx = idx[labels[idx] == 0]
        gof_idx = idx[labels[idx] == 1]

        # Exclude variants from the same controlled group from the reference pool.
        candidate_indices = np.where(group_ids != group_id)[0]

        if len(candidate_indices) == 0:
            continue

        for feature_set, X in X_by_feature.items():
            for k in k_values:
                scores, mean_sims = compute_knn_scores_for_queries(
                    X=X,
                    labels=labels,
                    query_indices=idx,
                    candidate_indices=candidate_indices,
                    k=k,
                )

                score_map = dict(zip(idx, scores))
                sim_map = dict(zip(idx, mean_sims))

                # Query-level results.
                preds = (scores >= 0.5).astype(int)

                for variant_index, score, pred, mean_sim in zip(idx, scores, preds, mean_sims):
                    row_meta = meta.iloc[variant_index]

                    query_rows.append(
                        {
                            "group_type": group_type,
                            "group_id": group_id,
                            "feature_set": feature_set,
                            "k": k,
                            "variant_index": int(variant_index),
                            "Gene": row_meta["Gene"],
                            "Protein_Change": row_meta.get("Protein_Change", ""),
                            "AA_Position": row_meta["AA_Position"],
                            "Ref_AA": row_meta["Ref_AA"],
                            "Alt_AA": row_meta["Alt_AA"],
                            "Mechanism_Label": row_meta["Mechanism_Label"],
                            "true_label": int(labels[variant_index]),
                            "gof_score": float(score),
                            "pred_label": int(pred),
                            "mean_topk_similarity": float(mean_sim),
                        }
                    )

                # Pair-level ranking results.
                for li in lof_idx:
                    for gi in gof_idx:
                        lof_score = score_map[li]
                        gof_score = score_map[gi]

                        if gof_score > lof_score:
                            pair_correct = 1.0
                        elif gof_score == lof_score:
                            pair_correct = 0.5
                        else:
                            pair_correct = 0.0

                        pair_rows.append(
                            {
                                "group_type": group_type,
                                "group_id": group_id,
                                "feature_set": feature_set,
                                "k": k,
                                "lof_variant_index": int(li),
                                "gof_variant_index": int(gi),
                                "lof_variant": meta.iloc[li].get("Protein_Change", ""),
                                "gof_variant": meta.iloc[gi].get("Protein_Change", ""),
                                "lof_gene": meta.iloc[li]["Gene"],
                                "gof_gene": meta.iloc[gi]["Gene"],
                                "lof_position": meta.iloc[li]["AA_Position"],
                                "gof_position": meta.iloc[gi]["AA_Position"],
                                "lof_alt": meta.iloc[li]["Alt_AA"],
                                "gof_alt": meta.iloc[gi]["Alt_AA"],
                                "lof_gof_score": float(lof_score),
                                "gof_gof_score": float(gof_score),
                                "score_difference_gof_minus_lof": float(gof_score - lof_score),
                                "pairwise_correct": pair_correct,
                            }
                        )

    return pd.DataFrame(query_rows), pd.DataFrame(pair_rows)


def summarize_results(query_df, pair_df):
    query_summary_rows = []

    for (group_type, feature_set, k), sub in query_df.groupby(["group_type", "feature_set", "k"]):
        y = sub["true_label"].to_numpy()
        scores = sub["gof_score"].to_numpy()
        preds = sub["pred_label"].to_numpy()

        query_summary_rows.append(
            {
                "group_type": group_type,
                "feature_set": feature_set,
                "k": k,
                "n_query_variants": len(sub),
                "n_groups": sub["group_id"].nunique(),
                "accuracy": accuracy_score(y, preds),
                "balanced_accuracy": balanced_accuracy_score(y, preds),
                "roc_auc": safe_auc(y, scores),
                "gof_f1": f1_score(y, preds, pos_label=1, zero_division=0),
            }
        )

    query_summary = pd.DataFrame(query_summary_rows)

    pair_summary = (
        pair_df.groupby(["group_type", "feature_set", "k"])
        .agg(
            n_pairs=("pairwise_correct", "size"),
            n_groups=("group_id", "nunique"),
            mean_pairwise_correct=("pairwise_correct", "mean"),
            mean_score_difference_gof_minus_lof=("score_difference_gof_minus_lof", "mean"),
            median_score_difference_gof_minus_lof=("score_difference_gof_minus_lof", "median"),
        )
        .reset_index()
        .sort_values(["group_type", "mean_pairwise_correct"], ascending=[True, False])
    )

    return query_summary, pair_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_FILE)
    labels = get_labels(meta)

    X_by_feature = {}

    for feature_set, path in FEATURE_FILES.items():
        X_by_feature[feature_set] = l2_normalize(np.load(path))

    exact_group_ids = make_exact_position_group(meta)
    cid_group_ids = make_cid_group(meta)

    exact_groups = identify_discordant_groups(
        meta=meta,
        labels=labels,
        group_ids=exact_group_ids,
        group_type="exact_gene_position_ref",
    )

    cid_groups = identify_discordant_groups(
        meta=meta,
        labels=labels,
        group_ids=cid_group_ids,
        group_type="family_alignment_cid",
    )

    exact_groups.to_csv(OUT_DIR / "exact_position_discordant_groups.csv", index=False)
    cid_groups.to_csv(OUT_DIR / "family_cid_discordant_groups.csv", index=False)

    print("\nExact same-gene/same-position discordant groups:")
    print(exact_groups.to_string(index=False) if len(exact_groups) else "None found.")

    print("\nFamily-alignment CID discordant groups:")
    print(cid_groups.to_string(index=False) if len(cid_groups) else "None found.")

    all_query_results = []
    all_pair_results = []

    if len(exact_groups):
        q, p = run_discordant_pair_test(
            meta=meta,
            labels=labels,
            X_by_feature=X_by_feature,
            group_df=exact_groups,
            group_ids=exact_group_ids.to_numpy(),
            group_type="exact_gene_position_ref",
            k_values=args.k_values,
        )
        all_query_results.append(q)
        all_pair_results.append(p)

    if len(cid_groups):
        q, p = run_discordant_pair_test(
            meta=meta,
            labels=labels,
            X_by_feature=X_by_feature,
            group_df=cid_groups,
            group_ids=cid_group_ids.to_numpy(),
            group_type="family_alignment_cid",
            k_values=args.k_values,
        )
        all_query_results.append(q)
        all_pair_results.append(p)

    if not all_query_results:
        print("\nNo discordant same-position or same-CID groups found.")
        return

    query_df = pd.concat(all_query_results, ignore_index=True)
    pair_df = pd.concat(all_pair_results, ignore_index=True)

    query_summary, pair_summary = summarize_results(query_df, pair_df)

    query_df.to_csv(OUT_DIR / "discordant_query_level_knn_scores.csv", index=False)
    pair_df.to_csv(OUT_DIR / "discordant_pairwise_ranking_results.csv", index=False)
    query_summary.to_csv(OUT_DIR / "discordant_query_level_summary.csv", index=False)
    pair_summary.to_csv(OUT_DIR / "discordant_pairwise_summary.csv", index=False)

    print("\nQuery-level summary:")
    print(query_summary.to_string(index=False))

    print("\nPairwise ranking summary:")
    print(pair_summary.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()