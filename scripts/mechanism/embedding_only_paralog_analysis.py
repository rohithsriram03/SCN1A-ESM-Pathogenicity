from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "embedding_only_paralog_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}

K_VALUES = [1, 3, 5]
N_PERMUTATIONS = 5000
RANDOM_SEED = 42


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def cosine_similarity_matrix(X_query, X_candidate):
    X_query = l2_normalize(X_query)
    X_candidate = l2_normalize(X_candidate)
    return X_query @ X_candidate.T


def safe_metrics(y_true, scores):
    y_true = np.array(y_true).astype(int)
    scores = np.array(scores).astype(float)

    preds = (scores >= 0.5).astype(int)

    out = {
        "n": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y_true, preds, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["average_precision"] = average_precision_score(y_true, scores)
    else:
        out["roc_auc"] = np.nan
        out["average_precision"] = np.nan

    return out


def compute_knn_scores_from_neighbor_lists(y_all, neighbor_lists):
    scores = []

    for neighbors in neighbor_lists:
        labels = y_all[neighbors]
        scores.append(float(np.mean(labels)))

    return np.array(scores)


def permutation_test_knn_auc(y_true, y_all, neighbor_lists, observed_auc):
    """
    Randomly permutes mechanism labels across all variants while keeping the ESM geometry fixed.
    If observed AUC is much higher than this null, it means the embedding neighborhoods
    contain real mechanism information.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    null_aucs = []

    for _ in range(N_PERMUTATIONS):
        perm_y_all = rng.permutation(y_all)
        perm_scores = compute_knn_scores_from_neighbor_lists(
            perm_y_all,
            neighbor_lists,
        )

        if len(np.unique(y_true)) < 2:
            continue

        null_aucs.append(roc_auc_score(y_true, perm_scores))

    null_aucs = np.array(null_aucs)

    p_value = (np.sum(null_aucs >= observed_auc) + 1) / (len(null_aucs) + 1)

    return {
        "permutation_auc_null_mean": float(np.mean(null_aucs)),
        "permutation_auc_null_std": float(np.std(null_aucs)),
        "permutation_p_auc_greater_than_null": float(p_value),
    }


def run_scn1a_knn_transfer(meta, X, feature_name, candidate_mode, k):
    """
    candidate_mode:
    - all_non_scn1a: each SCN1A query uses all non-SCN1A variants as candidates
    - same_cid_non_scn1a: each SCN1A query only uses non-SCN1A variants with same Family_Alignment_CID
    """
    y_all = meta["Mechanism_Binary"].astype(int).values

    query_indices = meta.index[meta["Gene"] == "SCN1A"].tolist()

    neighbor_lists = []
    query_rows = []

    for query_idx in query_indices:
        query_gene = meta.loc[query_idx, "Gene"]
        query_cid = meta.loc[query_idx, "Family_Alignment_CID"]

        if candidate_mode == "all_non_scn1a":
            candidate_indices = meta.index[meta["Gene"] != "SCN1A"].tolist()

        elif candidate_mode == "same_cid_non_scn1a":
            candidate_indices = meta.index[
                (meta["Gene"] != "SCN1A")
                & (meta["Family_Alignment_CID"] == query_cid)
            ].tolist()

        else:
            raise ValueError(f"Unknown candidate_mode: {candidate_mode}")

        if len(candidate_indices) == 0:
            continue

        query_vec = X[[query_idx]]
        candidate_vecs = X[candidate_indices]

        sims = cosine_similarity_matrix(query_vec, candidate_vecs)[0]

        order = np.argsort(-sims)
        k_eff = min(k, len(order))

        top_local = order[:k_eff]
        top_global = [candidate_indices[i] for i in top_local]

        neighbor_lists.append(np.array(top_global, dtype=int))

        top_labels = meta.loc[top_global, "Mechanism_Label"].tolist()
        top_genes = meta.loc[top_global, "Gene"].tolist()
        top_variants = meta.loc[top_global, "Variant_Key"].tolist()
        top_cids = meta.loc[top_global, "Family_Alignment_CID"].tolist()
        top_sims = sims[top_local].tolist()

        query_rows.append(
            {
                "feature_set": feature_name,
                "candidate_mode": candidate_mode,
                "k": k,
                "k_effective": k_eff,
                "query_index": query_idx,
                "query_variant": meta.loc[query_idx, "Variant_Key"],
                "query_protein_change": meta.loc[query_idx, "Protein_Change"],
                "query_gene": query_gene,
                "query_cid": query_cid,
                "true_label": meta.loc[query_idx, "Mechanism_Label"],
                "true_binary": int(meta.loc[query_idx, "Mechanism_Binary"]),
                "knn_gof_score": float(np.mean(y_all[top_global])),
                "knn_predicted_label": "GOF" if np.mean(y_all[top_global]) >= 0.5 else "LOF",
                "correct": int(
                    ("GOF" if np.mean(y_all[top_global]) >= 0.5 else "LOF")
                    == meta.loc[query_idx, "Mechanism_Label"]
                ),
                "top_neighbor_variants": ";".join(top_variants),
                "top_neighbor_genes": ";".join(top_genes),
                "top_neighbor_labels": ";".join(top_labels),
                "top_neighbor_cids": ";".join([str(x) for x in top_cids]),
                "top_neighbor_cosine_similarities": ";".join(
                    [f"{x:.6f}" for x in top_sims]
                ),
            }
        )

    pred_df = pd.DataFrame(query_rows)

    if len(pred_df) == 0:
        return None, None

    y_true = pred_df["true_binary"].astype(int).values
    scores = pred_df["knn_gof_score"].astype(float).values

    metrics = safe_metrics(y_true, scores)

    if not np.isnan(metrics["roc_auc"]):
        perm = permutation_test_knn_auc(
            y_true=y_true,
            y_all=y_all,
            neighbor_lists=neighbor_lists,
            observed_auc=metrics["roc_auc"],
        )
        metrics.update(perm)

    metrics.update(
        {
            "analysis": "scn1a_knn_label_transfer",
            "feature_set": feature_name,
            "candidate_mode": candidate_mode,
            "k": k,
        }
    )

    return pred_df, metrics


def run_global_cross_gene_knn(meta, X, feature_name, k):
    """
    For every variant, find nearest neighbors from other genes only.
    Measures whether same-mechanism variants are closer than expected.
    """
    y_all = meta["Mechanism_Binary"].astype(int).values

    rows = []

    for query_idx in meta.index:
        query_gene = meta.loc[query_idx, "Gene"]

        candidate_indices = meta.index[meta["Gene"] != query_gene].tolist()

        if len(candidate_indices) == 0:
            continue

        query_vec = X[[query_idx]]
        candidate_vecs = X[candidate_indices]

        sims = cosine_similarity_matrix(query_vec, candidate_vecs)[0]

        order = np.argsort(-sims)
        k_eff = min(k, len(order))

        top_global = [candidate_indices[i] for i in order[:k_eff]]

        same_label_rate = float(np.mean(y_all[top_global] == y_all[query_idx]))
        candidate_baseline_same_label_rate = float(
            np.mean(y_all[candidate_indices] == y_all[query_idx])
        )

        rows.append(
            {
                "feature_set": feature_name,
                "k": k,
                "query_index": query_idx,
                "query_variant": meta.loc[query_idx, "Variant_Key"],
                "query_gene": query_gene,
                "query_label": meta.loc[query_idx, "Mechanism_Label"],
                "same_label_rate_top_k": same_label_rate,
                "candidate_baseline_same_label_rate": candidate_baseline_same_label_rate,
                "lift_over_candidate_baseline": (
                    same_label_rate / candidate_baseline_same_label_rate
                    if candidate_baseline_same_label_rate > 0
                    else np.nan
                ),
                "top_neighbor_variants": ";".join(meta.loc[top_global, "Variant_Key"].tolist()),
                "top_neighbor_genes": ";".join(meta.loc[top_global, "Gene"].tolist()),
                "top_neighbor_labels": ";".join(meta.loc[top_global, "Mechanism_Label"].tolist()),
            }
        )

    df = pd.DataFrame(rows)

    summary = {
        "analysis": "global_cross_gene_same_mechanism_neighbor_enrichment",
        "feature_set": feature_name,
        "k": k,
        "n": int(len(df)),
        "mean_same_label_rate_top_k": float(df["same_label_rate_top_k"].mean()),
        "mean_candidate_baseline_same_label_rate": float(
            df["candidate_baseline_same_label_rate"].mean()
        ),
        "mean_lift_over_candidate_baseline": float(
            df["lift_over_candidate_baseline"].mean()
        ),
    }

    return df, summary


def build_discordant_pairs(meta, pair_mode):
    rows = []

    if pair_mode == "same_gene_same_position":
        grouped = meta.groupby(["Gene", "AA_Position"], dropna=False)

        for group_key, group in grouped:
            if len(group) < 2:
                continue

            gofs = group[group["Mechanism_Label"] == "GOF"]
            lofs = group[group["Mechanism_Label"] == "LOF"]

            for gof_idx in gofs.index:
                for lof_idx in lofs.index:
                    rows.append(
                        {
                            "pair_mode": pair_mode,
                            "group_key": str(group_key),
                            "gof_index": int(gof_idx),
                            "lof_index": int(lof_idx),
                        }
                    )

    elif pair_mode == "same_cid_cross_gene":
        grouped = meta.groupby("Family_Alignment_CID", dropna=False)

        for cid, group in grouped:
            if len(group) < 2:
                continue

            gofs = group[group["Mechanism_Label"] == "GOF"]
            lofs = group[group["Mechanism_Label"] == "LOF"]

            for gof_idx in gofs.index:
                for lof_idx in lofs.index:
                    if meta.loc[gof_idx, "Gene"] == meta.loc[lof_idx, "Gene"]:
                        continue

                    rows.append(
                        {
                            "pair_mode": pair_mode,
                            "group_key": f"CID_{cid}",
                            "gof_index": int(gof_idx),
                            "lof_index": int(lof_idx),
                        }
                    )

    elif pair_mode == "scn1a_same_cid_cross_gene":
        grouped = meta.groupby("Family_Alignment_CID", dropna=False)

        for cid, group in grouped:
            if len(group) < 2:
                continue

            gofs = group[group["Mechanism_Label"] == "GOF"]
            lofs = group[group["Mechanism_Label"] == "LOF"]

            for gof_idx in gofs.index:
                for lof_idx in lofs.index:
                    genes = {meta.loc[gof_idx, "Gene"], meta.loc[lof_idx, "Gene"]}

                    if "SCN1A" not in genes:
                        continue

                    if len(genes) < 2:
                        continue

                    rows.append(
                        {
                            "pair_mode": pair_mode,
                            "group_key": f"CID_{cid}",
                            "gof_index": int(gof_idx),
                            "lof_index": int(lof_idx),
                        }
                    )
    else:
        raise ValueError(f"Unknown pair_mode: {pair_mode}")

    return pd.DataFrame(rows)


def knn_score_for_single_variant(meta, X, query_idx, k, exclude_indices=None):
    y_all = meta["Mechanism_Binary"].astype(int).values

    if exclude_indices is None:
        exclude_indices = set()
    else:
        exclude_indices = set(exclude_indices)

    query_gene = meta.loc[query_idx, "Gene"]

    candidate_indices = [
        idx
        for idx in meta.index
        if meta.loc[idx, "Gene"] != query_gene and idx not in exclude_indices
    ]

    if len(candidate_indices) == 0:
        return np.nan

    query_vec = X[[query_idx]]
    candidate_vecs = X[candidate_indices]

    sims = cosine_similarity_matrix(query_vec, candidate_vecs)[0]

    order = np.argsort(-sims)
    k_eff = min(k, len(order))

    top_global = [candidate_indices[i] for i in order[:k_eff]]

    return float(np.mean(y_all[top_global]))


def evaluate_pairwise_discordant_challenge(meta, X, feature_name, pair_mode, k):
    pairs = build_discordant_pairs(meta, pair_mode)

    if len(pairs) == 0:
        return pairs, {
            "analysis": "pairwise_discordant_knn_score_ranking",
            "feature_set": feature_name,
            "pair_mode": pair_mode,
            "k": k,
            "n_pairs": 0,
            "pairwise_accuracy": np.nan,
        }

    rows = []

    for _, row in pairs.iterrows():
        gof_idx = int(row["gof_index"])
        lof_idx = int(row["lof_index"])

        # Exclude the paired variant to avoid direct answer leakage.
        gof_score = knn_score_for_single_variant(
            meta,
            X,
            gof_idx,
            k=k,
            exclude_indices={lof_idx},
        )

        lof_score = knn_score_for_single_variant(
            meta,
            X,
            lof_idx,
            k=k,
            exclude_indices={gof_idx},
        )

        if np.isnan(gof_score) or np.isnan(lof_score):
            success = np.nan
        elif gof_score > lof_score:
            success = 1.0
        elif gof_score < lof_score:
            success = 0.0
        else:
            success = 0.5

        rows.append(
            {
                "feature_set": feature_name,
                "pair_mode": pair_mode,
                "k": k,
                "group_key": row["group_key"],
                "gof_variant": meta.loc[gof_idx, "Variant_Key"],
                "lof_variant": meta.loc[lof_idx, "Variant_Key"],
                "gof_gene": meta.loc[gof_idx, "Gene"],
                "lof_gene": meta.loc[lof_idx, "Gene"],
                "gof_position": meta.loc[gof_idx, "AA_Position"],
                "lof_position": meta.loc[lof_idx, "AA_Position"],
                "gof_cid": meta.loc[gof_idx, "Family_Alignment_CID"],
                "lof_cid": meta.loc[lof_idx, "Family_Alignment_CID"],
                "gof_knn_gof_score": gof_score,
                "lof_knn_gof_score": lof_score,
                "pairwise_success": success,
            }
        )

    out = pd.DataFrame(rows)

    valid = out.dropna(subset=["pairwise_success"]).copy()

    summary = {
        "analysis": "pairwise_discordant_knn_score_ranking",
        "feature_set": feature_name,
        "pair_mode": pair_mode,
        "k": k,
        "n_pairs": int(len(valid)),
        "n_unique_groups": int(valid["group_key"].nunique()) if len(valid) else 0,
        "pairwise_accuracy": float(valid["pairwise_success"].mean()) if len(valid) else np.nan,
        "strict_successes": int((valid["pairwise_success"] == 1.0).sum()) if len(valid) else 0,
        "strict_failures": int((valid["pairwise_success"] == 0.0).sum()) if len(valid) else 0,
        "ties": int((valid["pairwise_success"] == 0.5).sum()) if len(valid) else 0,
    }

    return out, summary


def main():
    print("Reading metadata:")
    print(META_FILE)

    meta = pd.read_csv(META_FILE).reset_index(drop=True)

    print("\nDataset:")
    print("Rows:", len(meta))
    print("\nMechanism counts:")
    print(meta["Mechanism_Label"].value_counts().to_string())
    print("\nGene × mechanism:")
    print(pd.crosstab(meta["Gene"], meta["Mechanism_Label"]).to_string())

    y_all = meta["Mechanism_Binary"].astype(int).values

    print("\nLoading frozen ESM embeddings:")
    features = {}

    for feature_name, path in FEATURE_FILES.items():
        X = np.load(path)
        features[feature_name] = X
        print(f"{feature_name}: {X.shape}")

    all_metric_rows = []
    all_prediction_tables = []
    all_global_summaries = []
    all_pairwise_summaries = []
    all_pairwise_tables = []

    for feature_name, X in features.items():
        print("\n" + "=" * 100)
        print("FEATURE:", feature_name)
        print("=" * 100)

        for k in K_VALUES:
            for candidate_mode in ["all_non_scn1a", "same_cid_non_scn1a"]:
                pred_df, metrics = run_scn1a_knn_transfer(
                    meta,
                    X,
                    feature_name=feature_name,
                    candidate_mode=candidate_mode,
                    k=k,
                )

                if pred_df is not None:
                    all_prediction_tables.append(pred_df)
                    all_metric_rows.append(metrics)

            global_df, global_summary = run_global_cross_gene_knn(
                meta,
                X,
                feature_name=feature_name,
                k=k,
            )
            global_df.to_csv(
                OUTPUT_DIR / f"global_cross_gene_neighbors_{feature_name}_k{k}.csv",
                index=False,
            )
            all_global_summaries.append(global_summary)

            for pair_mode in [
                "same_gene_same_position",
                "same_cid_cross_gene",
                "scn1a_same_cid_cross_gene",
            ]:
                pair_df, pair_summary = evaluate_pairwise_discordant_challenge(
                    meta,
                    X,
                    feature_name=feature_name,
                    pair_mode=pair_mode,
                    k=k,
                )

                pair_df.to_csv(
                    OUTPUT_DIR / f"pairwise_{pair_mode}_{feature_name}_k{k}.csv",
                    index=False,
                )

                all_pairwise_summaries.append(pair_summary)
                all_pairwise_tables.append(pair_df)

    metrics_df = pd.DataFrame(all_metric_rows)
    predictions_df = pd.concat(all_prediction_tables, ignore_index=True)
    global_summary_df = pd.DataFrame(all_global_summaries)
    pairwise_summary_df = pd.DataFrame(all_pairwise_summaries)

    metrics_df.to_csv(OUTPUT_DIR / "scn1a_knn_label_transfer_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "scn1a_knn_label_transfer_predictions.csv", index=False)
    global_summary_df.to_csv(OUTPUT_DIR / "global_cross_gene_neighbor_enrichment_summary.csv", index=False)
    pairwise_summary_df.to_csv(OUTPUT_DIR / "pairwise_discordant_challenge_summary.csv", index=False)

    qc = {
        "n_variants": int(len(meta)),
        "mechanism_counts": meta["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": meta["Gene"].value_counts().to_dict(),
        "n_scn1a": int((meta["Gene"] == "SCN1A").sum()),
        "n_scn1a_with_same_cid_non_scn1a_analog": int(
            sum(
                (
                    (meta["Gene"] == "SCN1A")
                    & meta["Family_Alignment_CID"].isin(
                        meta.loc[meta["Gene"] != "SCN1A", "Family_Alignment_CID"]
                    )
                )
            )
        ),
        "feature_sets": {k: list(v.shape) for k, v in features.items()},
        "k_values": K_VALUES,
        "n_permutations": N_PERMUTATIONS,
        "output_dir": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "embedding_only_paralog_analysis_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("EMBEDDING-ONLY PARALOG ANALYSIS QC")
    print("=" * 100)
    print(json.dumps(qc, indent=4))

    print("\n" + "=" * 100)
    print("SCN1A KNN LABEL TRANSFER METRICS")
    print("=" * 100)
    print(
        metrics_df.sort_values(
            ["candidate_mode", "k", "roc_auc"],
            ascending=[True, True, False],
        ).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("GLOBAL CROSS-GENE SAME-MECHANISM NEIGHBOR ENRICHMENT")
    print("=" * 100)
    print(
        global_summary_df.sort_values(
            ["k", "mean_lift_over_candidate_baseline"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("PAIRWISE DISCORDANT CHALLENGE SUMMARY")
    print("=" * 100)
    print(
        pairwise_summary_df.sort_values(
            ["pair_mode", "k", "pairwise_accuracy"],
            ascending=[True, True, False],
        ).to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()