from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "global_neighbor_enrichment_statistics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
}

K_VALUES = [1, 3, 5]
N_BOOTSTRAPS = 10000
N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms


def cosine_similarity_matrix(X_query, X_candidate):
    X_query = l2_normalize(X_query)
    X_candidate = l2_normalize(X_candidate)
    return X_query @ X_candidate.T


def get_cross_gene_neighbor_indices(meta, X, k):
    """
    For each variant, find top-k nearest neighbors from OTHER genes only.
    """
    neighbor_lists = []
    candidate_lists = []

    for query_idx in meta.index:
        query_gene = meta.loc[query_idx, "Gene"]

        candidate_indices = meta.index[meta["Gene"] != query_gene].tolist()

        query_vec = X[[query_idx]]
        candidate_vecs = X[candidate_indices]

        sims = cosine_similarity_matrix(query_vec, candidate_vecs)[0]

        order = np.argsort(-sims)
        k_eff = min(k, len(order))

        top_neighbors = [candidate_indices[i] for i in order[:k_eff]]

        neighbor_lists.append(np.array(top_neighbors, dtype=int))
        candidate_lists.append(np.array(candidate_indices, dtype=int))

    return neighbor_lists, candidate_lists


def compute_same_label_rates(y, neighbor_lists, candidate_lists):
    """
    Computes:
    1. same-label rate among top-k nearest neighbors
    2. same-label rate expected from all cross-gene candidates
    3. per-variant lift
    """
    y = np.array(y).astype(int)

    topk_rates = []
    baseline_rates = []
    lifts = []

    for query_idx, neighbors in enumerate(neighbor_lists):
        candidates = candidate_lists[query_idx]

        topk_same = np.mean(y[neighbors] == y[query_idx])
        baseline_same = np.mean(y[candidates] == y[query_idx])

        topk_rates.append(topk_same)
        baseline_rates.append(baseline_same)

        if baseline_same > 0:
            lifts.append(topk_same / baseline_same)
        else:
            lifts.append(np.nan)

    return (
        np.array(topk_rates, dtype=float),
        np.array(baseline_rates, dtype=float),
        np.array(lifts, dtype=float),
    )


def bootstrap_mean_ci(values, n_boot=N_BOOTSTRAPS, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]

    boots = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        boots.append(np.mean(values[idx]))

    boots = np.array(boots)

    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def permute_labels_global(y, rng):
    return rng.permutation(y)


def permute_labels_within_gene(y, genes, rng):
    """
    Preserves each gene's GoF/LoF balance.
    This is a stricter null because it controls for gene-level class imbalance.
    """
    y_perm = np.array(y).copy()
    genes = np.array(genes)

    for gene in np.unique(genes):
        idx = np.where(genes == gene)[0]
        y_perm[idx] = rng.permutation(y_perm[idx])

    return y_perm


def permutation_test(
    y,
    genes,
    neighbor_lists,
    candidate_lists,
    observed_mean_topk,
    mode,
    n_perm=N_PERMUTATIONS,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)

    null_values = []

    for _ in range(n_perm):
        if mode == "global_label_permutation":
            y_perm = permute_labels_global(y, rng)
        elif mode == "within_gene_label_permutation":
            y_perm = permute_labels_within_gene(y, genes, rng)
        else:
            raise ValueError(f"Unknown permutation mode: {mode}")

        topk_rates, _, _ = compute_same_label_rates(
            y_perm,
            neighbor_lists,
            candidate_lists,
        )

        null_values.append(np.mean(topk_rates))

    null_values = np.array(null_values)

    p_value = (np.sum(null_values >= observed_mean_topk) + 1) / (len(null_values) + 1)

    return {
        f"{mode}_null_mean": float(np.mean(null_values)),
        f"{mode}_null_std": float(np.std(null_values)),
        f"{mode}_p_value": float(p_value),
    }


def main():
    print("Reading metadata:")
    print(META_FILE)

    meta = pd.read_csv(META_FILE).reset_index(drop=True)

    y = meta["Mechanism_Binary"].astype(int).values
    genes = meta["Gene"].astype(str).values

    print("\nDataset:")
    print("Rows:", len(meta))
    print("\nMechanism counts:")
    print(meta["Mechanism_Label"].value_counts().to_string())
    print("\nGene counts:")
    print(meta["Gene"].value_counts().to_string())

    all_rows = []
    all_query_rows = []

    for feature_name, path in FEATURE_FILES.items():
        print("\n" + "=" * 100)
        print("FEATURE:", feature_name)
        print("=" * 100)

        X = np.load(path)
        print("Shape:", X.shape)

        for k in K_VALUES:
            print(f"\nRunning k={k}")

            neighbor_lists, candidate_lists = get_cross_gene_neighbor_indices(
                meta,
                X,
                k=k,
            )

            topk_rates, baseline_rates, lifts = compute_same_label_rates(
                y,
                neighbor_lists,
                candidate_lists,
            )

            observed_topk = float(np.mean(topk_rates))
            observed_baseline = float(np.mean(baseline_rates))
            observed_lift_ratio = (
                observed_topk / observed_baseline
                if observed_baseline > 0
                else np.nan
            )

            topk_ci = bootstrap_mean_ci(topk_rates, seed=RANDOM_SEED)
            baseline_ci = bootstrap_mean_ci(baseline_rates, seed=RANDOM_SEED + 1)
            lift_ci = bootstrap_mean_ci(lifts, seed=RANDOM_SEED + 2)

            global_perm = permutation_test(
                y,
                genes,
                neighbor_lists,
                candidate_lists,
                observed_mean_topk=observed_topk,
                mode="global_label_permutation",
                seed=RANDOM_SEED + 3,
            )

            within_gene_perm = permutation_test(
                y,
                genes,
                neighbor_lists,
                candidate_lists,
                observed_mean_topk=observed_topk,
                mode="within_gene_label_permutation",
                seed=RANDOM_SEED + 4,
            )

            row = {
                "feature_set": feature_name,
                "k": k,
                "n_variants": int(len(meta)),

                "observed_mean_same_label_topk": observed_topk,
                "observed_mean_same_label_topk_ci_low": topk_ci["ci_low"],
                "observed_mean_same_label_topk_ci_high": topk_ci["ci_high"],

                "observed_mean_cross_gene_baseline": observed_baseline,
                "observed_mean_cross_gene_baseline_ci_low": baseline_ci["ci_low"],
                "observed_mean_cross_gene_baseline_ci_high": baseline_ci["ci_high"],

                "observed_lift_ratio_mean_rates": observed_lift_ratio,

                "mean_per_variant_lift": lift_ci["mean"],
                "mean_per_variant_lift_ci_low": lift_ci["ci_low"],
                "mean_per_variant_lift_ci_high": lift_ci["ci_high"],
            }

            row.update(global_perm)
            row.update(within_gene_perm)

            all_rows.append(row)

            for query_idx, neighbors in enumerate(neighbor_lists):
                candidate_indices = candidate_lists[query_idx]

                top_neighbor_variants = meta.loc[neighbors, "Variant_Key"].tolist()
                top_neighbor_genes = meta.loc[neighbors, "Gene"].tolist()
                top_neighbor_labels = meta.loc[neighbors, "Mechanism_Label"].tolist()

                all_query_rows.append(
                    {
                        "feature_set": feature_name,
                        "k": k,
                        "query_index": query_idx,
                        "query_variant": meta.loc[query_idx, "Variant_Key"],
                        "query_gene": meta.loc[query_idx, "Gene"],
                        "query_label": meta.loc[query_idx, "Mechanism_Label"],
                        "same_label_rate_topk": topk_rates[query_idx],
                        "cross_gene_baseline_same_label_rate": baseline_rates[query_idx],
                        "lift_over_baseline": lifts[query_idx],
                        "top_neighbor_variants": ";".join(top_neighbor_variants),
                        "top_neighbor_genes": ";".join(top_neighbor_genes),
                        "top_neighbor_labels": ";".join(top_neighbor_labels),
                    }
                )

    results = pd.DataFrame(all_rows)
    query_table = pd.DataFrame(all_query_rows)

    results_file = OUTPUT_DIR / "global_neighbor_enrichment_with_statistics.csv"
    query_file = OUTPUT_DIR / "global_neighbor_enrichment_query_level_table.csv"

    results.to_csv(results_file, index=False)
    query_table.to_csv(query_file, index=False)

    qc = {
        "n_variants": int(len(meta)),
        "mechanism_counts": meta["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": meta["Gene"].value_counts().to_dict(),
        "feature_sets": list(FEATURE_FILES.keys()),
        "k_values": K_VALUES,
        "n_bootstraps": N_BOOTSTRAPS,
        "n_permutations": N_PERMUTATIONS,
        "results_file": str(results_file),
        "query_file": str(query_file),
    }

    with open(OUTPUT_DIR / "global_neighbor_enrichment_statistics_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("GLOBAL NEIGHBOR ENRICHMENT WITH STATISTICS")
    print("=" * 100)
    print(
        results.sort_values(
            ["k", "observed_lift_ratio_mean_rates"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()