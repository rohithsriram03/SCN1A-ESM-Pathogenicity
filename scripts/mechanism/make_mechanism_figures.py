from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FIG_DIR = PROJECT_ROOT / "results" / "mechanism" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

META_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
    / "scion_embedding_metadata.csv"
)

ALPHA_MERGED_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scn1a_alphamissense_mechanism_comparison"
    / "scn1a_mechanism_with_alphamissense.csv"
)

ALPHA_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scn1a_alphamissense_mechanism_comparison"
    / "scn1a_gof_lof_score_comparison.csv"
)

KNN_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "embedding_only_paralog_analysis"
    / "scn1a_knn_label_transfer_metrics.csv"
)

GLOBAL_ENRICHMENT_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "global_neighbor_enrichment_statistics"
    / "global_neighbor_enrichment_with_statistics.csv"
)


def savefig(name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def figure_1_benchmark_overview():
    meta = pd.read_csv(META_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel A: overall GoF/LoF counts
    counts = meta["Mechanism_Label"].value_counts().reindex(["LOF", "GOF"])
    axes[0].bar(counts.index, counts.values)
    axes[0].set_title("A. SCN mechanism benchmark")
    axes[0].set_ylabel("Number of variants")
    axes[0].set_xlabel("Functional mechanism")

    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, str(v), ha="center", fontsize=10)

    # Panel B: gene-by-mechanism counts
    gene_counts = pd.crosstab(meta["Gene"], meta["Mechanism_Label"])
    gene_counts = gene_counts.reindex(gene_counts.sum(axis=1).sort_values(ascending=False).index)

    bottom = np.zeros(len(gene_counts))
    x = np.arange(len(gene_counts))

    for label in ["LOF", "GOF"]:
        values = gene_counts[label].values if label in gene_counts.columns else np.zeros(len(gene_counts))
        axes[1].bar(x, values, bottom=bottom, label=label)
        bottom += values

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gene_counts.index, rotation=45, ha="right")
    axes[1].set_ylabel("Number of variants")
    axes[1].set_xlabel("SCN gene")
    axes[1].set_title("B. Variants across sodium-channel paralogs")
    axes[1].legend(frameon=False)

    savefig("figure_1_benchmark_overview")


def figure_2_alphamissense_mechanism_failure():
    alpha = pd.read_csv(ALPHA_MERGED_FILE)
    metrics = pd.read_csv(ALPHA_METRICS_FILE)

    alpha = alpha[alpha["Gene"] == "SCN1A"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Panel A: AlphaMissense scores by mechanism
    groups = [
        alpha.loc[alpha["Mechanism_Label"] == "LOF", "AlphaMissense_Score"].dropna(),
        alpha.loc[alpha["Mechanism_Label"] == "GOF", "AlphaMissense_Score"].dropna(),
    ]

    axes[0].boxplot(groups, tick_labels=["LOF", "GOF"], showmeans=True)
    axes[0].set_title("A. AlphaMissense scores are high for both mechanisms")
    axes[0].set_ylabel("AlphaMissense pathogenicity score")
    axes[0].set_xlabel("SCN1A mechanism")

    # Add individual points
    for i, vals in enumerate(groups, start=1):
        jitter = np.random.default_rng(42).normal(0, 0.035, size=len(vals))
        axes[0].scatter(np.full(len(vals), i) + jitter, vals, alpha=0.65, s=20)

    # Panel B: AUC comparison
    keep_scores = [
        "ESM_mutant_site_mechanism_model",
        "ESM_wt_mut_site_mechanism_model",
        "biochem_only_model",
        "AlphaMissense_raw_pathogenicity_score",
    ]

    plot_df = metrics[metrics["score_name"].isin(keep_scores)].copy()
    plot_df["pretty_name"] = plot_df["score_name"].map(
        {
            "ESM_mutant_site_mechanism_model": "ESM mutant-site\nlinear probe",
            "ESM_wt_mut_site_mechanism_model": "ESM WT+mutant\nlinear probe",
            "biochem_only_model": "Biochemical\nbaseline",
            "AlphaMissense_raw_pathogenicity_score": "AlphaMissense\npathogenicity",
        }
    )

    plot_df = plot_df.sort_values("roc_auc", ascending=False)

    axes[1].bar(plot_df["pretty_name"], plot_df["roc_auc"])
    axes[1].axhline(0.5, linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("AUC for SCN1A GoF vs LoF")
    axes[1].set_title("B. Pathogenicity score does not resolve mechanism")

    for i, v in enumerate(plot_df["roc_auc"]):
        axes[1].text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=9)

    savefig("figure_2_alphamissense_vs_esm")


def figure_3_scn1a_knn_transfer():
    knn = pd.read_csv(KNN_METRICS_FILE)

    # Use all non-SCN1A neighbors because this is the cleanest broad transfer test.
    df = knn[
        (knn["candidate_mode"] == "all_non_scn1a")
        & (knn["feature_set"].isin(["wt_mut_site", "mutant_site", "wt_site", "delta_site"]))
    ].copy()

    pretty = {
        "wt_mut_site": "WT+mutant",
        "mutant_site": "Mutant site",
        "wt_site": "WT site",
        "delta_site": "Delta",
    }

    df["pretty_feature"] = df["feature_set"].map(pretty)

    fig, ax = plt.subplots(figsize=(9, 5))

    feature_order = ["WT+mutant", "Mutant site", "WT site", "Delta"]
    k_order = [1, 3, 5]

    width = 0.22
    x = np.arange(len(feature_order))

    for j, k in enumerate(k_order):
        vals = []
        for feature in feature_order:
            row = df[(df["pretty_feature"] == feature) & (df["k"] == k)]
            vals.append(float(row["roc_auc"].iloc[0]) if len(row) else np.nan)

        ax.bar(x + (j - 1) * width, vals, width=width, label=f"k={k}")

    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_order)
    ax.set_ylabel("AUC for SCN1A GoF vs LoF")
    ax.set_xlabel("Frozen ESM feature")
    ax.set_title("SCN1A mechanism transfer using only cross-gene ESM nearest neighbors")
    ax.legend(frameon=False)

    savefig("figure_3_scn1a_knn_transfer")


def figure_4_global_neighbor_enrichment():
    enrich = pd.read_csv(GLOBAL_ENRICHMENT_FILE)

    # Main figure: k=1, all feature sets
    df = enrich[enrich["k"] == 1].copy()

    order = ["wt_mut_site", "wt_site", "mutant_site", "delta_site"]
    pretty = {
        "wt_mut_site": "WT+mutant",
        "wt_site": "WT site",
        "mutant_site": "Mutant site",
        "delta_site": "Delta",
    }

    df["pretty_feature"] = df["feature_set"].map(pretty)
    df["feature_order"] = df["feature_set"].map({name: i for i, name in enumerate(order)})
    df = df.sort_values("feature_order")

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(df))
    y = df["observed_mean_same_label_topk"].values

    yerr_low = y - df["observed_mean_same_label_topk_ci_low"].values
    yerr_high = df["observed_mean_same_label_topk_ci_high"].values - y

    ax.bar(x, y, yerr=[yerr_low, yerr_high], capsize=5)

    baseline = float(df["observed_mean_cross_gene_baseline"].iloc[0])
    ax.axhline(baseline, linestyle="--", linewidth=1, label=f"Cross-gene baseline = {baseline:.3f}")

    ax.set_ylim(0, 0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["pretty_feature"])
    ax.set_ylabel("Same-mechanism rate among top-1 cross-gene neighbors")
    ax.set_xlabel("Frozen ESM feature")
    ax.set_title("Frozen ESM neighborhoods are enriched for shared GoF/LoF mechanism")
    ax.legend(frameon=False)

    for i, row in enumerate(df.itertuples()):
        ax.text(
            i,
            row.observed_mean_same_label_topk + 0.035,
            f"lift {row.observed_lift_ratio_mean_rates:.2f}×\np={row.within_gene_label_permutation_p_value:.4f}",
            ha="center",
            fontsize=8,
        )

    savefig("figure_4_global_neighbor_enrichment_k1")


def figure_5_enrichment_across_k():
    enrich = pd.read_csv(GLOBAL_ENRICHMENT_FILE)

    fig, ax = plt.subplots(figsize=(8, 5))

    pretty = {
        "wt_mut_site": "WT+mutant",
        "wt_site": "WT site",
        "mutant_site": "Mutant site",
        "delta_site": "Delta",
    }

    for feature_set, group in enrich.groupby("feature_set"):
        group = group.sort_values("k")
        ax.plot(
            group["k"],
            group["observed_lift_ratio_mean_rates"],
            marker="o",
            label=pretty.get(feature_set, feature_set),
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xticks(sorted(enrich["k"].unique()))
    ax.set_xlabel("Number of nearest cross-gene neighbors, k")
    ax.set_ylabel("Lift over cross-gene baseline")
    ax.set_title("Same-mechanism enrichment decreases as neighborhoods widen")
    ax.legend(frameon=False)

    savefig("figure_5_neighbor_lift_across_k")


def main():
    print("Creating figures in:")
    print(FIG_DIR)

    required = [
        META_FILE,
        ALPHA_MERGED_FILE,
        ALPHA_METRICS_FILE,
        KNN_METRICS_FILE,
        GLOBAL_ENRICHMENT_FILE,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    figure_1_benchmark_overview()
    figure_2_alphamissense_mechanism_failure()
    figure_3_scn1a_knn_transfer()
    figure_4_global_neighbor_enrichment()
    figure_5_enrichment_across_k()

    print("\nDone. Figures saved to:")
    print(FIG_DIR)


if __name__ == "__main__":
    main()