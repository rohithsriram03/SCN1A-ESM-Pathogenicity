from pathlib import Path
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kruskal
import scikit_posthocs as sp


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "results" / "custom_nn_domain_uncertainty"


def summarize_uncertainty_by_domain(df, output_prefix):
    clean_df = df.dropna(
        subset=["Mapped_Functional_Category", "Uncertainty", "Correct"]
    ).copy()

    categories = sorted(clean_df["Mapped_Functional_Category"].unique())

    summary_rows = []

    for category in categories:
        sub = clean_df[clean_df["Mapped_Functional_Category"] == category]

        summary_rows.append({
            "Mapped_Functional_Category": category,
            "n": int(len(sub)),
            "mean_uncertainty": float(sub["Uncertainty"].mean()),
            "median_uncertainty": float(sub["Uncertainty"].median()),
            "std_uncertainty": float(sub["Uncertainty"].std()),
            "mean_confidence": float(sub["Confidence"].mean()),
            "accuracy": float(sub["Correct"].mean()),
            "mean_pathogenic_probability": float(sub["Pathogenic_Probability"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "mean_uncertainty",
        ascending=False
    )

    summary.to_csv(
        OUTPUT_DIR / f"{output_prefix}_uncertainty_by_domain_summary.csv",
        index=False,
    )

    valid_categories = []
    valid_groups = []

    for category in categories:
        values = clean_df.loc[
            clean_df["Mapped_Functional_Category"] == category,
            "Uncertainty"
        ].values

        if len(values) >= 2:
            valid_categories.append(category)
            valid_groups.append(values)

    h_stat, p_value = kruskal(*valid_groups)

    kruskal_results = {
        "test": "Kruskal-Wallis",
        "n_categories": len(valid_categories),
        "categories": valid_categories,
        "H_statistic": float(h_stat),
        "p_value": float(p_value),
        "input_file_type": output_prefix,
    }

    with open(OUTPUT_DIR / f"{output_prefix}_kruskal_wallis_results.json", "w") as f:
        json.dump(kruskal_results, f, indent=4)

    dunn = sp.posthoc_dunn(
        clean_df,
        val_col="Uncertainty",
        group_col="Mapped_Functional_Category",
        p_adjust="bonferroni",
    )

    dunn.to_csv(
        OUTPUT_DIR / f"{output_prefix}_dunn_posthoc_bonferroni.csv"
    )

    plt.figure(figsize=(8, 5))

    plot_data = [
        clean_df.loc[
            clean_df["Mapped_Functional_Category"] == category,
            "Uncertainty"
        ].values
        for category in valid_categories
    ]

    plt.boxplot(plot_data, showfliers=True)

    plt.xticks(
        ticks=range(1, len(valid_categories) + 1),
        labels=valid_categories,
        rotation=25,
        ha="right"
    )

    plt.ylabel("Prediction uncertainty")
    plt.xlabel("Functional region")
    plt.title("Custom NN uncertainty by SCN1A functional region")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{output_prefix}_uncertainty_by_domain_boxplot.png",
        dpi=300,
    )

    plt.close()

    return summary, kruskal_results, dunn


def main():
    variant_avg_path = OUTPUT_DIR / "custom_nn_domain_uncertainty_variant_averaged_predictions.csv"
    all_predictions_path = OUTPUT_DIR / "custom_nn_domain_uncertainty_predictions_all_seeds.csv"

    variant_avg = pd.read_csv(variant_avg_path)
    all_predictions = pd.read_csv(all_predictions_path)

    print("\nRunning uncertainty analysis on variant-averaged predictions...")
    variant_summary, variant_kw, variant_dunn = summarize_uncertainty_by_domain(
        variant_avg,
        output_prefix="variant_averaged",
    )

    print("\nVariant-averaged uncertainty by domain:")
    print(variant_summary.to_string(index=False))

    print("\nVariant-averaged Kruskal-Wallis:")
    print(variant_kw)

    print("\nVariant-averaged Dunn post hoc test:")
    print(variant_dunn)

    print("\nRunning uncertainty analysis on all seed-level predictions...")
    all_summary, all_kw, all_dunn = summarize_uncertainty_by_domain(
        all_predictions,
        output_prefix="all_seed_predictions",
    )

    print("\nAll seed-level uncertainty by domain:")
    print(all_summary.to_string(index=False))

    print("\nAll seed-level Kruskal-Wallis:")
    print(all_kw)

    print("\nSaved postprocessed uncertainty results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()