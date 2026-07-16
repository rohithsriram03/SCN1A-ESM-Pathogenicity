from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact, mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "vus_prioritization" / "ranked_vus_candidates"
INPUT_FILE = INPUT_DIR / "ranked_vus_experimental_followup_candidates.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "vus_prioritization" / "domain_blind_enrichment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_PERMUTATIONS = 10000

TOP_K_VALUES = [10, 25, 50, 100]

FUNCTIONAL_DOMAINS = {
    "Voltage_Sensor",
    "Pore",
    "Inactivation_Gate",
}


def minmax(series):
    values = series.astype(float).values

    if np.all(values == values[0]):
        return np.zeros(len(values))

    return (values - values.min()) / (values.max() - values.min())


def compute_domain_blind_scores(df):
    """
    Domain-blind means:
    - no functional domain priority
    - no AA position
    - no region label

    Only model-derived / embedding-derived signals are used.
    """

    df = df.copy()

    df["Uncertainty_Normalized_DomainBlind"] = minmax(df["Model_Uncertainty"])
    df["Ensemble_Disagreement_Normalized_DomainBlind"] = minmax(
        df["Ensemble_Disagreement"]
    )
    df["Neighbor_Pathogenicity_Normalized_DomainBlind"] = minmax(
        df["neighbor_pathogenic_fraction_k20"]
    )

    # Main domain-blind ambiguity score:
    # high uncertainty + high ensemble disagreement + mixed/pathogenic neighborhood
    df["Domain_Blind_Ambiguity_Score"] = (
        0.45 * df["Uncertainty_Normalized_DomainBlind"]
        + 0.30 * df["Ensemble_Disagreement_Normalized_DomainBlind"]
        + 0.25 * df["Neighbor_Pathogenicity_Normalized_DomainBlind"]
    )

    # A purer ambiguity score that does NOT include pathogenic-neighbor fraction.
    # This asks: does uncertainty/disagreement alone recover functional domains?
    df["Pure_Model_Ambiguity_Score"] = (
        0.60 * df["Uncertainty_Normalized_DomainBlind"]
        + 0.40 * df["Ensemble_Disagreement_Normalized_DomainBlind"]
    )

    return df


def add_functional_domain_flags(df):
    df = df.copy()

    df["Is_Functional_Domain"] = df["Mapped_Functional_Category"].isin(
        FUNCTIONAL_DOMAINS
    ).astype(int)

    return df


def fisher_enrichment_for_top_k(df, score_col, top_k):
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    top = ranked.head(top_k)
    rest = ranked.iloc[top_k:]

    a = int(top["Is_Functional_Domain"].sum())
    b = int(len(top) - a)
    c = int(rest["Is_Functional_Domain"].sum())
    d = int(len(rest) - c)

    table = [[a, b], [c, d]]

    odds_ratio, p_value = fisher_exact(table, alternative="greater")

    background_prop = df["Is_Functional_Domain"].mean()
    top_prop = top["Is_Functional_Domain"].mean()

    expected_count = top_k * background_prop

    fold_enrichment = (
        top_prop / background_prop
        if background_prop > 0
        else np.nan
    )

    return {
        "score_col": score_col,
        "top_k": top_k,
        "top_functional_count": a,
        "top_other_count": b,
        "rest_functional_count": c,
        "rest_other_count": d,
        "expected_functional_count_by_background": expected_count,
        "background_functional_fraction": background_prop,
        "top_functional_fraction": top_prop,
        "fold_enrichment": fold_enrichment,
        "fisher_odds_ratio": odds_ratio,
        "fisher_p_value": p_value,
    }


def permutation_enrichment_test(df, score_col, top_k, n_permutations=10000):
    rng = np.random.default_rng(RANDOM_SEED)

    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    observed = int(ranked.head(top_k)["Is_Functional_Domain"].sum())

    labels = df["Is_Functional_Domain"].values.copy()

    perm_counts = []

    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_count = int(shuffled[:top_k].sum())
        perm_counts.append(perm_count)

    perm_counts = np.array(perm_counts)

    p_value = (np.sum(perm_counts >= observed) + 1) / (n_permutations + 1)

    return {
        "score_col": score_col,
        "top_k": top_k,
        "observed_functional_count": observed,
        "permutation_mean_functional_count": float(perm_counts.mean()),
        "permutation_std_functional_count": float(perm_counts.std()),
        "permutation_p_value": float(p_value),
    }


def domain_specific_enrichment(df, score_col, top_k):
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    top = ranked.head(top_k)
    rest = ranked.iloc[top_k:]

    rows = []

    for domain in sorted(FUNCTIONAL_DOMAINS):
        top_domain = int((top["Mapped_Functional_Category"] == domain).sum())
        top_not_domain = int(len(top) - top_domain)

        rest_domain = int((rest["Mapped_Functional_Category"] == domain).sum())
        rest_not_domain = int(len(rest) - rest_domain)

        table = [[top_domain, top_not_domain], [rest_domain, rest_not_domain]]

        odds_ratio, p_value = fisher_exact(table, alternative="greater")

        background_count = int((df["Mapped_Functional_Category"] == domain).sum())
        background_fraction = background_count / len(df)

        top_fraction = top_domain / len(top)

        fold_enrichment = (
            top_fraction / background_fraction
            if background_fraction > 0
            else np.nan
        )

        rows.append({
            "score_col": score_col,
            "top_k": top_k,
            "domain": domain,
            "top_domain_count": top_domain,
            "background_domain_count": background_count,
            "expected_domain_count_by_background": top_k * background_fraction,
            "top_domain_fraction": top_fraction,
            "background_domain_fraction": background_fraction,
            "fold_enrichment": fold_enrichment,
            "fisher_odds_ratio": odds_ratio,
            "fisher_p_value": p_value,
        })

    return rows


def score_distribution_test(df, score_col):
    functional_scores = df[df["Is_Functional_Domain"] == 1][score_col]
    other_scores = df[df["Is_Functional_Domain"] == 0][score_col]

    stat, p_value = mannwhitneyu(
        functional_scores,
        other_scores,
        alternative="greater",
    )

    return {
        "score_col": score_col,
        "functional_n": int(len(functional_scores)),
        "other_n": int(len(other_scores)),
        "functional_mean_score": float(functional_scores.mean()),
        "other_mean_score": float(other_scores.mean()),
        "functional_median_score": float(functional_scores.median()),
        "other_median_score": float(other_scores.median()),
        "mannwhitney_u_statistic": float(stat),
        "mannwhitney_p_value": float(p_value),
    }


def save_top_ranked_tables(df):
    for score_col in [
        "Domain_Blind_Ambiguity_Score",
        "Pure_Model_Ambiguity_Score",
    ]:
        ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

        output_file = OUTPUT_DIR / f"top_vus_by_{score_col}.csv"
        ranked.to_csv(output_file, index=False)

        display_cols = [
            "Variant_ID",
            "Protein_Change_Parsed",
            "Clinical_Significance",
            "Mapped_Functional_Category",
            "Pathogenic_Probability_Mean",
            "Pathogenic_Probability_Std",
            "Model_Uncertainty",
            "Ensemble_Disagreement",
            "neighbor_pathogenic_fraction_k20",
            score_col,
        ]

        print("\n" + "=" * 100)
        print(f"TOP 25 VUS BY {score_col}")
        print("=" * 100)
        print(ranked[display_cols].head(25).to_string(index=False))


def make_enrichment_plot(enrichment_df):
    plot_df = enrichment_df[
        enrichment_df["score_col"] == "Domain_Blind_Ambiguity_Score"
    ].copy()

    plt.figure(figsize=(7, 5))
    plt.bar(
        plot_df["top_k"].astype(str),
        plot_df["fold_enrichment"],
    )
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Top K VUS by domain-blind ambiguity score")
    plt.ylabel("Fold enrichment for functional domains")
    plt.title("Domain-blind enrichment of functional SCN1A regions")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "domain_blind_functional_domain_fold_enrichment.png",
        dpi=300,
    )
    plt.close()


def main():
    print("Reading:", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    df = compute_domain_blind_scores(df)
    df = add_functional_domain_flags(df)

    print("\nTotal VUS:", len(df))
    print("Functional-domain VUS:", int(df["Is_Functional_Domain"].sum()))
    print("Other VUS:", int((1 - df["Is_Functional_Domain"]).sum()))

    print("\nFunctional category counts:")
    print(df["Mapped_Functional_Category"].value_counts().to_string())

    background_fraction = df["Is_Functional_Domain"].mean()
    print(f"\nBackground functional-domain fraction: {background_fraction:.4f}")

    score_cols = [
        "Domain_Blind_Ambiguity_Score",
        "Pure_Model_Ambiguity_Score",
    ]

    enrichment_rows = []
    permutation_rows = []
    domain_specific_rows = []
    distribution_rows = []

    for score_col in score_cols:
        distribution_rows.append(
            score_distribution_test(df, score_col)
        )

        for top_k in TOP_K_VALUES:
            enrichment_rows.append(
                fisher_enrichment_for_top_k(df, score_col, top_k)
            )

            permutation_rows.append(
                permutation_enrichment_test(
                    df,
                    score_col,
                    top_k,
                    n_permutations=N_PERMUTATIONS,
                )
            )

            domain_specific_rows.extend(
                domain_specific_enrichment(df, score_col, top_k)
            )

    enrichment_df = pd.DataFrame(enrichment_rows)
    permutation_df = pd.DataFrame(permutation_rows)
    domain_specific_df = pd.DataFrame(domain_specific_rows)
    distribution_df = pd.DataFrame(distribution_rows)

    df.to_csv(
        OUTPUT_DIR / "vus_with_domain_blind_scores.csv",
        index=False,
    )

    enrichment_df.to_csv(
        OUTPUT_DIR / "domain_blind_functional_domain_enrichment.csv",
        index=False,
    )

    permutation_df.to_csv(
        OUTPUT_DIR / "domain_blind_permutation_enrichment.csv",
        index=False,
    )

    domain_specific_df.to_csv(
        OUTPUT_DIR / "domain_blind_domain_specific_enrichment.csv",
        index=False,
    )

    distribution_df.to_csv(
        OUTPUT_DIR / "domain_blind_score_distribution_tests.csv",
        index=False,
    )

    make_enrichment_plot(enrichment_df)

    print("\n" + "=" * 100)
    print("FUNCTIONAL DOMAIN ENRICHMENT")
    print("=" * 100)
    print(enrichment_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("PERMUTATION ENRICHMENT")
    print("=" * 100)
    print(permutation_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("DOMAIN-SPECIFIC ENRICHMENT")
    print("=" * 100)
    print(domain_specific_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("SCORE DISTRIBUTION TESTS")
    print("=" * 100)
    print(distribution_df.to_string(index=False))

    save_top_ranked_tables(df)

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()