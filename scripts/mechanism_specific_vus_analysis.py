from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal, mannwhitneyu, fisher_exact


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "vus_prioritization"
RANKED_DIR = INPUT_DIR / "ranked_vus_candidates"
DOMAIN_BLIND_DIR = INPUT_DIR / "domain_blind_enrichment"

INPUT_FILE = DOMAIN_BLIND_DIR / "vus_with_domain_blind_scores.csv"

if not INPUT_FILE.exists():
    INPUT_FILE = RANKED_DIR / "ranked_vus_experimental_followup_candidates.csv"

OUTPUT_DIR = INPUT_DIR / "mechanism_specific_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TOP_K_VALUES = [10, 25, 50, 100]


def normalize_text(x):
    return str(x).strip().upper().replace("-", "_").replace(" ", "_")


def infer_repeat(region):
    region_norm = normalize_text(region)

    # Check longer symbols first so DIII does not get caught as DI.
    if "DIV" in region_norm or "D4" in region_norm:
        return "DIV"
    if "DIII" in region_norm or "D3" in region_norm:
        return "DIII"
    if "DII" in region_norm or "D2" in region_norm:
        return "DII"
    if "DI" in region_norm or "D1" in region_norm:
        return "DI"

    return "Unknown"


def infer_mechanistic_subregion(row):
    region = normalize_text(row.get("Mapped_Region", ""))
    category = normalize_text(row.get("Mapped_Functional_Category", ""))
    position = int(row.get("AA_Position", -1))

    # Fast inactivation gate / III-IV linker
    if (
        "INACTIVATION" in category
        or "INACTIVATION" in region
        or "III_IV" in region
        or "DIII_DIV" in region
        or "D3_D4" in region
    ):
        return "Fast_Inactivation_Gate"

    # Voltage-sensing S4 helix
    if "VOLTAGE_SENSOR" in category or "S4" in region:
        return "S4_Voltage_Sensing_Helix"

    # Pore/selectivity region
    if (
        "PORE" in category
        or "PORE" in region
        or "SELECTIVITY" in region
        or "P_LOOP" in region
        or "P-LOOP" in region
        or "S5" in region
        or "S6" in region
    ):
        return "Ion_Conduction_Pore_Region"

    # Other transmembrane segments that are not annotated as voltage sensor/pore
    if "S1" in region or "S2" in region or "S3" in region:
        return "Voltage_Sensor_Scaffold_S1_S3"

    # Linker/loop regions
    if "LINKER" in region or "LOOP" in region:
        return "Loop_or_Linker"

    # Termini
    if "N_TERM" in region or "N-TERM" in region or "N_TERMINUS" in region:
        return "N_Terminus"

    if "C_TERM" in region or "C-TERM" in region or "C_TERMINUS" in region:
        return "C_Terminus"

    return "Other"


def infer_mechanism_class(subregion):
    if subregion in [
        "S4_Voltage_Sensing_Helix",
        "Fast_Inactivation_Gate",
        "Voltage_Sensor_Scaffold_S1_S3",
    ]:
        return "Dynamic_Gating_Related"

    if subregion == "Ion_Conduction_Pore_Region":
        return "Ion_Conduction_Pore"

    return "Other_Context"


def is_conflicting(clinical_text):
    return int("conflicting" in str(clinical_text).lower())


def summarize_by_group(df, group_col):
    metric_cols = [
        "Pathogenic_Probability_Mean",
        "Model_Uncertainty",
        "Ensemble_Disagreement",
        "neighbor_pathogenic_fraction_k20",
    ]

    if "Domain_Blind_Ambiguity_Score" in df.columns:
        metric_cols.append("Domain_Blind_Ambiguity_Score")

    if "Pure_Model_Ambiguity_Score" in df.columns:
        metric_cols.append("Pure_Model_Ambiguity_Score")

    summary = (
        df
        .groupby(group_col)
        .agg(
            n=("Variant_ID", "count"),
            conflicting_count=("Is_Conflicting", "sum"),
            conflicting_fraction=("Is_Conflicting", "mean"),
            mean_pathogenic_probability=("Pathogenic_Probability_Mean", "mean"),
            median_pathogenic_probability=("Pathogenic_Probability_Mean", "median"),
            mean_uncertainty=("Model_Uncertainty", "mean"),
            median_uncertainty=("Model_Uncertainty", "median"),
            mean_ensemble_disagreement=("Ensemble_Disagreement", "mean"),
            median_ensemble_disagreement=("Ensemble_Disagreement", "median"),
            mean_neighbor_pathogenic_fraction=("neighbor_pathogenic_fraction_k20", "mean"),
            median_neighbor_pathogenic_fraction=("neighbor_pathogenic_fraction_k20", "median"),
            mean_domain_blind_ambiguity=("Domain_Blind_Ambiguity_Score", "mean")
                if "Domain_Blind_Ambiguity_Score" in df.columns else ("Model_Uncertainty", "mean"),
            mean_pure_model_ambiguity=("Pure_Model_Ambiguity_Score", "mean")
                if "Pure_Model_Ambiguity_Score" in df.columns else ("Model_Uncertainty", "mean"),
        )
        .reset_index()
        .sort_values("mean_uncertainty", ascending=False)
    )

    return summary


def kruskal_by_group(df, group_col, metric_col, min_n=5):
    groups = []

    for name, sub in df.groupby(group_col):
        values = sub[metric_col].dropna().values

        if len(values) >= min_n:
            groups.append((name, values))

    if len(groups) < 2:
        return {
            "group_col": group_col,
            "metric_col": metric_col,
            "n_groups_used": len(groups),
            "H_statistic": np.nan,
            "p_value": np.nan,
            "groups_used": ",".join([g[0] for g in groups]),
        }

    H, p = kruskal(*[g[1] for g in groups])

    return {
        "group_col": group_col,
        "metric_col": metric_col,
        "n_groups_used": len(groups),
        "H_statistic": float(H),
        "p_value": float(p),
        "groups_used": ",".join([g[0] for g in groups]),
    }


def pairwise_mannwhitney(df, group_col, metric_col, min_n=5):
    rows = []
    groups = []

    for name, sub in df.groupby(group_col):
        values = sub[metric_col].dropna().values
        if len(values) >= min_n:
            groups.append((name, values))

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            name_a, values_a = groups[i]
            name_b, values_b = groups[j]

            stat, p = mannwhitneyu(
                values_a,
                values_b,
                alternative="two-sided",
            )

            rows.append({
                "group_col": group_col,
                "metric_col": metric_col,
                "group_a": name_a,
                "group_b": name_b,
                "n_a": len(values_a),
                "n_b": len(values_b),
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "median_a": float(np.median(values_a)),
                "median_b": float(np.median(values_b)),
                "mannwhitney_u": float(stat),
                "p_value_uncorrected": float(p),
            })

    pairwise = pd.DataFrame(rows)

    if len(pairwise) > 0:
        pairwise["bonferroni_p_value"] = np.minimum(
            pairwise["p_value_uncorrected"] * len(pairwise),
            1.0,
        )

    return pairwise


def top_k_mechanism_enrichment(df, score_col):
    rows = []

    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    mechanism_values = sorted(df["Mechanism_Class"].dropna().unique())

    for top_k in TOP_K_VALUES:
        top = ranked.head(top_k)
        rest = ranked.iloc[top_k:]

        for mechanism in mechanism_values:
            a = int((top["Mechanism_Class"] == mechanism).sum())
            b = int(len(top) - a)
            c = int((rest["Mechanism_Class"] == mechanism).sum())
            d = int(len(rest) - c)

            table = [[a, b], [c, d]]

            odds_ratio, p_value = fisher_exact(table, alternative="greater")

            background_count = int((df["Mechanism_Class"] == mechanism).sum())
            background_fraction = background_count / len(df)
            top_fraction = a / top_k

            fold_enrichment = (
                top_fraction / background_fraction
                if background_fraction > 0
                else np.nan
            )

            rows.append({
                "score_col": score_col,
                "top_k": top_k,
                "mechanism": mechanism,
                "top_count": a,
                "background_count": background_count,
                "expected_count_by_background": top_k * background_fraction,
                "top_fraction": top_fraction,
                "background_fraction": background_fraction,
                "fold_enrichment": fold_enrichment,
                "fisher_odds_ratio": odds_ratio,
                "fisher_p_value": p_value,
            })

    return pd.DataFrame(rows)


def make_bar_plot(summary, group_col, value_col, filename, title):
    plot_df = summary.sort_values(value_col, ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df[group_col], plot_df[value_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(value_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main():
    print("Reading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    if "Ensemble_Disagreement" not in df.columns:
        df["Ensemble_Disagreement"] = df["Pathogenic_Probability_Std"]

    if "Domain_Blind_Ambiguity_Score" not in df.columns:
        def minmax(values):
            values = np.asarray(values, dtype=float)
            if np.all(values == values[0]):
                return np.zeros(len(values))
            return (values - values.min()) / (values.max() - values.min())

        df["Uncertainty_Normalized_DomainBlind"] = minmax(df["Model_Uncertainty"])
        df["Ensemble_Disagreement_Normalized_DomainBlind"] = minmax(df["Ensemble_Disagreement"])
        df["Neighbor_Pathogenicity_Normalized_DomainBlind"] = minmax(df["neighbor_pathogenic_fraction_k20"])

        df["Domain_Blind_Ambiguity_Score"] = (
            0.45 * df["Uncertainty_Normalized_DomainBlind"]
            + 0.30 * df["Ensemble_Disagreement_Normalized_DomainBlind"]
            + 0.25 * df["Neighbor_Pathogenicity_Normalized_DomainBlind"]
        )

        df["Pure_Model_Ambiguity_Score"] = (
            0.60 * df["Uncertainty_Normalized_DomainBlind"]
            + 0.40 * df["Ensemble_Disagreement_Normalized_DomainBlind"]
        )

    df["Mechanistic_Subregion"] = df.apply(infer_mechanistic_subregion, axis=1)
    df["Mechanism_Class"] = df["Mechanistic_Subregion"].apply(infer_mechanism_class)
    df["Channel_Repeat"] = df["Mapped_Region"].apply(infer_repeat)
    df["Is_Conflicting"] = df["Clinical_Significance"].apply(is_conflicting)

    df.to_csv(
        OUTPUT_DIR / "vus_with_mechanistic_subregions.csv",
        index=False,
    )

    subregion_summary = summarize_by_group(df, "Mechanistic_Subregion")
    mechanism_summary = summarize_by_group(df, "Mechanism_Class")
    repeat_summary = summarize_by_group(df, "Channel_Repeat")

    subregion_summary.to_csv(
        OUTPUT_DIR / "mechanistic_subregion_summary.csv",
        index=False,
    )

    mechanism_summary.to_csv(
        OUTPUT_DIR / "mechanism_class_summary.csv",
        index=False,
    )

    repeat_summary.to_csv(
        OUTPUT_DIR / "channel_repeat_summary.csv",
        index=False,
    )

    stats_rows = []

    for group_col in ["Mechanistic_Subregion", "Mechanism_Class", "Channel_Repeat"]:
        for metric_col in [
            "Model_Uncertainty",
            "Ensemble_Disagreement",
            "Pathogenic_Probability_Mean",
            "neighbor_pathogenic_fraction_k20",
            "Domain_Blind_Ambiguity_Score",
            "Pure_Model_Ambiguity_Score",
        ]:
            stats_rows.append(
                kruskal_by_group(df, group_col, metric_col, min_n=5)
            )

    stats_df = pd.DataFrame(stats_rows)

    stats_df.to_csv(
        OUTPUT_DIR / "mechanism_kruskal_tests.csv",
        index=False,
    )

    pairwise_uncertainty = pairwise_mannwhitney(
        df,
        group_col="Mechanism_Class",
        metric_col="Model_Uncertainty",
        min_n=5,
    )

    pairwise_ambiguity = pairwise_mannwhitney(
        df,
        group_col="Mechanism_Class",
        metric_col="Domain_Blind_Ambiguity_Score",
        min_n=5,
    )

    pairwise_uncertainty.to_csv(
        OUTPUT_DIR / "pairwise_mechanism_class_uncertainty_tests.csv",
        index=False,
    )

    pairwise_ambiguity.to_csv(
        OUTPUT_DIR / "pairwise_mechanism_class_domain_blind_ambiguity_tests.csv",
        index=False,
    )

    enrichment_domain_blind = top_k_mechanism_enrichment(
        df,
        "Domain_Blind_Ambiguity_Score",
    )

    enrichment_pure = top_k_mechanism_enrichment(
        df,
        "Pure_Model_Ambiguity_Score",
    )

    enrichment_domain_blind.to_csv(
        OUTPUT_DIR / "mechanism_enrichment_domain_blind_score.csv",
        index=False,
    )

    enrichment_pure.to_csv(
        OUTPUT_DIR / "mechanism_enrichment_pure_model_score.csv",
        index=False,
    )

    make_bar_plot(
        mechanism_summary,
        group_col="Mechanism_Class",
        value_col="mean_uncertainty",
        filename="mechanism_class_mean_uncertainty.png",
        title="Mean VUS uncertainty by mechanism class",
    )

    make_bar_plot(
        subregion_summary,
        group_col="Mechanistic_Subregion",
        value_col="mean_uncertainty",
        filename="mechanistic_subregion_mean_uncertainty.png",
        title="Mean VUS uncertainty by mechanistic subregion",
    )

    make_bar_plot(
        mechanism_summary,
        group_col="Mechanism_Class",
        value_col="mean_domain_blind_ambiguity",
        filename="mechanism_class_domain_blind_ambiguity.png",
        title="Mean domain-blind ambiguity by mechanism class",
    )

    display_cols = [
        "Variant_ID",
        "Protein_Change_Parsed",
        "Clinical_Significance",
        "Mapped_Region",
        "Mapped_Functional_Category",
        "Mechanistic_Subregion",
        "Mechanism_Class",
        "Pathogenic_Probability_Mean",
        "Model_Uncertainty",
        "Ensemble_Disagreement",
        "neighbor_pathogenic_fraction_k20",
        "Domain_Blind_Ambiguity_Score",
        "Pure_Model_Ambiguity_Score",
    ]

    top_domain_blind = (
        df.sort_values("Domain_Blind_Ambiguity_Score", ascending=False)
        .head(50)
    )

    top_pure = (
        df.sort_values("Pure_Model_Ambiguity_Score", ascending=False)
        .head(50)
    )

    top_domain_blind[display_cols].to_csv(
        OUTPUT_DIR / "top50_mechanism_domain_blind_ambiguity.csv",
        index=False,
    )

    top_pure[display_cols].to_csv(
        OUTPUT_DIR / "top50_mechanism_pure_model_ambiguity.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("MECHANISM CLASS SUMMARY")
    print("=" * 100)
    print(mechanism_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("MECHANISTIC SUBREGION SUMMARY")
    print("=" * 100)
    print(subregion_summary.to_string(index=False))

    print("\n" + "=" * 100)
    print("KRUSKAL TESTS")
    print("=" * 100)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("PAIRWISE MECHANISM CLASS TESTS: UNCERTAINTY")
    print("=" * 100)
    print(pairwise_uncertainty.to_string(index=False))

    print("\n" + "=" * 100)
    print("PAIRWISE MECHANISM CLASS TESTS: DOMAIN-BLIND AMBIGUITY")
    print("=" * 100)
    print(pairwise_ambiguity.to_string(index=False))

    print("\n" + "=" * 100)
    print("MECHANISM ENRICHMENT: DOMAIN-BLIND SCORE")
    print("=" * 100)
    print(enrichment_domain_blind.to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP 25 MECHANISM-SPECIFIC DOMAIN-BLIND AMBIGUOUS VUS")
    print("=" * 100)
    print(top_domain_blind[display_cols].head(25).to_string(index=False))

    print("\nSaved outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()