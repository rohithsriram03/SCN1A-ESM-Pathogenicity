from pathlib import Path
import pandas as pd
from scipy.stats import kruskal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "results" / "baseline_model" / "baseline_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "uncertainty_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Summary by biological category
summary = (
    df.groupby("Mapped_Functional_Category")
    .agg(
        n=("Variant_ID", "count"),
        mean_uncertainty=("Uncertainty", "mean"),
        median_uncertainty=("Uncertainty", "median"),
        mean_confidence=("Confidence", "mean"),
        accuracy=("Predicted_Label", lambda x: (x == df.loc[x.index, "True_Label"]).mean()),
    )
    .reset_index()
    .sort_values("mean_uncertainty", ascending=False)
)

summary.to_csv(OUTPUT_DIR / "uncertainty_by_region_summary.csv", index=False)

print("\nUncertainty by functional category:")
print(summary)

# Statistical test: are uncertainty distributions different across categories?
groups = [
    group["Uncertainty"].dropna().values
    for _, group in df.groupby("Mapped_Functional_Category")
    if len(group) >= 5
]

stat, p = kruskal(*groups)

stats_out = pd.DataFrame([{
    "test": "Kruskal-Wallis",
    "statistic": stat,
    "p_value": p
}])

stats_out.to_csv(OUTPUT_DIR / "uncertainty_region_stats.csv", index=False)

print("\nKruskal-Wallis test:")
print(stats_out)

# Highest uncertainty variants
top_uncertain = df.sort_values("Uncertainty", ascending=False).head(50)
top_uncertain.to_csv(OUTPUT_DIR / "top_50_uncertain_variants.csv", index=False)

print("\nSaved outputs to:", OUTPUT_DIR)