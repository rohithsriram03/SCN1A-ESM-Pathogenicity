from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = PROJECT_ROOT / "results" / "baseline_model" / "baseline_predictions.csv"
OUT = PROJECT_ROOT / "results" / "uncertainty_analysis"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# ----------------------------
# Overall Kruskal-Wallis test
# ----------------------------

groups = [
    group["Uncertainty"].values
    for _, group in df.groupby("Mapped_Functional_Category")
]

H, p = kruskal(*groups)

print("\nKruskal-Wallis")
print(f"H = {H:.3f}")
print(f"p = {p:.6f}")

# ----------------------------
# Dunn's post hoc test
# ----------------------------

dunn = sp.posthoc_dunn(
    df,
    val_col="Uncertainty",
    group_col="Mapped_Functional_Category",
    p_adjust="bonferroni"
)

print("\nDunn's Test (Bonferroni corrected):")
print(dunn)

dunn.to_csv(OUT / "dunn_posthoc_results.csv")

# ----------------------------
# Publication figure
# ----------------------------

order = [
    "Other",
    "Pore",
    "Voltage_Sensor",
    "Inactivation_Gate"
]

plot_data = []

for category in order:
    plot_data.append(
        df[df["Mapped_Functional_Category"] == category]["Uncertainty"]
    )

plt.figure(figsize=(8,6))

plt.boxplot(
    plot_data,
    tick_labels=order
)

plt.ylabel("Model Uncertainty")

plt.title(
    "Prediction Uncertainty Across Functional Sodium Channel Regions"
)

plt.tight_layout()

plt.savefig(
    OUT / "uncertainty_boxplot.png",
    dpi=300
)

print("\nSaved figure.")