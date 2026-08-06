from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = [
    {"Method": "Majority class", "Balanced accuracy": 0.500, "ROC-AUC": 0.500},
    {"Method": "SCION features: logistic regression", "Balanced accuracy": 0.577, "ROC-AUC": 0.616},
    {"Method": "SCION features: kNN k=1", "Balanced accuracy": 0.651, "ROC-AUC": 0.651},
    {"Method": "SCION features: gradient boosting", "Balanced accuracy": 0.639, "ROC-AUC": 0.675},
    {"Method": "SCION features: random forest", "Balanced accuracy": 0.671, "ROC-AUC": 0.733},
    {"Method": "ESM kNN WT+mutant k=1", "Balanced accuracy": 0.742, "ROC-AUC": 0.742},
    {"Method": "ESM kNN WT+mutant k=3", "Balanced accuracy": 0.764, "ROC-AUC": 0.787},
    {"Method": "ESM linear probe WT+mutant", "Balanced accuracy": 0.680, "ROC-AUC": 0.786},
]

df = pd.DataFrame(data)

# Sort by ROC-AUC for clean visual ranking.
df = df.sort_values("ROC-AUC", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 5.5))

y = range(len(df))
bar_height = 0.36

ax.barh(
    [i - bar_height / 2 for i in y],
    df["Balanced accuracy"],
    height=bar_height,
    label="Balanced accuracy",
)

ax.barh(
    [i + bar_height / 2 for i in y],
    df["ROC-AUC"],
    height=bar_height,
    label="ROC-AUC",
)

ax.axvline(0.5, linestyle="--", linewidth=1)
ax.set_yticks(list(y))
ax.set_yticklabels(df["Method"])
ax.set_xlim(0.45, 0.85)
ax.set_xlabel("Score")
ax.set_title("Held-out SCN1A GoF/LoF prediction")
ax.legend(frameon=False)

for i, row in df.iterrows():
    ax.text(row["Balanced accuracy"] + 0.006, i - bar_height / 2, f"{row['Balanced accuracy']:.3f}", va="center", fontsize=8)
    ax.text(row["ROC-AUC"] + 0.006, i + bar_height / 2, f"{row['ROC-AUC']:.3f}", va="center", fontsize=8)

plt.tight_layout()

png_path = OUT_DIR / "figure_baseline_comparison_scn1a.png"
pdf_path = OUT_DIR / "figure_baseline_comparison_scn1a.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Saved:")
print(png_path)
print(pdf_path)