from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCION_BASELINE_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_feature_baselines"
    / "heldout_scn1a_scion_feature_baselines.csv"
)

ESM_KNN_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "embedding_only_paralog_analysis"
    / "scn1a_knn_label_transfer_metrics.csv"
)

LORA_SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "contrastive_lora_multiseed"
    / "summary_by_feature_k.csv"
)


def pct(x):
    return round(float(x) * 100, 1)


rows = []

# -----------------------------
# 1. SCION engineered-feature baselines
# -----------------------------
scion = pd.read_csv(SCION_BASELINE_FILE)

scion_method_names = {
    "majority_class": "Majority class, always LoF",
    "logistic_regression": "SCION features: logistic regression",
    "scion_feature_knn_k1": "SCION features: kNN, k=1",
    "hist_gradient_boosting": "SCION features: gradient boosting",
    "random_forest": "SCION features: random forest",
}

for model_key, display_name in scion_method_names.items():
    match = scion[scion["model"] == model_key]

    if len(match) == 0:
        print(f"WARNING: missing SCION baseline row for {model_key}")
        continue

    r = match.iloc[0]

    rows.append(
        {
            "Method": display_name,
            "Test setting": "Held-out SCN1A",
            "Raw accuracy": pct(r["accuracy"]),
            "Balanced accuracy": pct(r["balanced_accuracy"]),
        }
    )


# -----------------------------
# 2. Frozen ESM kNN results
# -----------------------------
esm = pd.read_csv(ESM_KNN_FILE)

esm = esm[esm["candidate_mode"] == "all_non_scn1a"].copy()

esm_methods = [
    ("mutant_site", 1, "Frozen ESM kNN, mutant-site, k=1"),
    ("wt_mut_site", 1, "Frozen ESM kNN, WT+mutant, k=1"),
    ("wt_mut_site", 3, "Frozen ESM kNN, WT+mutant, k=3"),
]

for feature_set, k, display_name in esm_methods:
    match = esm[
        (esm["feature_set"] == feature_set)
        & (esm["k"] == k)
    ]

    if len(match) == 0:
        print(f"WARNING: missing ESM kNN row for {feature_set}, k={k}")
        continue

    r = match.iloc[0]

    rows.append(
        {
            "Method": display_name,
            "Test setting": "Held-out SCN1A",
            "Raw accuracy": pct(r["accuracy"]),
            "Balanced accuracy": pct(r["balanced_accuracy"]),
        }
    )


# -----------------------------
# 3. Contrastive LoRA multi-seed result
# -----------------------------
if LORA_SUMMARY_FILE.exists():
    lora = pd.read_csv(LORA_SUMMARY_FILE)

    match = lora[
        (lora["feature_set"] == "mutant_site")
        & (lora["k"] == 1)
    ]

    if len(match) > 0:
        r = match.iloc[0]

        rows.append(
            {
                "Method": "Contrastive LoRA + ESM kNN, mutant-site, k=1",
                "Test setting": "Held-out SCN1A, 3-seed mean",
                "Raw accuracy": pct(r["mean_accuracy"]),
                "Balanced accuracy": pct(r["mean_balanced_accuracy"]),
            }
        )
    else:
        print("WARNING: missing LoRA mutant_site k=1 summary row")
else:
    print("WARNING: LoRA summary file not found; skipping LoRA row")


# -----------------------------
# Save table
# -----------------------------
table = pd.DataFrame(rows)

csv_path = OUT_DIR / "heldout_scn1a_accuracy_table.csv"
md_path = OUT_DIR / "heldout_scn1a_accuracy_table.md"
png_path = OUT_DIR / "heldout_scn1a_accuracy_table.png"

table.to_csv(csv_path, index=False)

with open(md_path, "w") as f:
    f.write(table.to_markdown(index=False))

print("\nFinal accuracy table:")
print(table.to_markdown(index=False))

print("\nSaved:")
print(csv_path)
print(md_path)


# -----------------------------
# Save figure version
# -----------------------------
plot_df = table.copy()
plot_df = plot_df.sort_values("Balanced accuracy", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5.8))

y = range(len(plot_df))
bar_height = 0.36

ax.barh(
    [i - bar_height / 2 for i in y],
    plot_df["Raw accuracy"],
    height=bar_height,
    label="Raw accuracy",
)

ax.barh(
    [i + bar_height / 2 for i in y],
    plot_df["Balanced accuracy"],
    height=bar_height,
    label="Balanced accuracy",
)

ax.axvline(50, linestyle="--", linewidth=1)
ax.set_yticks(list(y))
ax.set_yticklabels(plot_df["Method"])
ax.set_xlabel("Accuracy (%)")
ax.set_title("Held-out SCN1A GoF/LoF prediction accuracy")
ax.set_xlim(45, 85)
ax.legend(frameon=False)

for i, row in plot_df.iterrows():
    y_pos = list(plot_df.index).index(i)
    ax.text(row["Raw accuracy"] + 0.5, y_pos - bar_height / 2, f"{row['Raw accuracy']:.1f}", va="center", fontsize=8)
    ax.text(row["Balanced accuracy"] + 0.5, y_pos + bar_height / 2, f"{row['Balanced accuracy']:.1f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close()

print(png_path)