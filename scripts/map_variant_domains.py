from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "SCN1A_variant_master_with_gnomAD_processed (1).xlsx"
OUTPUT_FILE = PROJECT_ROOT / "results" / "SCN1A_variant_master_domain_mapped.xlsx"

variants = pd.read_excel(INPUT_FILE, sheet_name="Combined_Training_Set")
domains = pd.read_excel(INPUT_FILE, sheet_name="Domain_Boundaries")

def map_domain(row):
    position = row["AA_Position"]
    gene = row["Gene"]

    if pd.isna(position):
        return pd.Series(["Unmapped", "Other"])

    position = int(position)

    matches = domains[
        (domains["Gene"] == gene) &
        (domains["Start"] <= position) &
        (domains["End"] >= position)
    ]

    if len(matches) == 0:
        return pd.Series(["Unmapped", "Other"])

    match = matches.iloc[0]
    return pd.Series([match["Region"], match["Functional_Category"]])

variants[["Mapped_Region", "Mapped_Functional_Category"]] = variants.apply(
    map_domain,
    axis=1
)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    variants.to_excel(writer, sheet_name="Variant_Domain_Map", index=False)
    domains.to_excel(writer, sheet_name="Domain_Boundaries", index=False)

print("Saved:", OUTPUT_FILE)

print("\nFunctional category counts:")
print(variants["Mapped_Functional_Category"].value_counts())

print("\nMapped region counts:")
print(variants["Mapped_Region"].value_counts())