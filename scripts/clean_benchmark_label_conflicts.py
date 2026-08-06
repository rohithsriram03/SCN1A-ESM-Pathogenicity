from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmarking"

MASTER_FILE = BENCHMARK_DIR / "scn1a_benchmark_master_table.csv"
DUPLICATES_FILE = BENCHMARK_DIR / "scn1a_benchmark_duplicate_variants.csv"

CLEAN_OUTPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_master_table_clean.csv"
CONFLICT_OUTPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_label_conflicts_removed.csv"
QC_OUTPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_master_table_clean_qc.json"


KEY_COLS = ["Gene", "AA_Position", "Ref_AA", "Alt_AA"]


def make_variant_key(df):
    return (
        df["Gene"].astype(str)
        + ":p."
        + df["Ref_AA"].astype(str)
        + df["AA_Position"].astype(int).astype(str)
        + df["Alt_AA"].astype(str)
    )


def main():
    print("Reading master table:")
    print(MASTER_FILE)

    master = pd.read_csv(MASTER_FILE)

    print("\nReading duplicate rows:")
    print(DUPLICATES_FILE)

    duplicates = pd.read_csv(DUPLICATES_FILE)

    # Standardize position type
    master["AA_Position"] = pd.to_numeric(master["AA_Position"], errors="coerce").astype(int)
    duplicates["AA_Position"] = pd.to_numeric(duplicates["AA_Position"], errors="coerce").astype(int)

    # Identify exact protein substitutions with conflicting labels
    conflict_groups = (
        duplicates
        .groupby(KEY_COLS)["Binary_Label"]
        .nunique()
        .reset_index(name="n_unique_labels")
    )

    conflict_groups = conflict_groups[conflict_groups["n_unique_labels"] > 1].copy()

    print("\nNumber of exact protein substitutions with label conflicts:")
    print(len(conflict_groups))

    # Add key for filtering
    master["Variant_Key_Filter"] = make_variant_key(master)
    conflict_groups["Variant_Key_Filter"] = make_variant_key(conflict_groups)

    conflict_keys = set(conflict_groups["Variant_Key_Filter"])

    removed = master[master["Variant_Key_Filter"].isin(conflict_keys)].copy()
    clean = master[~master["Variant_Key_Filter"].isin(conflict_keys)].copy()

    # Remove helper column
    removed = removed.drop(columns=["Variant_Key_Filter"])
    clean = clean.drop(columns=["Variant_Key_Filter"])

    removed.to_csv(CONFLICT_OUTPUT_FILE, index=False)
    clean.to_csv(CLEAN_OUTPUT_FILE, index=False)

    qc = {
        "original_master_variants": int(len(master)),
        "label_conflict_variant_groups_removed": int(len(conflict_groups)),
        "removed_rows_from_master": int(len(removed)),
        "clean_final_variants": int(len(clean)),
        "class_counts_clean": clean["Label_Name"].value_counts().to_dict(),
        "functional_category_counts_clean": clean["Mapped_Functional_Category"].value_counts().to_dict(),
        "unique_positions_clean": int(clean["AA_Position"].nunique()),
        "clean_output_file": str(CLEAN_OUTPUT_FILE),
        "conflicts_removed_file": str(CONFLICT_OUTPUT_FILE),
    }

    with open(QC_OUTPUT_FILE, "w") as f:
        json.dump(qc, f, indent=4)

    print("\nSaved clean benchmark table:")
    print(CLEAN_OUTPUT_FILE)

    print("\nSaved removed conflicted variants:")
    print(CONFLICT_OUTPUT_FILE)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nPreview of clean table:")
    print(clean.head(10).to_string(index=False))


if __name__ == "__main__":
    main()