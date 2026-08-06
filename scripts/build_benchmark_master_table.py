from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CANDIDATES = [
    PROJECT_ROOT / "results" / "paired_embeddings" / "paired_embedding_metadata.csv",
    PROJECT_ROOT / "results" / "embedding_metadata.csv",
    PROJECT_ROOT / "results" / "SCN1A_mutant_sequences.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmarking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "scn1a_benchmark_master_table.csv"
QC_FILE = OUTPUT_DIR / "scn1a_benchmark_master_table_qc.json"
DUPLICATES_FILE = OUTPUT_DIR / "scn1a_benchmark_duplicate_variants.csv"
MISSING_FILE = OUTPUT_DIR / "scn1a_benchmark_missing_required_fields.csv"


REQUIRED_COLUMNS = [
    "Gene",
    "AA_Position",
    "Ref_AA",
    "Alt_AA",
    "Binary_Label",
]


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find a benchmark input file. Checked:\n"
        + "\n".join(str(path) for path in candidates)
    )


def normalize_column_names(df):
    df = df.copy()

    rename_map = {}

    possible_mappings = {
        "Gene": ["Gene", "gene"],
        "Variant_ID": ["Variant_ID", "variant_id", "Variation ID", "Variation_ID"],
        "AA_Position": ["AA_Position", "aa_position", "Position", "position"],
        "Ref_AA": ["Ref_AA", "ref_aa", "Reference_AA", "Reference"],
        "Alt_AA": ["Alt_AA", "alt_aa", "Alternate_AA", "Alternate"],
        "Binary_Label": ["Binary_Label", "binary_label", "Label", "label"],
        "ClinVar_Label": ["ClinVar_Label", "clinvar_label", "Clinical_Significance", "Classification"],
        "Mapped_Region": ["Mapped_Region", "mapped_region", "Region"],
        "Mapped_Functional_Category": [
            "Mapped_Functional_Category",
            "mapped_functional_category",
            "Functional_Category",
            "Functional category",
        ],
        "Mutation_Status": ["Mutation_Status", "mutation_status"],
    }

    normalized_existing = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for standard_name, possible_names in possible_mappings.items():
        for possible in possible_names:
            key = str(possible).strip().lower().replace(" ", "_")
            if key in normalized_existing:
                rename_map[normalized_existing[key]] = standard_name
                break

    df = df.rename(columns=rename_map)

    return df


def clean_master_table(df):
    df = normalize_column_names(df)

    missing_required = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_required:
        raise ValueError(
            f"Input file is missing required columns: {missing_required}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    # Standardize core fields
    df["Gene"] = df["Gene"].astype(str).str.strip().str.upper()
    df["AA_Position"] = pd.to_numeric(df["AA_Position"], errors="coerce")
    df["Ref_AA"] = df["Ref_AA"].astype(str).str.strip().str.upper()
    df["Alt_AA"] = df["Alt_AA"].astype(str).str.strip().str.upper()
    df["Binary_Label"] = pd.to_numeric(df["Binary_Label"], errors="coerce")

    # Optional columns
    if "Variant_ID" not in df.columns:
        df["Variant_ID"] = ""

    if "ClinVar_Label" not in df.columns:
        df["ClinVar_Label"] = ""

    if "Mapped_Region" not in df.columns:
        df["Mapped_Region"] = "Unknown"

    if "Mapped_Functional_Category" not in df.columns:
        df["Mapped_Functional_Category"] = "Unknown"

    if "Mutation_Status" not in df.columns:
        df["Mutation_Status"] = "Unknown"

    # Keep only SCN1A
    df = df[df["Gene"] == "SCN1A"].copy()

    # Keep rows with required fields
    missing_mask = (
        df["AA_Position"].isna()
        | df["Binary_Label"].isna()
        | df["Ref_AA"].isna()
        | df["Alt_AA"].isna()
        | (df["Ref_AA"].str.len() != 1)
        | (df["Alt_AA"].str.len() != 1)
    )

    missing_rows = df[missing_mask].copy()
    missing_rows.to_csv(MISSING_FILE, index=False)

    df = df[~missing_mask].copy()

    df["AA_Position"] = df["AA_Position"].astype(int)
    df["Binary_Label"] = df["Binary_Label"].astype(int)

    # Keep only binary 0/1 labels
    df = df[df["Binary_Label"].isin([0, 1])].copy()

    # Protein-level standardized variant notation
    df["Protein_Change_Parsed"] = (
        df["Ref_AA"].astype(str)
        + df["AA_Position"].astype(str)
        + df["Alt_AA"].astype(str)
    )

    df["Variant_Key"] = (
        df["Gene"].astype(str)
        + ":p."
        + df["Protein_Change_Parsed"].astype(str)
    )

    df["Label_Name"] = df["Binary_Label"].map({
        0: "Benign",
        1: "Pathogenic",
    })

    df["Dataset_Source"] = "SCN1A_ClinVar_gnomAD_training_set"

    # Clean strings
    for col in [
        "Variant_ID",
        "ClinVar_Label",
        "Mapped_Region",
        "Mapped_Functional_Category",
        "Mutation_Status",
    ]:
        df[col] = df[col].astype(str).str.strip()

    # Detect duplicate exact protein substitutions
    duplicate_mask = df.duplicated(
        subset=["Gene", "AA_Position", "Ref_AA", "Alt_AA"],
        keep=False,
    )

    duplicates = df[duplicate_mask].copy()
    duplicates.to_csv(DUPLICATES_FILE, index=False)

    # Deduplicate exact same protein substitution.
    # If duplicates have same label, keep one.
    # If duplicates conflict in label, keep first but report warning in QC.
    conflict_count = (
        duplicates
        .groupby(["Gene", "AA_Position", "Ref_AA", "Alt_AA"])["Binary_Label"]
        .nunique()
        .gt(1)
        .sum()
        if len(duplicates) > 0
        else 0
    )

    df = (
        df
        .sort_values(["Gene", "AA_Position", "Ref_AA", "Alt_AA"])
        .drop_duplicates(
            subset=["Gene", "AA_Position", "Ref_AA", "Alt_AA"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Final column order
    final_columns = [
        "Variant_Key",
        "Variant_ID",
        "Gene",
        "AA_Position",
        "Ref_AA",
        "Alt_AA",
        "Protein_Change_Parsed",
        "Binary_Label",
        "Label_Name",
        "ClinVar_Label",
        "Mapped_Region",
        "Mapped_Functional_Category",
        "Mutation_Status",
        "Dataset_Source",
    ]

    extra_columns = [col for col in df.columns if col not in final_columns]

    df = df[final_columns + extra_columns]

    qc = {
        "n_final_variants": int(len(df)),
        "n_missing_required_rows_written": int(len(missing_rows)),
        "n_duplicate_rows_written": int(len(duplicates)),
        "n_duplicate_label_conflicts": int(conflict_count),
        "class_counts": df["Label_Name"].value_counts().to_dict(),
        "functional_category_counts": df["Mapped_Functional_Category"].value_counts().to_dict(),
        "region_counts_top20": df["Mapped_Region"].value_counts().head(20).to_dict(),
        "aa_position_min": int(df["AA_Position"].min()) if len(df) else None,
        "aa_position_max": int(df["AA_Position"].max()) if len(df) else None,
        "unique_positions": int(df["AA_Position"].nunique()),
        "output_file": str(OUTPUT_FILE),
        "duplicates_file": str(DUPLICATES_FILE),
        "missing_file": str(MISSING_FILE),
    }

    return df, qc


def main():
    input_file = find_existing_file(INPUT_CANDIDATES)

    print("Reading input file:")
    print(input_file)

    df = pd.read_csv(input_file)

    print("\nInput rows:", len(df))
    print("Input columns:")
    print(list(df.columns))

    master, qc = clean_master_table(df)

    master.to_csv(OUTPUT_FILE, index=False)

    with open(QC_FILE, "w") as f:
        json.dump(qc, f, indent=4)

    print("\nSaved benchmark master table:")
    print(OUTPUT_FILE)

    print("\nSaved QC summary:")
    print(QC_FILE)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nPreview:")
    print(master.head(10).to_string(index=False))


if __name__ == "__main__":
    main()