from pathlib import Path
import json
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_DIR = PROJECT_ROOT / "results" / "benchmarking"
ALPHAMISSENSE_DIR = PROJECT_ROOT / "data" / "alphamissense"

CLEAN_BENCHMARK_FILE = BENCHMARK_DIR / "scn1a_benchmark_master_table_clean.csv"
FALLBACK_BENCHMARK_FILE = BENCHMARK_DIR / "scn1a_benchmark_master_table.csv"

OUTPUT_FILE = BENCHMARK_DIR / "scn1a_benchmark_with_alphamissense.csv"
UNMATCHED_FILE = BENCHMARK_DIR / "scn1a_variants_unmatched_to_alphamissense.csv"
QC_FILE = BENCHMARK_DIR / "alphamissense_merge_qc.json"


def find_benchmark_file():
    if CLEAN_BENCHMARK_FILE.exists():
        return CLEAN_BENCHMARK_FILE

    print("WARNING: Clean benchmark table not found. Using fallback master table.")
    return FALLBACK_BENCHMARK_FILE


def find_alphamissense_file():
    if not ALPHAMISSENSE_DIR.exists():
        raise FileNotFoundError(
            f"AlphaMissense directory does not exist: {ALPHAMISSENSE_DIR}\n"
            "Create it and place the P35498 AlphaMissense TSV file there."
        )

    candidates = list(ALPHAMISSENSE_DIR.glob("*.tsv")) + list(ALPHAMISSENSE_DIR.glob("*.tsv.gz"))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No AlphaMissense .tsv or .tsv.gz files found in {ALPHAMISSENSE_DIR}"
        )

    if len(candidates) > 1:
        print("Multiple AlphaMissense files found. Using the first one:")
        for path in candidates:
            print(" -", path)

    return candidates[0]


def normalize_columns(df):
    rename_map = {}

    possible = {
        "uniprot_acc": [
            "Uniprot ACC",
            "UniProt ACC",
            "uniprot_acc",
            "uniprot_id",
            "uniprot",
            "protein_id",
        ],
        "gene_name": [
            "Gene name",
            "Gene Name",
            "gene_name",
            "Gene",
            "gene",
        ],
        "protein_variant": [
            "protein variant",
            "Protein variant",
            "protein_variant",
            "Protein_Change",
            "variant",
        ],
        "ref_aa": [
            "a.a.1",
            "aa1",
            "ref_aa",
            "reference_aa",
        ],
        "position": [
            "position",
            "Position",
            "protein_position",
            "aa_position",
        ],
        "alt_aa": [
            "a.a.2",
            "aa2",
            "alt_aa",
            "alternate_aa",
        ],
        "am_pathogenicity": [
            "pathogenicity score",
            "Pathogenicity score",
            "am_pathogenicity",
            "pathogenicity_score",
            "score",
        ],
        "am_class": [
            "pathogenicity class",
            "Pathogenicity class",
            "am_class",
            "pathogenicity_class",
            "class",
        ],
    }

    normalized_existing = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in df.columns
    }

    for standard, names in possible.items():
        for name in names:
            key = str(name).strip().lower().replace(" ", "_")
            if key in normalized_existing:
                rename_map[normalized_existing[key]] = standard
                break

    return df.rename(columns=rename_map)


def parse_protein_variant(value):
    """
    Parses variants like:
    M1T
    p.M1T
    A123V

    Returns ref, position, alt.
    """
    text = str(value).strip()

    match = re.search(r"([A-Z])(\d+)([A-Z])", text)

    if not match:
        return None, None, None

    ref = match.group(1)
    pos = int(match.group(2))
    alt = match.group(3)

    return ref, pos, alt


def prepare_alphamissense(am):
    am = normalize_columns(am)

    print("\nAlphaMissense columns after normalization:")
    print(list(am.columns))

    # If ref/position/alt columns are missing, parse from protein_variant.
    if not {"ref_aa", "position", "alt_aa"}.issubset(am.columns):
        if "protein_variant" not in am.columns:
            raise ValueError(
                "Could not find ref/position/alt columns or protein_variant column "
                f"in AlphaMissense file. Columns: {list(am.columns)}"
            )

        parsed = am["protein_variant"].apply(parse_protein_variant)
        am["ref_aa"] = parsed.apply(lambda x: x[0])
        am["position"] = parsed.apply(lambda x: x[1])
        am["alt_aa"] = parsed.apply(lambda x: x[2])

    required = ["ref_aa", "position", "alt_aa", "am_pathogenicity"]

    missing = [col for col in required if col not in am.columns]

    if missing:
        raise ValueError(
            f"AlphaMissense file missing required columns: {missing}\n"
            f"Available columns: {list(am.columns)}"
        )

    am = am.copy()

    am["ref_aa"] = am["ref_aa"].astype(str).str.strip().str.upper()
    am["alt_aa"] = am["alt_aa"].astype(str).str.strip().str.upper()
    am["position"] = pd.to_numeric(am["position"], errors="coerce")
    am["am_pathogenicity"] = pd.to_numeric(am["am_pathogenicity"], errors="coerce")

    am = am.dropna(subset=["ref_aa", "position", "alt_aa", "am_pathogenicity"]).copy()

    am["position"] = am["position"].astype(int)

    # Keep only valid single amino acid substitutions.
    am = am[
        (am["ref_aa"].str.len() == 1)
        & (am["alt_aa"].str.len() == 1)
    ].copy()

    am["AlphaMissense_Key"] = (
        am["ref_aa"].astype(str)
        + am["position"].astype(str)
        + am["alt_aa"].astype(str)
    )

    keep_cols = [
        "AlphaMissense_Key",
        "am_pathogenicity",
    ]

    if "am_class" in am.columns:
        keep_cols.append("am_class")

    if "protein_variant" in am.columns:
        keep_cols.append("protein_variant")

    if "uniprot_acc" in am.columns:
        keep_cols.append("uniprot_acc")

    if "gene_name" in am.columns:
        keep_cols.append("gene_name")

    am = am[keep_cols].drop_duplicates(subset=["AlphaMissense_Key"], keep="first")

    return am


def main():
    benchmark_file = find_benchmark_file()
    alphamissense_file = find_alphamissense_file()

    print("Reading benchmark:")
    print(benchmark_file)

    benchmark = pd.read_csv(benchmark_file)

    print("\nReading AlphaMissense:")
    print(alphamissense_file)

    am = pd.read_csv(alphamissense_file, sep="\t", low_memory=False)

    am_clean = prepare_alphamissense(am)

    benchmark = benchmark.copy()

    benchmark["AlphaMissense_Key"] = (
        benchmark["Ref_AA"].astype(str).str.upper()
        + benchmark["AA_Position"].astype(int).astype(str)
        + benchmark["Alt_AA"].astype(str).str.upper()
    )

    merged = benchmark.merge(
        am_clean,
        on="AlphaMissense_Key",
        how="left",
    )

    merged = merged.rename(columns={
        "am_pathogenicity": "AlphaMissense_Score",
        "am_class": "AlphaMissense_Class",
        "protein_variant": "AlphaMissense_Protein_Variant",
    })

    unmatched = merged[merged["AlphaMissense_Score"].isna()].copy()
    unmatched.to_csv(UNMATCHED_FILE, index=False)

    merged.to_csv(OUTPUT_FILE, index=False)

    qc = {
        "benchmark_file": str(benchmark_file),
        "alphamissense_file": str(alphamissense_file),
        "n_benchmark_variants": int(len(benchmark)),
        "n_alphamissense_rows_loaded": int(len(am)),
        "n_alphamissense_unique_substitutions": int(len(am_clean)),
        "n_matched": int(merged["AlphaMissense_Score"].notna().sum()),
        "n_unmatched": int(merged["AlphaMissense_Score"].isna().sum()),
        "match_rate": float(merged["AlphaMissense_Score"].notna().mean()),
        "score_summary": {
            "mean": float(merged["AlphaMissense_Score"].mean()),
            "median": float(merged["AlphaMissense_Score"].median()),
            "min": float(merged["AlphaMissense_Score"].min()),
            "max": float(merged["AlphaMissense_Score"].max()),
        },
        "alphamissense_class_counts": (
            merged["AlphaMissense_Class"].value_counts(dropna=False).to_dict()
            if "AlphaMissense_Class" in merged.columns
            else {}
        ),
        "class_label_score_means": (
            merged.groupby("Label_Name")["AlphaMissense_Score"].mean().to_dict()
            if "Label_Name" in merged.columns
            else {}
        ),
        "output_file": str(OUTPUT_FILE),
        "unmatched_file": str(UNMATCHED_FILE),
    }

    with open(QC_FILE, "w") as f:
        json.dump(qc, f, indent=4)

    print("\nSaved merged benchmark:")
    print(OUTPUT_FILE)

    print("\nSaved unmatched variants:")
    print(UNMATCHED_FILE)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nPreview:")
    preview_cols = [
        "Variant_Key",
        "Protein_Change_Parsed",
        "Binary_Label",
        "Label_Name",
        "Mapped_Functional_Category",
        "AlphaMissense_Score",
    ]

    if "AlphaMissense_Class" in merged.columns:
        preview_cols.append("AlphaMissense_Class")

    print(merged[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()