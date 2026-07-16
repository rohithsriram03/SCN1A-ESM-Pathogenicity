from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" 
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "vus_prioritization"
OUTPUT_DIR.mkdir(exist_ok=True)

FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"

# Put your ClinVar VUS export here.
INPUT_FILE_CANDIDATES = [
    RAW_DIR / "scn1a_clinvar_vus_export.csv",
    RAW_DIR / "scn1a_clinvar_vus_export.tsv",
    RAW_DIR / "scn1a_clinvar_vus_export.xlsx",
]

DOMAIN_BOUNDARY_CANDIDATES = [
    RESULTS_DIR / "SCN1A_variant_master_domain_mapped.xlsx",
    PROJECT_ROOT / "data" / "SCN1A_variant_master_domain_mapped.xlsx",
]


AA3_TO_AA1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
}


def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find input file. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def read_table(path):
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


def read_fasta_sequence(fasta_path):
    sequence_lines = []

    with open(fasta_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                continue
            sequence_lines.append(line)

    return "".join(sequence_lines)


def normalize_colname(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_column(df, possible_names):
    normalized = {normalize_colname(col): col for col in df.columns}

    for name in possible_names:
        key = normalize_colname(name)
        if key in normalized:
            return normalized[key]

    # Fallback: partial match
    for col in df.columns:
        col_norm = normalize_colname(col)
        for name in possible_names:
            name_norm = normalize_colname(name)
            if name_norm in col_norm or col_norm in name_norm:
                return col

    return None


def get_text_from_possible_columns(row, columns):
    values = []

    for col in columns:
        if col is None:
            continue
        if col in row and pd.notna(row[col]):
            values.append(str(row[col]))

    return " | ".join(values)


def extract_protein_change(text):
    """
    Supports examples like:
    p.Arg1648His
    p.R1648H
    Arg1648His
    R1648H

    Excludes obvious non-missense events like Ter, fs, del, dup.
    """

    if not isinstance(text, str):
        return None

    bad_tokens = ["Ter", "*", "fs", "del", "dup", "ins", "ext", "?"]
    if any(token in text for token in bad_tokens):
        return None

    # 3-letter HGVS protein format
    pattern_3 = re.compile(
        r"p\.?\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?"
    )

    match = pattern_3.search(text)
    if match:
        ref3, pos, alt3 = match.groups()
        ref = AA3_TO_AA1.get(ref3)
        alt = AA3_TO_AA1.get(alt3)

        if ref and alt:
            return {
                "AA_Position": int(pos),
                "Ref_AA": ref,
                "Alt_AA": alt,
                "Protein_Change_Parsed": f"{ref}{pos}{alt}",
            }

    # 1-letter format
    pattern_1 = re.compile(r"\b([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])\b")

    match = pattern_1.search(text)
    if match:
        ref, pos, alt = match.groups()
        return {
            "AA_Position": int(pos),
            "Ref_AA": ref,
            "Alt_AA": alt,
            "Protein_Change_Parsed": f"{ref}{pos}{alt}",
        }

    return None


def load_domain_boundaries():
    boundary_file = find_existing_file(DOMAIN_BOUNDARY_CANDIDATES)

    xls = pd.ExcelFile(boundary_file)

    sheet_name = None
    for candidate in ["Domain_Boundaries", "domain_boundaries", "Domain Boundaries"]:
        if candidate in xls.sheet_names:
            sheet_name = candidate
            break

    if sheet_name is None:
        raise ValueError(
            f"Could not find Domain_Boundaries sheet in {boundary_file}. "
            f"Available sheets: {xls.sheet_names}"
        )

    boundaries = pd.read_excel(boundary_file, sheet_name=sheet_name)

    required = ["Gene", "Region", "Start", "End", "Functional_Category"]
    missing = [col for col in required if col not in boundaries.columns]

    if missing:
        raise ValueError(f"Domain boundary table missing columns: {missing}")

    # Clean boundary table
    boundaries = boundaries.copy()

    boundaries["Gene"] = boundaries["Gene"].astype(str).str.strip().str.upper()
    boundaries["Region"] = boundaries["Region"].astype(str).str.strip()
    boundaries["Functional_Category"] = boundaries["Functional_Category"].astype(str).str.strip()

    boundaries["Start"] = pd.to_numeric(boundaries["Start"], errors="coerce")
    boundaries["End"] = pd.to_numeric(boundaries["End"], errors="coerce")

    before = len(boundaries)

    boundaries = boundaries.dropna(
        subset=["Gene", "Region", "Start", "End", "Functional_Category"]
    )

    boundaries = boundaries[
        (boundaries["Gene"] == "SCN1A")
        & (boundaries["Start"].notna())
        & (boundaries["End"].notna())
    ]

    boundaries["Start"] = boundaries["Start"].astype(int)
    boundaries["End"] = boundaries["End"].astype(int)

    boundaries = boundaries.sort_values(["Start", "End"]).reset_index(drop=True)

    after = len(boundaries)

    print(f"Cleaned domain boundaries: kept {after} of {before} rows")

    if after == 0:
        raise ValueError("No usable SCN1A domain boundary rows found after cleaning.")

    return boundaries


def map_domain(position, boundaries):
    position = int(position)

    sub = boundaries[
        (boundaries["Gene"] == "SCN1A")
        & (boundaries["Start"] <= position)
        & (boundaries["End"] >= position)
    ]

    if len(sub) == 0:
        return "Other", "Other"

    row = sub.iloc[0]

    return row["Region"], row["Functional_Category"]


def make_mutant_sequence(wildtype_sequence, position, ref_aa, alt_aa):
    pos0 = int(position) - 1

    if pos0 < 0 or pos0 >= len(wildtype_sequence):
        return None, "Position_out_of_range", None

    actual_ref = wildtype_sequence[pos0]

    if actual_ref != ref_aa:
        return None, "Ref_AA_mismatch", actual_ref

    seq_list = list(wildtype_sequence)
    seq_list[pos0] = alt_aa

    return "".join(seq_list), "Success", actual_ref


def main():
    input_file = find_existing_file(INPUT_FILE_CANDIDATES)

    print("Reading ClinVar VUS export:", input_file)

    raw = read_table(input_file)

    print("Raw rows:", len(raw))
    print("Columns:")
    print(list(raw.columns))

    wildtype_sequence = read_fasta_sequence(FASTA_FILE)
    print("Wild-type sequence length:", len(wildtype_sequence))

    boundaries = load_domain_boundaries()
    print("Loaded domain boundaries:", len(boundaries))

    gene_col = find_column(
        raw,
        ["Gene(s)", "Gene", "Genes", "Symbol"]
    )

    clinical_col = find_column(
        raw,
        [
            "Clinical significance",
            "Germline classification",
            "Germline clinical significance",
            "ClinicalSignificance",
            "Significance",
        ],
    )

    name_col = find_column(
        raw,
        [
            "Name",
            "Variation name",
            "HGVS",
            "Protein change",
            "Protein Change",
            "Molecular consequence",
        ],
    )

    variation_id_col = find_column(
        raw,
        [
            "Variation ID",
            "VariationID",
            "AlleleID",
            "VCV accession",
            "RCVaccession",
            "Accession",
        ],
    )

    protein_col = find_column(
        raw,
        [
            "Protein change",
            "Protein Change",
            "HGVS protein",
            "HGVS Protein",
            "HGVS.p",
            "Name",
        ],
    )

    print("\nDetected columns:")
    print("Gene column:", gene_col)
    print("Clinical significance column:", clinical_col)
    print("Name column:", name_col)
    print("Protein column:", protein_col)
    print("Variation ID column:", variation_id_col)

    candidate_rows = []

    for idx, row in raw.iterrows():
        gene_text = str(row[gene_col]) if gene_col and pd.notna(row[gene_col]) else ""

        if gene_col is not None and "SCN1A" not in gene_text.upper():
            continue

        clinical_text = (
            str(row[clinical_col])
            if clinical_col is not None and pd.notna(row[clinical_col])
            else ""
        )

        clinical_lower = clinical_text.lower()

        is_vus_like = (
            "uncertain" in clinical_lower
            or "conflicting" in clinical_lower
            or "vus" in clinical_lower
        )

        if not is_vus_like:
            continue

        text_for_parsing = get_text_from_possible_columns(
            row,
            [protein_col, name_col]
        )

        parsed = extract_protein_change(text_for_parsing)

        if parsed is None:
            continue

        position = parsed["AA_Position"]
        ref_aa = parsed["Ref_AA"]
        alt_aa = parsed["Alt_AA"]

        mutant_sequence, status, actual_ref = make_mutant_sequence(
            wildtype_sequence=wildtype_sequence,
            position=position,
            ref_aa=ref_aa,
            alt_aa=alt_aa,
        )

        if status != "Success":
            continue

        mapped_region, functional_category = map_domain(position, boundaries)

        variant_id = (
            str(row[variation_id_col])
            if variation_id_col is not None and pd.notna(row[variation_id_col])
            else f"VUS_{position}_{ref_aa}_{alt_aa}_{idx}"
        )

        candidate_rows.append({
            "Variant_ID": variant_id,
            "Gene": "SCN1A",
            "AA_Position": position,
            "Ref_AA": ref_aa,
            "Alt_AA": alt_aa,
            "Protein_Change_Parsed": parsed["Protein_Change_Parsed"],
            "Clinical_Significance": clinical_text,
            "Mapped_Region": mapped_region,
            "Mapped_Functional_Category": functional_category,
            "Actual_WT_AA_At_Position": actual_ref,
            "Mutation_Status": status,
            "Mutant_Sequence": mutant_sequence,
            "Source_Row_Index": idx,
            "Source_Text": text_for_parsing,
        })

    vus = pd.DataFrame(candidate_rows)

    if len(vus) == 0:
        print("\nNo VUS-like missense candidates were parsed.")
        print("Check whether your ClinVar export includes protein-change HGVS strings like p.Arg1648His.")
        return

    # Deduplicate exact protein substitutions
    vus = vus.drop_duplicates(
        subset=["AA_Position", "Ref_AA", "Alt_AA"]
    ).reset_index(drop=True)

    output_path = OUTPUT_DIR / "vus_candidates_sequences.csv"
    vus.to_csv(output_path, index=False)

    print("\nPrepared VUS candidates:", len(vus))

    print("\nClinical significance counts:")
    print(vus["Clinical_Significance"].value_counts())

    print("\nFunctional category counts:")
    print(vus["Mapped_Functional_Category"].value_counts())

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()