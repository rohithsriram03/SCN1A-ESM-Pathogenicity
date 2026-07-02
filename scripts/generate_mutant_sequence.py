from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "results" / "SCN1A_variant_master_domain_mapped.xlsx"
FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"
OUTPUT_FILE = PROJECT_ROOT / "results" / "SCN1A_mutant_sequences.csv"


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


def make_mutant_sequence(wildtype_sequence, position, ref_aa, alt_aa):
    # ClinVar/UniProt positions are 1-indexed.
    # Python strings are 0-indexed.
    index = int(position) - 1

    actual_ref = wildtype_sequence[index]

    if actual_ref != ref_aa:
        return None, actual_ref, "Ref_AA_mismatch"

    mutant_list = list(wildtype_sequence)
    mutant_list[index] = alt_aa
    mutant_sequence = "".join(mutant_list)

    return mutant_sequence, actual_ref, "Success"


def main():
    wildtype_sequence = read_fasta_sequence(FASTA_FILE)

    print("Wild-type sequence length:", len(wildtype_sequence))

    variants = pd.read_excel(INPUT_FILE, sheet_name="Variant_Domain_Map")

    output_rows = []

    for _, row in variants.iterrows():
        variant_id = row["Variant_ID"]
        position = row["AA_Position"]
        ref_aa = row["Ref_AA"]
        alt_aa = row["Alt_AA"]

        if pd.isna(position) or pd.isna(ref_aa) or pd.isna(alt_aa):
            mutant_sequence = None
            actual_ref = None
            status = "Missing_variant_information"
        else:
            mutant_sequence, actual_ref, status = make_mutant_sequence(
                wildtype_sequence=wildtype_sequence,
                position=position,
                ref_aa=str(ref_aa),
                alt_aa=str(alt_aa),
            )

        output_rows.append({
            "Gene": row.get("Gene"),
            "Variant_ID": variant_id,
            "AA_Position": position,
            "Ref_AA": ref_aa,
            "Alt_AA": alt_aa,
            "Actual_WT_AA_At_Position": actual_ref,
            "Mutation_Status": status,
            "ClinVar_Label": row.get("ClinVar_Label"),
            "Binary_Label": row.get("Binary_Label"),
            "Mapped_Region": row.get("Mapped_Region"),
            "Mapped_Functional_Category": row.get("Mapped_Functional_Category"),
            "Mutant_Sequence": mutant_sequence,
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("\nMutation status counts:")
    print(output_df["Mutation_Status"].value_counts())


if __name__ == "__main__":
    main()