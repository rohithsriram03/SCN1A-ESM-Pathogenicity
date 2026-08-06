from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "processed"
    / "scion_mechanism_variants_clean.csv"
)

FASTA_DIR = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "raw"
    / "scion"
    / "SCION"
    / "fasta"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "mechanism" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "scion_mechanism_variants_with_sequences.csv"
INVALID_FILE = OUTPUT_DIR / "scion_mechanism_sequence_invalid_rows.csv"
QC_FILE = OUTPUT_DIR / "scion_mechanism_sequence_qc.json"

WINDOW_SIZE = 1000
VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    sequence_lines = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                continue

            sequence_lines.append(line)

    return "".join(sequence_lines)


def make_centered_window(sequence, position_1indexed, window_size):
    pos0 = int(position_1indexed) - 1

    half = window_size // 2
    start = max(0, pos0 - half)
    end = min(len(sequence), start + window_size)

    if end - start < window_size:
        start = max(0, end - window_size)

    window = sequence[start:end]
    local_idx = pos0 - start

    return window, local_idx, start, end


def main():
    print("Reading SCION mechanism dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("\nRows:", len(df))
    print("Genes:", sorted(df["Gene"].unique()))

    fasta_sequences = {}

    for gene in sorted(df["Gene"].unique()):
        fasta_path = FASTA_DIR / f"{gene}.fasta"

        if not fasta_path.exists():
            raise FileNotFoundError(f"Missing FASTA for {gene}: {fasta_path}")

        seq = read_fasta(fasta_path)
        fasta_sequences[gene] = seq

        print(f"{gene}: length {len(seq)}")

    valid_records = []
    invalid_records = []

    for _, row in df.iterrows():
        gene = row["Gene"]
        pos = int(row["AA_Position"])
        ref = str(row["Ref_AA"]).strip().upper()
        alt = str(row["Alt_AA"]).strip().upper()

        sequence = fasta_sequences[gene]

        status = "Success"
        reason = ""

        try:
            if ref not in VALID_AAS or alt not in VALID_AAS:
                raise ValueError("Invalid amino acid symbol")

            if pos < 1 or pos > len(sequence):
                raise ValueError(
                    f"Position {pos} outside sequence length {len(sequence)}"
                )

            actual_ref = sequence[pos - 1]

            if actual_ref != ref:
                raise ValueError(
                    f"Reference mismatch: dataset has {ref}, FASTA has {actual_ref}"
                )

            wt_window, local_idx, window_start, window_end = make_centered_window(
                sequence,
                pos,
                WINDOW_SIZE,
            )

            mutant_window = (
                wt_window[:local_idx]
                + alt
                + wt_window[local_idx + 1:]
            )

            if wt_window[local_idx] != ref:
                raise ValueError("Window local reference mismatch")

            if mutant_window[local_idx] != alt:
                raise ValueError("Mutant window substitution failed")

            record = row.to_dict()
            record["Protein_Length"] = len(sequence)
            record["Actual_Ref_AA"] = actual_ref
            record["WT_Window"] = wt_window
            record["Mutant_Window"] = mutant_window
            record["Window_Start_0Indexed"] = window_start
            record["Window_End_0Indexed"] = window_end
            record["Local_Mutation_Index_0Indexed"] = local_idx
            record["Sequence_Status"] = status

            valid_records.append(record)

        except Exception as e:
            bad = row.to_dict()
            bad["Sequence_Status"] = "Invalid"
            bad["Invalid_Reason"] = str(e)
            invalid_records.append(bad)

    valid_df = pd.DataFrame(valid_records)
    invalid_df = pd.DataFrame(invalid_records)

    valid_df.to_csv(OUTPUT_FILE, index=False)
    invalid_df.to_csv(INVALID_FILE, index=False)

    qc = {
        "input_rows": int(len(df)),
        "valid_rows": int(len(valid_df)),
        "invalid_rows": int(len(invalid_df)),
        "window_size": WINDOW_SIZE,
        "mechanism_label_counts_valid": valid_df["Mechanism_Label"].value_counts().to_dict()
        if len(valid_df) > 0
        else {},
        "gene_counts_valid": valid_df["Gene"].value_counts().to_dict()
        if len(valid_df) > 0
        else {},
        "target_epilepsy_scn_counts_valid": valid_df[
            valid_df["Gene"].isin(["SCN1A", "SCN2A", "SCN3A", "SCN8A"])
        ]["Gene"].value_counts().to_dict()
        if len(valid_df) > 0
        else {},
        "invalid_reasons": invalid_df["Invalid_Reason"].value_counts().to_dict()
        if len(invalid_df) > 0
        else {},
        "output_file": str(OUTPUT_FILE),
        "invalid_file": str(INVALID_FILE),
    }

    with open(QC_FILE, "w") as f:
        json.dump(qc, f, indent=4)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nPreview:")
    preview_cols = [
        "Variant_Key",
        "Gene",
        "Protein_Change",
        "AA_Position",
        "Ref_AA",
        "Alt_AA",
        "Mechanism_Label",
        "Protein_Length",
        "Local_Mutation_Index_0Indexed",
        "Sequence_Status",
    ]

    print(valid_df[preview_cols].head(20).to_string(index=False))

    print("\nSaved validated sequence dataset:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()