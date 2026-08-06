from pathlib import Path
import json

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCION_DIR = PROJECT_ROOT / "data" / "mechanism" / "raw" / "scion" / "SCION"

CLEAN_TBL = SCION_DIR / "data" / "clean_tbl.csv"
CID_FILE = SCION_DIR / "app" / "cid.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "mechanism" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "scion_mechanism_variants_clean.csv"
QC_FILE = OUTPUT_DIR / "scion_mechanism_variants_qc.json"


UNIPROT_IDS = {
    "SCN1A": "P35498",
    "SCN2A": "Q99250",
    "SCN3A": "Q9NY46",
    "SCN4A": "P35499",
    "SCN5A": "Q14524",
    "SCN8A": "Q9UQD0",
    "SCN9A": "Q15858",
    "SCN10A": "Q9Y5Y9",
    "SCN11A": "Q9UI33",
}


TARGET_GENES = [
    "SCN1A",
    "SCN2A",
    "SCN3A",
    "SCN8A",
]


def find_family_alignment_cid(row, cid_table):
    gene = row["Gene"]
    pos = int(row["AA_Position"])

    if gene not in cid_table.columns:
        return None

    matches = cid_table[cid_table[gene] == pos]

    if len(matches) == 0:
        return None

    return int(matches.iloc[0]["cid"])


def get_analog_position(row, cid_table, analog_gene):
    cid = row["Family_Alignment_CID"]

    if pd.isna(cid):
        return None

    matches = cid_table[cid_table["cid"] == int(cid)]

    if len(matches) == 0:
        return None

    value = matches.iloc[0].get(analog_gene, None)

    if pd.isna(value):
        return None

    value = int(value)

    if value == 0:
        return None

    return value


def main():
    print("Reading SCION clean table:")
    print(CLEAN_TBL)

    df = pd.read_csv(CLEAN_TBL)

    print("Reading SCION alignment table:")
    print(CID_FILE)

    cid = pd.read_csv(CID_FILE)

    clean = df.copy()

    clean = clean.rename(
        columns={
            "gene": "Gene",
            "y": "Mechanism_Label",
            "pheno": "Phenotype",
            "aa1": "Ref_AA",
            "aa2": "Alt_AA",
            "pos": "AA_Position",
            "id": "SCION_ID",
        }
    )

    clean["Gene"] = clean["Gene"].astype(str).str.upper()
    clean["Ref_AA"] = clean["Ref_AA"].astype(str).str.upper()
    clean["Alt_AA"] = clean["Alt_AA"].astype(str).str.upper()
    clean["Mechanism_Label"] = clean["Mechanism_Label"].astype(str).str.upper()
    clean["AA_Position"] = clean["AA_Position"].astype(int)

    clean["Protein_Change"] = (
        clean["Ref_AA"]
        + clean["AA_Position"].astype(str)
        + clean["Alt_AA"]
    )

    clean["Variant_Key"] = clean["Gene"] + ":p." + clean["Protein_Change"]

    clean["UniProt_ID"] = clean["Gene"].map(UNIPROT_IDS)

    clean["Mechanism_Binary"] = clean["Mechanism_Label"].map(
        {
            "LOF": 0,
            "GOF": 1,
        }
    )

    clean["Source_Dataset"] = "SCION"
    clean["Source_Label_Type"] = "GOF_vs_LOF"
    clean["Treatment_Implication_Rough"] = clean.apply(
        lambda row: (
            "possible_sodium_channel_blocker_relevance"
            if row["Gene"] == "SCN1A" and row["Mechanism_Label"] == "GOF"
            else "mechanism_context_needed"
        ),
        axis=1,
    )

    # Add family alignment coordinate.
    clean["Family_Alignment_CID"] = clean.apply(
        lambda row: find_family_alignment_cid(row, cid),
        axis=1,
    )

    # Add analogous positions across major epilepsy sodium-channel genes.
    for analog_gene in TARGET_GENES:
        clean[f"Analog_{analog_gene}_Position"] = clean.apply(
            lambda row: get_analog_position(row, cid, analog_gene),
            axis=1,
        )

    # Add whether this row belongs to the main epilepsy sodium-channel set.
    clean["Is_Target_Epilepsy_SCN"] = clean["Gene"].isin(TARGET_GENES).astype(int)

    # Simple duplicate/conflict checks.
    duplicate_variants = clean[clean.duplicated(subset=["Variant_Key"], keep=False)].copy()

    label_conflicts = (
        clean.groupby("Variant_Key")["Mechanism_Label"]
        .nunique()
        .reset_index()
    )
    label_conflicts = label_conflicts[label_conflicts["Mechanism_Label"] > 1]

    ordered_cols = [
        "Variant_Key",
        "Gene",
        "UniProt_ID",
        "Protein_Change",
        "AA_Position",
        "Ref_AA",
        "Alt_AA",
        "Mechanism_Label",
        "Mechanism_Binary",
        "Phenotype",
        "SCION_ID",
        "Family_Alignment_CID",
        "Analog_SCN1A_Position",
        "Analog_SCN2A_Position",
        "Analog_SCN3A_Position",
        "Analog_SCN8A_Position",
        "Is_Target_Epilepsy_SCN",
        "Treatment_Implication_Rough",
        "Source_Dataset",
        "Source_Label_Type",
    ]

    clean = clean[ordered_cols]

    clean.to_csv(OUTPUT_FILE, index=False)

    qc = {
        "n_total_variants": int(len(clean)),
        "mechanism_label_counts": clean["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": clean["Gene"].value_counts().to_dict(),
        "target_epilepsy_scn_gene_counts": (
            clean[clean["Gene"].isin(TARGET_GENES)]["Gene"].value_counts().to_dict()
        ),
        "n_target_epilepsy_scn_variants": int(clean["Is_Target_Epilepsy_SCN"].sum()),
        "n_missing_uniprot": int(clean["UniProt_ID"].isna().sum()),
        "n_missing_family_alignment_cid": int(clean["Family_Alignment_CID"].isna().sum()),
        "n_duplicate_variant_rows": int(len(duplicate_variants)),
        "n_label_conflict_variants": int(len(label_conflicts)),
        "output_file": str(OUTPUT_FILE),
    }

    with open(QC_FILE, "w") as f:
        json.dump(qc, f, indent=4)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nPreview:")
    print(clean.head(20).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()