from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "results" / "SCN1A_mutant_sequences.csv"
FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"

OUTPUT_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_AA_LENGTH = 1000


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


def get_window(sequence, position, max_len=1000):
    """
    position is 1-indexed.
    Returns a centered sequence window and mutation index inside that window.
    """
    pos0 = int(position) - 1

    half = max_len // 2
    start = max(0, pos0 - half)
    end = min(len(sequence), start + max_len)

    start = max(0, end - max_len)

    window = sequence[start:end]
    mutation_index = pos0 - start

    return window, mutation_index, start + 1, end


def embed_window(model, tokenizer, sequence, mutation_index, device):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]

    # +1 because token 0 is the special start token
    mutation_token_index = mutation_index + 1

    site_embedding = hidden[mutation_token_index].cpu().numpy()

    # Mean embedding over amino-acid tokens only
    mean_embedding = hidden[1:-1].mean(dim=0).cpu().numpy()

    return site_embedding, mean_embedding


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[df["Mutation_Status"] == "Success"].copy()
    df = df.dropna(subset=["Mutant_Sequence", "Binary_Label", "AA_Position"])

    print("Usable variants:", len(df))

    wildtype_sequence = read_fasta_sequence(FASTA_FILE)
    print("Wild-type length:", len(wildtype_sequence))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    wt_site_embeddings = []
    mut_site_embeddings = []
    wt_mean_embeddings = []
    mut_mean_embeddings = []
    metadata_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        position = int(row["AA_Position"])
        ref_aa = str(row["Ref_AA"])
        alt_aa = str(row["Alt_AA"])

        wt_window, wt_mut_idx, window_start, window_end = get_window(
            wildtype_sequence,
            position,
            MAX_AA_LENGTH
        )

        mut_window, mut_mut_idx, _, _ = get_window(
            row["Mutant_Sequence"],
            position,
            MAX_AA_LENGTH
        )

        wt_check = wt_window[wt_mut_idx]
        mut_check = mut_window[mut_mut_idx]

        if wt_check != ref_aa or mut_check != alt_aa:
            continue

        wt_site, wt_mean = embed_window(
            model,
            tokenizer,
            wt_window,
            wt_mut_idx,
            device
        )

        mut_site, mut_mean = embed_window(
            model,
            tokenizer,
            mut_window,
            mut_mut_idx,
            device
        )

        wt_site_embeddings.append(wt_site)
        mut_site_embeddings.append(mut_site)
        wt_mean_embeddings.append(wt_mean)
        mut_mean_embeddings.append(mut_mean)

        metadata_rows.append({
            "Variant_ID": row["Variant_ID"],
            "Gene": row["Gene"],
            "AA_Position": position,
            "Ref_AA": ref_aa,
            "Alt_AA": alt_aa,
            "ClinVar_Label": row["ClinVar_Label"],
            "Binary_Label": int(row["Binary_Label"]),
            "Mapped_Region": row["Mapped_Region"],
            "Mapped_Functional_Category": row["Mapped_Functional_Category"],
            "Window_Start": window_start,
            "Window_End": window_end,
            "WT_AA_Check": wt_check,
            "Mutant_AA_Check": mut_check
        })

    wt_site_embeddings = np.vstack(wt_site_embeddings)
    mut_site_embeddings = np.vstack(mut_site_embeddings)
    wt_mean_embeddings = np.vstack(wt_mean_embeddings)
    mut_mean_embeddings = np.vstack(mut_mean_embeddings)

    delta_site = mut_site_embeddings - wt_site_embeddings
    abs_delta_site = np.abs(delta_site)

    delta_mean = mut_mean_embeddings - wt_mean_embeddings
    abs_delta_mean = np.abs(delta_mean)

    site_features = np.concatenate(
        [wt_site_embeddings, mut_site_embeddings, delta_site, abs_delta_site],
        axis=1
    )

    mean_features = np.concatenate(
        [wt_mean_embeddings, mut_mean_embeddings, delta_mean, abs_delta_mean],
        axis=1
    )

    metadata = pd.DataFrame(metadata_rows)

    np.save(OUTPUT_DIR / "wt_site_embeddings.npy", wt_site_embeddings)
    np.save(OUTPUT_DIR / "mut_site_embeddings.npy", mut_site_embeddings)
    np.save(OUTPUT_DIR / "delta_site_embeddings.npy", delta_site)
    np.save(OUTPUT_DIR / "abs_delta_site_embeddings.npy", abs_delta_site)
    np.save(OUTPUT_DIR / "site_wt_mut_delta_abs_features.npy", site_features)

    np.save(OUTPUT_DIR / "wt_mean_embeddings.npy", wt_mean_embeddings)
    np.save(OUTPUT_DIR / "mut_mean_embeddings.npy", mut_mean_embeddings)
    np.save(OUTPUT_DIR / "delta_mean_embeddings.npy", delta_mean)
    np.save(OUTPUT_DIR / "abs_delta_mean_embeddings.npy", abs_delta_mean)
    np.save(OUTPUT_DIR / "mean_wt_mut_delta_abs_features.npy", mean_features)

    metadata.to_csv(OUTPUT_DIR / "paired_embedding_metadata.csv", index=False)

    print("\nSaved paired embeddings.")
    print("WT site:", wt_site_embeddings.shape)
    print("Mutant site:", mut_site_embeddings.shape)
    print("Delta site:", delta_site.shape)
    print("Combined site features:", site_features.shape)
    print("Combined mean features:", mean_features.shape)
    print("Metadata rows:", len(metadata))


if __name__ == "__main__":
    main()