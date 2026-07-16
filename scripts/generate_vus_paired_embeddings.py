from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VUS_DIR = PROJECT_ROOT / "results" / "vus_prioritization"
VUS_FILE = VUS_DIR / "vus_candidates_sequences.csv"

FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"

OUTPUT_DIR = VUS_DIR / "vus_paired_embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
WINDOW_SIZE = 1000


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def get_mutation_centered_window(sequence, aa_position, window_size=1000):
    """
    aa_position is 1-indexed.
    Returns:
    - sequence window
    - 0-indexed mutation position inside the window
    """

    sequence_length = len(sequence)
    pos0 = int(aa_position) - 1

    half = window_size // 2

    start = max(0, pos0 - half)
    end = min(sequence_length, start + window_size)

    # If near the C-terminus, shift window left to preserve window size.
    start = max(0, end - window_size)

    mutation_index_in_window = pos0 - start

    return sequence[start:end], mutation_index_in_window, start, end


def embed_sequence_window(sequence_window, mutation_index_in_window, tokenizer, model, device):
    inputs = tokenizer(
        sequence_window,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=1022,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]

    # ESM tokenization includes a special beginning token.
    # Residue i in the raw sequence corresponds to token i + 1.
    residue_token_index = mutation_index_in_window + 1

    site_embedding = hidden[residue_token_index].detach().cpu().numpy()

    # Remove special tokens for mean residue embedding.
    residue_embeddings = hidden[1:-1]
    mean_embedding = residue_embeddings.mean(dim=0).detach().cpu().numpy()

    return site_embedding, mean_embedding


def main():
    device = get_device()
    print("Using device:", device)

    vus = pd.read_csv(VUS_FILE)
    wildtype_sequence = read_fasta_sequence(FASTA_FILE)

    print("Loaded VUS candidates:", len(vus))
    print("Wild-type sequence length:", len(wildtype_sequence))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    wt_site_embeddings = []
    mut_site_embeddings = []
    wt_mean_embeddings = []
    mut_mean_embeddings = []

    metadata_rows = []

    skipped = []

    for _, row in tqdm(vus.iterrows(), total=len(vus)):
        aa_position = int(row["AA_Position"])
        ref_aa = str(row["Ref_AA"]).strip().upper()
        alt_aa = str(row["Alt_AA"]).strip().upper()
        mutant_sequence = str(row["Mutant_Sequence"])

        wt_window, mut_index, start, end = get_mutation_centered_window(
            wildtype_sequence,
            aa_position,
            WINDOW_SIZE,
        )

        mut_window, mut_index_2, start_2, end_2 = get_mutation_centered_window(
            mutant_sequence,
            aa_position,
            WINDOW_SIZE,
        )

        if mut_index != mut_index_2 or start != start_2 or end != end_2:
            skipped.append({
                "Variant_ID": row["Variant_ID"],
                "Reason": "Window mismatch",
            })
            continue

        actual_wt = wt_window[mut_index]
        actual_mut = mut_window[mut_index]

        if actual_wt != ref_aa:
            skipped.append({
                "Variant_ID": row["Variant_ID"],
                "Reason": f"WT mismatch: expected {ref_aa}, found {actual_wt}",
            })
            continue

        if actual_mut != alt_aa:
            skipped.append({
                "Variant_ID": row["Variant_ID"],
                "Reason": f"Mutant mismatch: expected {alt_aa}, found {actual_mut}",
            })
            continue

        wt_site, wt_mean = embed_sequence_window(
            wt_window,
            mut_index,
            tokenizer,
            model,
            device,
        )

        mut_site, mut_mean = embed_sequence_window(
            mut_window,
            mut_index,
            tokenizer,
            model,
            device,
        )

        wt_site_embeddings.append(wt_site)
        mut_site_embeddings.append(mut_site)
        wt_mean_embeddings.append(wt_mean)
        mut_mean_embeddings.append(mut_mean)

        metadata_row = row.to_dict()
        metadata_row["Window_Start_1Indexed"] = start + 1
        metadata_row["Window_End_1Indexed"] = end
        metadata_row["Mutation_Index_In_Window_0Indexed"] = mut_index
        metadata_rows.append(metadata_row)

    wt_site_embeddings = np.array(wt_site_embeddings)
    mut_site_embeddings = np.array(mut_site_embeddings)
    wt_mean_embeddings = np.array(wt_mean_embeddings)
    mut_mean_embeddings = np.array(mut_mean_embeddings)

    delta_site_embeddings = mut_site_embeddings - wt_site_embeddings
    abs_delta_site_embeddings = np.abs(delta_site_embeddings)

    paired_site_features = np.concatenate(
        [wt_site_embeddings, mut_site_embeddings],
        axis=1,
    )

    full_site_features = np.concatenate(
        [
            wt_site_embeddings,
            mut_site_embeddings,
            delta_site_embeddings,
            abs_delta_site_embeddings,
        ],
        axis=1,
    )

    metadata = pd.DataFrame(metadata_rows)

    np.save(OUTPUT_DIR / "vus_wt_site_embeddings.npy", wt_site_embeddings)
    np.save(OUTPUT_DIR / "vus_mut_site_embeddings.npy", mut_site_embeddings)
    np.save(OUTPUT_DIR / "vus_delta_site_embeddings.npy", delta_site_embeddings)
    np.save(OUTPUT_DIR / "vus_abs_delta_site_embeddings.npy", abs_delta_site_embeddings)
    np.save(OUTPUT_DIR / "vus_wt_plus_mut_site_features.npy", paired_site_features)
    np.save(OUTPUT_DIR / "vus_full_site_features.npy", full_site_features)

    np.save(OUTPUT_DIR / "vus_wt_mean_embeddings.npy", wt_mean_embeddings)
    np.save(OUTPUT_DIR / "vus_mut_mean_embeddings.npy", mut_mean_embeddings)

    metadata.to_csv(OUTPUT_DIR / "vus_embedding_metadata.csv", index=False)

    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv(OUTPUT_DIR / "vus_embedding_skipped_variants.csv", index=False)

    print("\nSaved VUS paired embeddings.")
    print("VUS successfully embedded:", len(metadata))
    print("Skipped:", len(skipped))
    print("WT site shape:", wt_site_embeddings.shape)
    print("Mutant site shape:", mut_site_embeddings.shape)
    print("WT + mutant feature shape:", paired_site_features.shape)
    print("Full site feature shape:", full_site_features.shape)
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()