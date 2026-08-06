from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "scn1a_benchmark_with_alphamissense.csv"
)

FASTA_FILE = PROJECT_ROOT / "data" / "SCN1A_uniprot_P35498.fasta"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "benchmarking"
    / "esm2_650M_embeddings"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
WINDOW_SIZE = 1000
BATCH_SIZE = 1

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                continue
            lines.append(line.strip())
    return "".join(lines)


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


def make_mutant_window(wt_window, local_idx, ref, alt):
    observed = wt_window[local_idx]

    if observed != ref:
        raise ValueError(
            f"Reference mismatch inside window: expected {ref}, observed {observed}"
        )

    return wt_window[:local_idx] + alt + wt_window[local_idx + 1:]


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def embed_sequences(sequences, local_indices, tokenizer, model, device):
    all_site_embeddings = []

    for start in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Embedding"):
        batch_sequences = sequences[start:start + BATCH_SIZE]
        batch_local_indices = local_indices[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state

        for i, local_idx in enumerate(batch_local_indices):
            # ESM tokenization includes a start token, so add 1.
            token_idx = int(local_idx) + 1
            site_embedding = hidden[i, token_idx, :].detach().cpu().numpy()
            all_site_embeddings.append(site_embedding)

    return np.vstack(all_site_embeddings)


def main():
    print("Reading benchmark:")
    print(BENCHMARK_FILE)

    df = pd.read_csv(BENCHMARK_FILE)

    print("Rows:", len(df))

    print("\nReading WT sequence:")
    print(FASTA_FILE)

    wt_sequence = read_fasta(FASTA_FILE)
    print("WT length:", len(wt_sequence))

    records = []
    wt_windows = []
    mut_windows = []
    local_indices = []

    skipped = []

    for idx, row in df.iterrows():
        ref = str(row["Ref_AA"]).strip().upper()
        alt = str(row["Alt_AA"]).strip().upper()
        pos = int(row["AA_Position"])

        status = "OK"

        try:
            if ref not in VALID_AAS or alt not in VALID_AAS:
                raise ValueError("Invalid amino acid")

            wt_window, local_idx, window_start, window_end = make_centered_window(
                wt_sequence,
                pos,
                WINDOW_SIZE,
            )

            if wt_window[local_idx] != ref:
                raise ValueError(
                    f"Reference mismatch: expected {ref}, observed {wt_window[local_idx]}"
                )

            mut_window = make_mutant_window(wt_window, local_idx, ref, alt)

            wt_windows.append(wt_window)
            mut_windows.append(mut_window)
            local_indices.append(local_idx)

            records.append({
                "original_row_index": idx,
                "Variant_Key": row["Variant_Key"],
                "Variant_ID": row.get("Variant_ID", ""),
                "Protein_Change_Parsed": row["Protein_Change_Parsed"],
                "Gene": row["Gene"],
                "AA_Position": pos,
                "Ref_AA": ref,
                "Alt_AA": alt,
                "Binary_Label": row["Binary_Label"],
                "Label_Name": row["Label_Name"],
                "ClinVar_Label": row.get("ClinVar_Label", ""),
                "Mapped_Region": row.get("Mapped_Region", ""),
                "Mapped_Functional_Category": row.get("Mapped_Functional_Category", ""),
                "AlphaMissense_Score": row.get("AlphaMissense_Score", np.nan),
                "AlphaMissense_Class": row.get("AlphaMissense_Class", ""),
                "window_start_0indexed": window_start,
                "window_end_0indexed": window_end,
                "local_idx_0indexed": local_idx,
                "Embedding_Status": status,
            })

        except Exception as e:
            skipped.append({
                "original_row_index": idx,
                "Variant_Key": row.get("Variant_Key", ""),
                "Protein_Change_Parsed": row.get("Protein_Change_Parsed", ""),
                "AA_Position": pos,
                "Ref_AA": ref,
                "Alt_AA": alt,
                "Embedding_Status": str(e),
            })

    metadata = pd.DataFrame(records)
    skipped_df = pd.DataFrame(skipped)

    print("\nPrepared variants for embedding:", len(metadata))
    print("Skipped variants:", len(skipped_df))

    if len(skipped_df) > 0:
        skipped_df.to_csv(OUTPUT_DIR / "skipped_variants_650m.csv", index=False)

    print("\nLoading tokenizer/model:")
    print(MODEL_NAME)

    device = choose_device()
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("\nEmbedding WT windows...")
    wt_site = embed_sequences(
        wt_windows,
        local_indices,
        tokenizer,
        model,
        device,
    )

    print("\nEmbedding mutant windows...")
    mut_site = embed_sequences(
        mut_windows,
        local_indices,
        tokenizer,
        model,
        device,
    )

    wt_mut_features = np.concatenate([wt_site, mut_site], axis=1)

    print("\nWT site shape:", wt_site.shape)
    print("Mutant site shape:", mut_site.shape)
    print("WT+mut features shape:", wt_mut_features.shape)

    np.save(OUTPUT_DIR / "wt_site_embeddings_650m.npy", wt_site)
    np.save(OUTPUT_DIR / "mut_site_embeddings_650m.npy", mut_site)
    np.save(OUTPUT_DIR / "wt_mut_site_features_650m.npy", wt_mut_features)

    metadata.to_csv(OUTPUT_DIR / "paired_embedding_metadata_650m.csv", index=False)

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()