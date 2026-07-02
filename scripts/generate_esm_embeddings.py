from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "results" / "SCN1A_mutant_sequences.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "embeddings"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"  # small test model first
MAX_AA_LENGTH = 1000  # safe length for ESM-2


def get_window(sequence, position, max_len=1000):
    """
    position is 1-indexed from UniProt/ClinVar.
    Returns a sequence window centered around the mutation.
    """
    pos0 = int(position) - 1

    half = max_len // 2
    start = max(0, pos0 - half)
    end = min(len(sequence), start + max_len)

    # adjust start if we hit the end
    start = max(0, end - max_len)

    window_seq = sequence[start:end]
    mutation_index_in_window = pos0 - start

    return window_seq, mutation_index_in_window, start + 1, end


def main():
    df = pd.read_csv(INPUT_CSV)

    # keep only clean mutant sequences
    df = df[df["Mutation_Status"] == "Success"].copy()
    df = df.dropna(subset=["Mutant_Sequence", "Binary_Label", "AA_Position"])

    print("Usable variants:", len(df))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    mutation_embeddings = []
    mean_embeddings = []
    metadata_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        seq = row["Mutant_Sequence"]
        position = int(row["AA_Position"])

        window_seq, mutation_idx, window_start, window_end = get_window(
            seq,
            position,
            MAX_AA_LENGTH
        )

        inputs = tokenizer(
            window_seq,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden = outputs.last_hidden_state[0]  # shape: tokens x embedding_dim

        # token index = mutation_idx + 1 because token 0 is CLS/start token
        mutation_token_index = mutation_idx + 1
        mutation_embedding = hidden[mutation_token_index].cpu().numpy()

        # Mean pool over amino-acid tokens only, excluding special tokens
        amino_acid_embeddings = hidden[1:-1]
        mean_embedding = amino_acid_embeddings.mean(dim=0).cpu().numpy()

        mutation_embeddings.append(mutation_embedding)
        mean_embeddings.append(mean_embedding)

        metadata_rows.append({
            "Variant_ID": row["Variant_ID"],
            "Gene": row["Gene"],
            "AA_Position": row["AA_Position"],
            "Ref_AA": row["Ref_AA"],
            "Alt_AA": row["Alt_AA"],
            "ClinVar_Label": row["ClinVar_Label"],
            "Binary_Label": row["Binary_Label"],
            "Mapped_Region": row["Mapped_Region"],
            "Mapped_Functional_Category": row["Mapped_Functional_Category"],
            "Window_Start": window_start,
            "Window_End": window_end,
            "Mutation_Index_In_Window": mutation_idx
        })

    mutation_embeddings = np.vstack(mutation_embeddings)
    mean_embeddings = np.vstack(mean_embeddings)
    metadata = pd.DataFrame(metadata_rows)

    np.save(OUTPUT_DIR / "mutation_site_embeddings.npy", mutation_embeddings)
    np.save(OUTPUT_DIR / "mean_window_embeddings.npy", mean_embeddings)
    metadata.to_csv(OUTPUT_DIR / "embedding_metadata.csv", index=False)

    print("Saved mutation-site embeddings:", mutation_embeddings.shape)
    print("Saved mean-window embeddings:", mean_embeddings.shape)
    print("Saved metadata:", OUTPUT_DIR / "embedding_metadata.csv")


if __name__ == "__main__":
    main()