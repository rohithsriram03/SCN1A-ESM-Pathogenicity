from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "processed"
    / "scion_mechanism_variants_with_sequences.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 1


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def embed_site_sequences(sequences, local_indices, tokenizer, model, device):
    site_embeddings = []

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
            # ESM adds a beginning token, so mutation site is local_idx + 1.
            token_idx = int(local_idx) + 1
            emb = hidden[i, token_idx, :].detach().cpu().numpy()
            site_embeddings.append(emb)

    return np.vstack(site_embeddings)


def main():
    print("Reading validated SCION sequence dataset:")
    print(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    print("Rows:", len(df))
    print("Mechanism counts:")
    print(df["Mechanism_Label"].value_counts().to_string())

    wt_sequences = df["WT_Window"].astype(str).tolist()
    mut_sequences = df["Mutant_Window"].astype(str).tolist()
    local_indices = df["Local_Mutation_Index_0Indexed"].astype(int).tolist()

    device = choose_device()
    print("\nDevice:", device)

    print("\nLoading model:")
    print(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print("\nEmbedding WT windows...")
    wt_site = embed_site_sequences(
        wt_sequences,
        local_indices,
        tokenizer,
        model,
        device,
    )

    print("\nEmbedding mutant windows...")
    mut_site = embed_site_sequences(
        mut_sequences,
        local_indices,
        tokenizer,
        model,
        device,
    )

    delta_site = mut_site - wt_site
    abs_delta_site = np.abs(delta_site)

    wt_mut_site = np.concatenate([wt_site, mut_site], axis=1)
    full_site_features = np.concatenate(
        [wt_site, mut_site, delta_site, abs_delta_site],
        axis=1,
    )

    print("\nWT site shape:", wt_site.shape)
    print("Mutant site shape:", mut_site.shape)
    print("Delta site shape:", delta_site.shape)
    print("WT+mutant shape:", wt_mut_site.shape)
    print("Full feature shape:", full_site_features.shape)

    np.save(OUTPUT_DIR / "wt_site_embeddings.npy", wt_site)
    np.save(OUTPUT_DIR / "mutant_site_embeddings.npy", mut_site)
    np.save(OUTPUT_DIR / "delta_site_embeddings.npy", delta_site)
    np.save(OUTPUT_DIR / "abs_delta_site_embeddings.npy", abs_delta_site)
    np.save(OUTPUT_DIR / "wt_mut_site_features.npy", wt_mut_site)
    np.save(OUTPUT_DIR / "full_site_features.npy", full_site_features)

    metadata_file = OUTPUT_DIR / "scion_embedding_metadata.csv"
    df.to_csv(metadata_file, index=False)

    qc = {
        "input_file": str(INPUT_FILE),
        "n_variants": int(len(df)),
        "model_name": MODEL_NAME,
        "wt_site_shape": list(wt_site.shape),
        "mutant_site_shape": list(mut_site.shape),
        "wt_mut_site_shape": list(wt_mut_site.shape),
        "full_site_features_shape": list(full_site_features.shape),
        "mechanism_label_counts": df["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": df["Gene"].value_counts().to_dict(),
        "metadata_file": str(metadata_file),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "embedding_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\nQC summary:")
    print(json.dumps(qc, indent=4))

    print("\nSaved embeddings to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()