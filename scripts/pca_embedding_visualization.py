from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "pca_embedding_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_pca(X, n_components=2):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(X_scaled)

    return coords, pca


def plot_pca_by_label(coords, metadata, title, filename, explained_variance):
    df = metadata.copy()
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    label_map = {
        0: "Benign",
        1: "Pathogenic"
    }

    plt.figure(figsize=(7, 6))

    for label_value, label_name in label_map.items():
        sub = df[df["Binary_Label"] == label_value]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            s=14,
            alpha=0.65,
            label=label_name
        )

    plt.xlabel(f"PC1 ({explained_variance[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained_variance[1] * 100:.2f}% variance)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

    df.to_csv(OUTPUT_DIR / filename.replace(".png", "_coordinates.csv"), index=False)


def plot_pca_by_domain(coords, metadata, title, filename, explained_variance):
    df = metadata.copy()
    df["PC1"] = coords[:, 0]
    df["PC2"] = coords[:, 1]

    categories = sorted(df["Mapped_Functional_Category"].dropna().unique())

    plt.figure(figsize=(8, 6))

    for category in categories:
        sub = df[df["Mapped_Functional_Category"] == category]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            s=14,
            alpha=0.65,
            label=category
        )

    plt.xlabel(f"PC1 ({explained_variance[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained_variance[1] * 100:.2f}% variance)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

    df.to_csv(OUTPUT_DIR / filename.replace(".png", "_coordinates.csv"), index=False)


def plot_wt_to_mutant_shift(wt_embeddings, mut_embeddings, metadata):
    """
    Projects WT and mutant embeddings into the same PCA space,
    then draws arrows from WT to mutant for a random subset of variants.
    """

    combined = np.vstack([wt_embeddings, mut_embeddings])

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=2, random_state=42)
    combined_coords = pca.fit_transform(combined_scaled)

    n = len(metadata)

    wt_coords = combined_coords[:n]
    mut_coords = combined_coords[n:]

    df = metadata.copy()
    df["WT_PC1"] = wt_coords[:, 0]
    df["WT_PC2"] = wt_coords[:, 1]
    df["Mut_PC1"] = mut_coords[:, 0]
    df["Mut_PC2"] = mut_coords[:, 1]

    # Save full coordinate table
    df.to_csv(OUTPUT_DIR / "wt_to_mutant_shift_pca_coordinates.csv", index=False)

    # Plot only subset so arrows are readable
    rng = np.random.default_rng(42)
    subset_size = min(300, n)
    subset_indices = rng.choice(np.arange(n), size=subset_size, replace=False)

    plt.figure(figsize=(8, 7))

    for label_value, label_name in [(0, "Benign"), (1, "Pathogenic")]:
        selected = [
            idx for idx in subset_indices
            if metadata.iloc[idx]["Binary_Label"] == label_value
        ]

        plt.scatter(
            wt_coords[selected, 0],
            wt_coords[selected, 1],
            s=12,
            alpha=0.35,
            label=f"{label_name} WT"
        )

        plt.scatter(
            mut_coords[selected, 0],
            mut_coords[selected, 1],
            s=12,
            alpha=0.65,
            label=f"{label_name} Mutant"
        )

        for idx in selected:
            plt.arrow(
                wt_coords[idx, 0],
                wt_coords[idx, 1],
                mut_coords[idx, 0] - wt_coords[idx, 0],
                mut_coords[idx, 1] - wt_coords[idx, 1],
                alpha=0.18,
                length_includes_head=True,
                head_width=0.05
            )

    explained = pca.explained_variance_ratio_

    plt.xlabel(f"PC1 ({explained[0] * 100:.2f}% variance)")
    plt.ylabel(f"PC2 ({explained[1] * 100:.2f}% variance)")
    plt.title("PCA shift from wild-type to mutant embeddings")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "pca_wt_to_mutant_shift_arrows.png", dpi=300)
    plt.close()


def main():
    metadata = pd.read_csv(INPUT_DIR / "paired_embedding_metadata.csv")

    wt_site = np.load(INPUT_DIR / "wt_site_embeddings.npy")
    mut_site = np.load(INPUT_DIR / "mut_site_embeddings.npy")
    delta_site = np.load(INPUT_DIR / "delta_site_embeddings.npy")
    abs_delta_site = np.load(INPUT_DIR / "abs_delta_site_embeddings.npy")

    wt_plus_mut_site = np.concatenate([wt_site, mut_site], axis=1)

    full_site_features = np.concatenate(
        [wt_site, mut_site, delta_site, abs_delta_site],
        axis=1
    )

    feature_sets = {
        "wt_site": wt_site,
        "mutant_site": mut_site,
        "delta_site": delta_site,
        "wt_plus_mut_site": wt_plus_mut_site,
        "full_site_wt_mut_delta_abs": full_site_features,
    }

    print("Metadata rows:", len(metadata))
    print("WT site shape:", wt_site.shape)
    print("Mutant site shape:", mut_site.shape)
    print("Delta site shape:", delta_site.shape)
    print("WT + mutant shape:", wt_plus_mut_site.shape)
    print("Full site feature shape:", full_site_features.shape)

    summary_rows = []

    for feature_name, X in feature_sets.items():
        print(f"\nRunning PCA for: {feature_name}")
        print("Shape:", X.shape)

        coords, pca = run_pca(X, n_components=2)
        explained = pca.explained_variance_ratio_

        summary_rows.append({
            "feature_set": feature_name,
            "input_dim": X.shape[1],
            "pc1_variance_explained": explained[0],
            "pc2_variance_explained": explained[1],
            "total_pc1_pc2_variance_explained": explained[0] + explained[1],
        })

        plot_pca_by_label(
            coords=coords,
            metadata=metadata,
            title=f"PCA of {feature_name} embeddings by pathogenicity label",
            filename=f"pca_{feature_name}_by_label.png",
            explained_variance=explained,
        )

        plot_pca_by_domain(
            coords=coords,
            metadata=metadata,
            title=f"PCA of {feature_name} embeddings by functional region",
            filename=f"pca_{feature_name}_by_domain.png",
            explained_variance=explained,
        )

    print("\nRunning WT-to-mutant shift PCA arrow plot...")
    plot_wt_to_mutant_shift(
        wt_embeddings=wt_site,
        mut_embeddings=mut_site,
        metadata=metadata,
    )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "pca_explained_variance_summary.csv", index=False)

    print("\nPCA explained variance summary:")
    print(summary.to_string(index=False))

    print("\nSaved PCA visualizations to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()