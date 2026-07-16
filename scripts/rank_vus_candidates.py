from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWN_DIR = PROJECT_ROOT / "results" / "paired_embeddings"
VUS_DIR = PROJECT_ROOT / "results" / "vus_prioritization" / "vus_paired_embeddings"
OUTPUT_DIR = PROJECT_ROOT / "results" / "vus_prioritization" / "ranked_vus_candidates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))

USE_BIOCHEM_FEATURES = True

BATCH_SIZE = 32
MAX_EPOCHS = 80
PATIENCE = 12
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

K_NEIGHBORS = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VariantDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class CustomMutationNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.network(x)


def build_biochemical_features(metadata):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    hydropathy = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }

    molecular_weight = {
        "A": 89.09, "C": 121.16, "D": 133.10, "E": 147.13, "F": 165.19,
        "G": 75.07, "H": 155.16, "I": 131.17, "K": 146.19, "L": 131.17,
        "M": 149.21, "N": 132.12, "P": 115.13, "Q": 146.15, "R": 174.20,
        "S": 105.09, "T": 119.12, "V": 117.15, "W": 204.23, "Y": 181.19,
    }

    charge = {
        "D": -1, "E": -1,
        "K": 1, "R": 1, "H": 1,
    }

    polar = set(["S", "T", "N", "Q", "C", "Y", "D", "E", "K", "R", "H"])
    aromatic = set(["F", "W", "Y", "H"])
    sulfur = set(["C", "M"])
    special = set(["G", "P"])

    rows = []

    for _, row in metadata.iterrows():
        ref = str(row["Ref_AA"]).strip().upper()
        alt = str(row["Alt_AA"]).strip().upper()

        ref_hydro = hydropathy.get(ref, 0.0)
        alt_hydro = hydropathy.get(alt, 0.0)

        ref_mw = molecular_weight.get(ref, 0.0)
        alt_mw = molecular_weight.get(alt, 0.0)

        ref_charge = charge.get(ref, 0)
        alt_charge = charge.get(alt, 0)

        ref_polar = int(ref in polar)
        alt_polar = int(alt in polar)

        ref_aromatic = int(ref in aromatic)
        alt_aromatic = int(alt in aromatic)

        ref_sulfur = int(ref in sulfur)
        alt_sulfur = int(alt in sulfur)

        ref_special = int(ref in special)
        alt_special = int(alt in special)

        features = {
            "ref_hydropathy": ref_hydro,
            "alt_hydropathy": alt_hydro,
            "delta_hydropathy": alt_hydro - ref_hydro,
            "abs_delta_hydropathy": abs(alt_hydro - ref_hydro),

            "ref_molecular_weight": ref_mw,
            "alt_molecular_weight": alt_mw,
            "delta_molecular_weight": alt_mw - ref_mw,
            "abs_delta_molecular_weight": abs(alt_mw - ref_mw),

            "ref_charge": ref_charge,
            "alt_charge": alt_charge,
            "delta_charge": alt_charge - ref_charge,
            "abs_delta_charge": abs(alt_charge - ref_charge),

            "ref_polar": ref_polar,
            "alt_polar": alt_polar,
            "polarity_changed": int(ref_polar != alt_polar),

            "ref_aromatic": ref_aromatic,
            "alt_aromatic": alt_aromatic,
            "aromaticity_changed": int(ref_aromatic != alt_aromatic),

            "ref_sulfur": ref_sulfur,
            "alt_sulfur": alt_sulfur,
            "sulfur_status_changed": int(ref_sulfur != alt_sulfur),

            "ref_special_gly_pro": ref_special,
            "alt_special_gly_pro": alt_special,
            "special_status_changed": int(ref_special != alt_special),
        }

        for aa in amino_acids:
            features[f"ref_is_{aa}"] = int(ref == aa)
            features[f"alt_is_{aa}"] = int(alt == aa)

        rows.append(features)

    feature_df = pd.DataFrame(rows)
    return feature_df.values.astype(float), list(feature_df.columns)


def make_features(metadata, wt_site, mut_site, use_biochem=True):
    wt_plus_mut = np.concatenate([wt_site, mut_site], axis=1)

    if not use_biochem:
        return wt_plus_mut, []

    biochem, biochem_cols = build_biochemical_features(metadata)
    X = np.concatenate([wt_plus_mut, biochem], axis=1)

    return X, biochem_cols


def make_position_disjoint_train_val_split(y, groups, seed):
    indices = np.arange(len(y))

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    train_idx, val_idx = next(cv.split(indices, y, groups=groups))

    train_positions = set(groups[train_idx])
    val_positions = set(groups[val_idx])

    if train_positions.intersection(val_positions):
        raise ValueError("Train/validation position overlap detected.")

    return train_idx, val_idx


def predict_probs(model, X, device):
    dataset = VariantDataset(X)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    probs = []

    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            batch_probs = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(batch_probs.cpu().numpy())

    return np.array(probs)


def train_one_seed_and_predict_vus(
    X_known,
    y_known,
    groups,
    X_vus,
    seed,
    device,
):
    set_seed(seed)

    train_idx, val_idx = make_position_disjoint_train_val_split(
        y=y_known,
        groups=groups,
        seed=seed,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_known[train_idx])
    X_val = scaler.transform(X_known[val_idx])
    X_vus_scaled = scaler.transform(X_vus)

    y_train = y_known[train_idx]
    y_val = y_known[val_idx]

    train_dataset = VariantDataset(X_train, y_train)
    val_dataset = VariantDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_counts = np.bincount(y_train)
    total = class_counts.sum()

    class_weights = torch.tensor(
        [
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1]),
        ],
        dtype=torch.float32,
    ).to(device)

    model = CustomMutationNN(input_dim=X_known.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_auc = -1
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        val_probs = predict_probs(model, X_val, device)

        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(best_model_state)
    model.to(device)

    vus_probs = predict_probs(model, X_vus_scaled, device)

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auc": float(best_val_auc),
        "vus_probs": vus_probs,
    }


def compute_nearest_neighbor_features(X_known, y_known, known_metadata, X_vus, vus_metadata):
    """
    Uses standardized WT+mutant(+biochem) feature space to find nearest known variants.
    """

    scaler = StandardScaler()
    X_known_scaled = scaler.fit_transform(X_known)
    X_vus_scaled = scaler.transform(X_vus)

    nn = NearestNeighbors(
        n_neighbors=K_NEIGHBORS,
        metric="cosine",
    )

    nn.fit(X_known_scaled)

    distances, indices = nn.kneighbors(X_vus_scaled)

    neighbor_rows = []
    top_neighbor_rows = []

    for i in range(len(vus_metadata)):
        neighbor_idx = indices[i]
        neighbor_distances = distances[i]
        neighbor_labels = y_known[neighbor_idx]

        pathogenic_fraction = float(np.mean(neighbor_labels == 1))
        benign_fraction = float(np.mean(neighbor_labels == 0))

        mean_distance = float(np.mean(neighbor_distances))
        closest_distance = float(neighbor_distances[0])
        closest_label = int(neighbor_labels[0])

        neighbor_rows.append({
            "neighbor_pathogenic_fraction_k20": pathogenic_fraction,
            "neighbor_benign_fraction_k20": benign_fraction,
            "neighbor_mean_cosine_distance_k20": mean_distance,
            "closest_neighbor_cosine_distance": closest_distance,
            "closest_neighbor_label": closest_label,
        })

        vus_variant_id = vus_metadata.iloc[i]["Variant_ID"]

        for rank, known_idx in enumerate(neighbor_idx, start=1):
            known_row = known_metadata.iloc[known_idx]

            top_neighbor_rows.append({
                "VUS_Variant_ID": vus_variant_id,
                "Neighbor_Rank": rank,
                "Neighbor_Distance_Cosine": float(neighbor_distances[rank - 1]),
                "Known_Variant_ID": known_row.get("Variant_ID", ""),
                "Known_AA_Position": known_row.get("AA_Position", ""),
                "Known_Ref_AA": known_row.get("Ref_AA", ""),
                "Known_Alt_AA": known_row.get("Alt_AA", ""),
                "Known_Label": int(y_known[known_idx]),
                "Known_Label_Name": "Pathogenic" if int(y_known[known_idx]) == 1 else "Benign",
                "Known_Functional_Category": known_row.get("Mapped_Functional_Category", ""),
                "Known_ClinVar_Label": known_row.get("ClinVar_Label", ""),
            })

    neighbor_features = pd.DataFrame(neighbor_rows)
    top_neighbors = pd.DataFrame(top_neighbor_rows)

    return neighbor_features, top_neighbors


def domain_priority(category):
    category = str(category)

    if category == "Voltage_Sensor":
        return 1.00
    if category == "Inactivation_Gate":
        return 1.00
    if category == "Pore":
        return 0.70
    if category == "Other":
        return 0.20

    return 0.20


def minmax(values):
    values = np.array(values).reshape(-1, 1)

    if np.all(values == values[0]):
        return np.zeros(len(values))

    return MinMaxScaler().fit_transform(values).ravel()


def main():
    device = get_device()
    print("Using device:", device)

    known_metadata = pd.read_csv(KNOWN_DIR / "paired_embedding_metadata.csv")
    vus_metadata = pd.read_csv(VUS_DIR / "vus_embedding_metadata.csv")

    y_known = known_metadata["Binary_Label"].astype(int).values
    groups = known_metadata["AA_Position"].astype(int).values

    known_wt_site = np.load(KNOWN_DIR / "wt_site_embeddings.npy")
    known_mut_site = np.load(KNOWN_DIR / "mut_site_embeddings.npy")

    vus_wt_site = np.load(VUS_DIR / "vus_wt_site_embeddings.npy")
    vus_mut_site = np.load(VUS_DIR / "vus_mut_site_embeddings.npy")

    X_known, biochem_cols = make_features(
        known_metadata,
        known_wt_site,
        known_mut_site,
        use_biochem=USE_BIOCHEM_FEATURES,
    )

    X_vus, _ = make_features(
        vus_metadata,
        vus_wt_site,
        vus_mut_site,
        use_biochem=USE_BIOCHEM_FEATURES,
    )

    print("Known variants:", len(known_metadata))
    print("VUS variants:", len(vus_metadata))
    print("Known feature shape:", X_known.shape)
    print("VUS feature shape:", X_vus.shape)
    print("Using biochemical features:", USE_BIOCHEM_FEATURES)

    if USE_BIOCHEM_FEATURES:
        with open(OUTPUT_DIR / "biochemical_feature_columns.json", "w") as f:
            json.dump(biochem_cols, f, indent=4)

    all_seed_probs = []
    seed_summary_rows = []

    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"Training seed {seed}")
        print("=" * 80)

        result = train_one_seed_and_predict_vus(
            X_known=X_known,
            y_known=y_known,
            groups=groups,
            X_vus=X_vus,
            seed=seed,
            device=device,
        )

        all_seed_probs.append(result["vus_probs"])

        seed_summary_rows.append({
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "best_val_auc": result["best_val_auc"],
        })

        print(
            f"Seed {seed}: best_epoch={result['best_epoch']}, "
            f"best_val_auc={result['best_val_auc']:.4f}"
        )

    seed_probs = np.vstack(all_seed_probs).T

    seed_prob_df = pd.DataFrame(
        seed_probs,
        columns=[f"pathogenic_probability_seed_{seed}" for seed in SEEDS],
    )

    seed_summary = pd.DataFrame(seed_summary_rows)
    seed_summary.to_csv(OUTPUT_DIR / "vus_model_seed_summary.csv", index=False)

    vus_results = vus_metadata.copy()

    vus_results["Pathogenic_Probability_Mean"] = seed_probs.mean(axis=1)
    vus_results["Pathogenic_Probability_Std"] = seed_probs.std(axis=1)

    vus_results["Predicted_Label_Mean"] = (
        vus_results["Pathogenic_Probability_Mean"] >= 0.5
    ).astype(int)

    vus_results["Predicted_Label_Name"] = vus_results["Predicted_Label_Mean"].map(
        {0: "Predicted_Benign_Like", 1: "Predicted_Pathogenic_Like"}
    )

    vus_results["Model_Confidence"] = np.maximum(
        vus_results["Pathogenic_Probability_Mean"],
        1 - vus_results["Pathogenic_Probability_Mean"],
    )

    vus_results["Model_Uncertainty"] = 1 - vus_results["Model_Confidence"]

    vus_results["Ensemble_Disagreement"] = vus_results["Pathogenic_Probability_Std"]

    neighbor_features, top_neighbors = compute_nearest_neighbor_features(
        X_known=X_known,
        y_known=y_known,
        known_metadata=known_metadata,
        X_vus=X_vus,
        vus_metadata=vus_metadata,
    )

    vus_results = pd.concat(
        [vus_results.reset_index(drop=True), neighbor_features.reset_index(drop=True)],
        axis=1,
    )

    vus_results["Functional_Domain_Priority"] = vus_results[
        "Mapped_Functional_Category"
    ].apply(domain_priority)

    vus_results["Uncertainty_Normalized"] = minmax(vus_results["Model_Uncertainty"])
    vus_results["Ensemble_Disagreement_Normalized"] = minmax(vus_results["Ensemble_Disagreement"])
    vus_results["Pathogenic_Probability_Normalized"] = minmax(
        vus_results["Pathogenic_Probability_Mean"]
    )
    vus_results["Neighbor_Pathogenicity_Normalized"] = minmax(
        vus_results["neighbor_pathogenic_fraction_k20"]
    )

    # Main experimental follow-up score:
    # high uncertainty + important domain + close to pathogenic neighborhoods + ensemble disagreement
    vus_results["Experimental_Followup_Priority_Score"] = (
        0.35 * vus_results["Uncertainty_Normalized"]
        + 0.25 * vus_results["Functional_Domain_Priority"]
        + 0.25 * vus_results["Neighbor_Pathogenicity_Normalized"]
        + 0.15 * vus_results["Ensemble_Disagreement_Normalized"]
    )

    # Separate likely-pathogenic score:
    # useful if you want candidates that look high-risk rather than just uncertain.
    vus_results["Likely_Pathogenic_VUS_Score"] = (
        0.45 * vus_results["Pathogenic_Probability_Normalized"]
        + 0.25 * vus_results["Neighbor_Pathogenicity_Normalized"]
        + 0.20 * vus_results["Functional_Domain_Priority"]
        + 0.10 * vus_results["Ensemble_Disagreement_Normalized"]
    )

    # Separate ambiguous-mechanism score:
    # useful for variants that the model finds hard in important domains.
    vus_results["Ambiguous_Mechanism_Score"] = (
        0.50 * vus_results["Uncertainty_Normalized"]
        + 0.30 * vus_results["Functional_Domain_Priority"]
        + 0.20 * vus_results["Ensemble_Disagreement_Normalized"]
    )

    vus_results = pd.concat(
        [vus_results.reset_index(drop=True), seed_prob_df.reset_index(drop=True)],
        axis=1,
    )

    ranked_main = vus_results.sort_values(
        "Experimental_Followup_Priority_Score",
        ascending=False,
    )

    ranked_likely_pathogenic = vus_results.sort_values(
        "Likely_Pathogenic_VUS_Score",
        ascending=False,
    )

    ranked_ambiguous = vus_results.sort_values(
        "Ambiguous_Mechanism_Score",
        ascending=False,
    )

    ranked_main.to_csv(
        OUTPUT_DIR / "ranked_vus_experimental_followup_candidates.csv",
        index=False,
    )

    ranked_likely_pathogenic.to_csv(
        OUTPUT_DIR / "ranked_vus_likely_pathogenic_candidates.csv",
        index=False,
    )

    ranked_ambiguous.to_csv(
        OUTPUT_DIR / "ranked_vus_ambiguous_mechanism_candidates.csv",
        index=False,
    )

    top_neighbors.to_csv(
        OUTPUT_DIR / "vus_top20_known_neighbors.csv",
        index=False,
    )

    summary_by_domain = (
        vus_results
        .groupby("Mapped_Functional_Category")
        .agg(
            n=("Variant_ID", "count"),
            mean_pathogenic_probability=("Pathogenic_Probability_Mean", "mean"),
            mean_uncertainty=("Model_Uncertainty", "mean"),
            mean_ensemble_disagreement=("Ensemble_Disagreement", "mean"),
            mean_neighbor_pathogenic_fraction=("neighbor_pathogenic_fraction_k20", "mean"),
            mean_followup_priority_score=("Experimental_Followup_Priority_Score", "mean"),
        )
        .reset_index()
        .sort_values("mean_followup_priority_score", ascending=False)
    )

    summary_by_domain.to_csv(
        OUTPUT_DIR / "vus_summary_by_functional_domain.csv",
        index=False,
    )

    config = {
        "seeds": SEEDS,
        "feature_set": "WT + mutant site embeddings + biochemical features"
        if USE_BIOCHEM_FEATURES
        else "WT + mutant site embeddings",
        "k_neighbors": K_NEIGHBORS,
        "priority_score_formula": {
            "Experimental_Followup_Priority_Score": {
                "uncertainty": 0.35,
                "functional_domain_priority": 0.25,
                "neighbor_pathogenic_fraction": 0.25,
                "ensemble_disagreement": 0.15,
            },
            "Likely_Pathogenic_VUS_Score": {
                "pathogenic_probability": 0.45,
                "neighbor_pathogenic_fraction": 0.25,
                "functional_domain_priority": 0.20,
                "ensemble_disagreement": 0.10,
            },
            "Ambiguous_Mechanism_Score": {
                "uncertainty": 0.50,
                "functional_domain_priority": 0.30,
                "ensemble_disagreement": 0.20,
            },
        },
        "important_warning": (
            "These rankings are for experimental-prioritization research only. "
            "They are not clinical classifications."
        ),
    }

    with open(OUTPUT_DIR / "vus_ranking_config.json", "w") as f:
        json.dump(config, f, indent=4)

    display_cols = [
        "Variant_ID",
        "Protein_Change_Parsed",
        "Clinical_Significance",
        "Mapped_Functional_Category",
        "Pathogenic_Probability_Mean",
        "Pathogenic_Probability_Std",
        "Model_Uncertainty",
        "neighbor_pathogenic_fraction_k20",
        "Functional_Domain_Priority",
        "Experimental_Followup_Priority_Score",
        "Likely_Pathogenic_VUS_Score",
        "Ambiguous_Mechanism_Score",
    ]

    print("\n" + "=" * 100)
    print("VUS SUMMARY BY FUNCTIONAL DOMAIN")
    print("=" * 100)
    print(summary_by_domain.to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP 25 EXPERIMENTAL FOLLOW-UP VUS CANDIDATES")
    print("=" * 100)
    print(ranked_main[display_cols].head(25).to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP 25 LIKELY PATHOGENIC-LIKE VUS CANDIDATES")
    print("=" * 100)
    print(ranked_likely_pathogenic[display_cols].head(25).to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP 25 AMBIGUOUS MECHANISM VUS CANDIDATES")
    print("=" * 100)
    print(ranked_ambiguous[display_cols].head(25).to_string(index=False))

    print("\nSaved VUS ranking outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()