from pathlib import Path
import json
from itertools import combinations
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "paralog_same_position_challenges"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
    "biochem_only": None,
}

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

MOLECULAR_WEIGHT = {
    "A": 89.1, "C": 121.2, "D": 133.1, "E": 147.1, "F": 165.2,
    "G": 75.1, "H": 155.2, "I": 131.2, "K": 146.2, "L": 131.2,
    "M": 149.2, "N": 132.1, "P": 115.1, "Q": 146.2, "R": 174.2,
    "S": 105.1, "T": 119.1, "V": 117.1, "W": 204.2, "Y": 181.2,
}

CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.5}
POLAR = set(["D", "E", "K", "R", "H", "N", "Q", "S", "T", "Y", "C"])
AROMATIC = set(["F", "W", "Y", "H"])


def aa_one_hot(aa):
    return [1 if aa == x else 0 for x in AA_LIST]


def build_biochem_features(meta):
    rows = []

    for _, row in meta.iterrows():
        ref = str(row["Ref_AA"])
        alt = str(row["Alt_AA"])

        ref_h = HYDROPATHY.get(ref, 0)
        alt_h = HYDROPATHY.get(alt, 0)

        ref_w = MOLECULAR_WEIGHT.get(ref, 0)
        alt_w = MOLECULAR_WEIGHT.get(alt, 0)

        ref_c = CHARGE.get(ref, 0)
        alt_c = CHARGE.get(alt, 0)

        features = [
            ref_h,
            alt_h,
            alt_h - ref_h,
            abs(alt_h - ref_h),
            ref_w,
            alt_w,
            alt_w - ref_w,
            abs(alt_w - ref_w),
            ref_c,
            alt_c,
            alt_c - ref_c,
            abs(alt_c - ref_c),
            int(ref in POLAR),
            int(alt in POLAR),
            int(ref in POLAR) != int(alt in POLAR),
            int(ref in AROMATIC),
            int(alt in AROMATIC),
            int(ref in AROMATIC) != int(alt in AROMATIC),
            int(ref == "G"),
            int(alt == "G"),
            int(ref == "P"),
            int(alt == "P"),
        ]

        features.extend(aa_one_hot(ref))
        features.extend(aa_one_hot(alt))

        rows.append(features)

    return np.array(rows, dtype=float)


def make_model(X_train, seed=0):
    steps = [("scaler", StandardScaler())]

    if X_train.shape[1] > 80:
        n_components = min(50, X_train.shape[0] - 2, X_train.shape[1])
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=seed,
                    svd_solver="randomized",
                ),
            )
        )

    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
            ),
        )
    )

    return Pipeline(steps)


def safe_metrics(y_true, scores):
    y_true = np.array(y_true)
    scores = np.array(scores)

    if len(y_true) == 0:
        return None

    preds = (scores >= 0.5).astype(int)

    metrics = {
        "n": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y_true, preds, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["average_precision"] = average_precision_score(y_true, scores)
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def bootstrap_ci(values, n_boot=5000, seed=0):
    values = np.array(values, dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        boots.append(np.mean(values[idx]))

    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_features(meta):
    feature_arrays = {}

    for name, path in FEATURE_FILES.items():
        if name == "biochem_only":
            X = build_biochem_features(meta)
        else:
            X = np.load(path)

        feature_arrays[name] = X
        print(f"{name}: {X.shape}")

    return feature_arrays


def generate_leave_one_gene_predictions(meta, feature_arrays):
    """
    For every gene, train on all other genes and predict the held-out gene.
    This avoids using the same gene's labels when making predictions.
    """
    out = meta.copy()
    y = meta["Mechanism_Binary"].astype(int).values

    metrics_rows = []

    for feature_name, X in feature_arrays.items():
        score_col = f"logo_{feature_name}_gof_score"
        out[score_col] = np.nan

        for gene in sorted(meta["Gene"].unique()):
            train_idx = np.where(meta["Gene"].values != gene)[0]
            test_idx = np.where(meta["Gene"].values == gene)[0]

            y_train = y[train_idx]
            y_test = y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            model = make_model(X[train_idx], seed=0)
            model.fit(X[train_idx], y_train)

            scores = model.predict_proba(X[test_idx])[:, 1]
            out.loc[out.index[test_idx], score_col] = scores

            if len(np.unique(y_test)) == 2:
                m = safe_metrics(y_test, scores)
                m.update(
                    {
                        "experiment": "leave_one_gene_out",
                        "feature_set": feature_name,
                        "heldout_gene": gene,
                    }
                )
                metrics_rows.append(m)

    pred_file = OUTPUT_DIR / "leave_one_gene_predictions.csv"
    metrics_file = OUTPUT_DIR / "leave_one_gene_metrics.csv"

    out.to_csv(pred_file, index=False)
    pd.DataFrame(metrics_rows).to_csv(metrics_file, index=False)

    return out, pd.DataFrame(metrics_rows)


def make_scn1a_paralog_analogy_table(pred):
    """
    For each SCN1A variant, find variants in other SCN genes with the same Family_Alignment_CID.
    """
    rows = []
    scn1a = pred[pred["Gene"] == "SCN1A"].copy()

    for _, row in scn1a.iterrows():
        cid = row["Family_Alignment_CID"]

        analogs = pred[
            (pred["Family_Alignment_CID"] == cid)
            & (pred["Gene"] != "SCN1A")
        ].copy()

        n_analog = len(analogs)
        n_analog_gof = int((analogs["Mechanism_Label"] == "GOF").sum())
        n_analog_lof = int((analogs["Mechanism_Label"] == "LOF").sum())

        if n_analog > 0:
            analog_gof_fraction = n_analog_gof / n_analog
            analog_genes = ";".join(sorted(analogs["Gene"].unique()))
            analog_variants = ";".join(analogs["Variant_Key"].astype(str).tolist())
            analog_labels = ";".join(analogs["Mechanism_Label"].astype(str).tolist())
        else:
            analog_gof_fraction = np.nan
            analog_genes = ""
            analog_variants = ""
            analog_labels = ""

        out = row.to_dict()

        out.update(
            {
                "n_same_cid_non_scn1a_analogs": n_analog,
                "n_same_cid_non_scn1a_gof": n_analog_gof,
                "n_same_cid_non_scn1a_lof": n_analog_lof,
                "same_cid_non_scn1a_gof_fraction": analog_gof_fraction,
                "same_cid_non_scn1a_genes": analog_genes,
                "same_cid_non_scn1a_variants": analog_variants,
                "same_cid_non_scn1a_labels": analog_labels,
                "has_same_cid_non_scn1a_analog": int(n_analog > 0),
                "has_mixed_same_cid_non_scn1a_analogs": int(
                    n_analog_gof > 0 and n_analog_lof > 0
                ),
            }
        )

        rows.append(out)

    analogy = pd.DataFrame(rows)
    analogy.to_csv(OUTPUT_DIR / "scn1a_paralog_analogy_table.csv", index=False)

    return analogy


def evaluate_scn1a_analogy_subsets(analogy):
    rows = []

    subsets = {
        "all_scn1a": analogy,
        "same_cid_analog_supported": analogy[
            analogy["has_same_cid_non_scn1a_analog"] == 1
        ],
        "no_same_cid_analog": analogy[
            analogy["has_same_cid_non_scn1a_analog"] == 0
        ],
        "mixed_same_cid_analogs": analogy[
            analogy["has_mixed_same_cid_non_scn1a_analogs"] == 1
        ],
    }

    score_cols = {
        "mutant_site_logo_model": "logo_mutant_site_gof_score",
        "wt_mut_site_logo_model": "logo_wt_mut_site_gof_score",
        "biochem_only_logo_model": "logo_biochem_only_gof_score",
        "same_cid_analog_majority": "same_cid_non_scn1a_gof_fraction",
    }

    for subset_name, df in subsets.items():
        for score_name, score_col in score_cols.items():
            eval_df = df.dropna(subset=[score_col]).copy()

            if len(eval_df) == 0:
                continue

            y = eval_df["Mechanism_Binary"].astype(int).values
            scores = eval_df[score_col].astype(float).values

            m = safe_metrics(y, scores)

            if m is None:
                continue

            m.update(
                {
                    "subset": subset_name,
                    "score_name": score_name,
                }
            )

            rows.append(m)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "scn1a_paralog_analogy_metrics.csv", index=False)

    return metrics


def add_pair_scores(pair, gof_row, lof_row, feature_names):
    for feature in feature_names:
        score_col = f"logo_{feature}_gof_score"

        gof_score = float(gof_row[score_col])
        lof_score = float(lof_row[score_col])

        if gof_score > lof_score:
            success = 1.0
        elif gof_score < lof_score:
            success = 0.0
        else:
            success = 0.5

        pair[f"{feature}_gof_score_for_true_gof"] = gof_score
        pair[f"{feature}_gof_score_for_true_lof"] = lof_score
        pair[f"{feature}_pairwise_success"] = success

    return pair


def build_same_gene_same_position_pairs(pred, feature_names):
    """
    Within the same gene and same AA position:
    Does the model score true GoF substitutions higher than true LoF substitutions?
    """
    rows = []

    grouped = pred.groupby(["Gene", "AA_Position"], dropna=False)

    for (gene, position), group in grouped:
        if len(group) < 2:
            continue

        gofs = group[group["Mechanism_Label"] == "GOF"]
        lofs = group[group["Mechanism_Label"] == "LOF"]

        if len(gofs) == 0 or len(lofs) == 0:
            continue

        for _, gof_row in gofs.iterrows():
            for _, lof_row in lofs.iterrows():
                pair = {
                    "challenge_type": "same_gene_same_position_discordant",
                    "group_id": f"{gene}_{int(position)}",
                    "Family_Alignment_CID": gof_row["Family_Alignment_CID"],
                    "gof_variant": gof_row["Variant_Key"],
                    "lof_variant": lof_row["Variant_Key"],
                    "gof_gene": gof_row["Gene"],
                    "lof_gene": lof_row["Gene"],
                    "gof_position": gof_row["AA_Position"],
                    "lof_position": lof_row["AA_Position"],
                    "gof_label": gof_row["Mechanism_Label"],
                    "lof_label": lof_row["Mechanism_Label"],
                }

                pair = add_pair_scores(pair, gof_row, lof_row, feature_names)
                rows.append(pair)

    pairs = pd.DataFrame(rows)
    pairs.to_csv(
        OUTPUT_DIR / "same_gene_same_position_discordant_pairs.csv",
        index=False,
    )

    return pairs


def build_same_cid_cross_gene_pairs(pred, feature_names, require_scn1a=False):
    """
    Across different sodium-channel genes but same family-aligned position.
    """
    rows = []

    grouped = pred.groupby("Family_Alignment_CID", dropna=False)

    for cid, group in grouped:
        if len(group) < 2:
            continue

        gofs = group[group["Mechanism_Label"] == "GOF"]
        lofs = group[group["Mechanism_Label"] == "LOF"]

        if len(gofs) == 0 or len(lofs) == 0:
            continue

        for _, gof_row in gofs.iterrows():
            for _, lof_row in lofs.iterrows():
                if gof_row["Gene"] == lof_row["Gene"]:
                    continue

                if require_scn1a:
                    genes = {gof_row["Gene"], lof_row["Gene"]}
                    if "SCN1A" not in genes:
                        continue
                    if len(genes) == 1:
                        continue

                pair = {
                    "challenge_type": (
                        "scn1a_same_cid_cross_paralog_discordant"
                        if require_scn1a
                        else "same_cid_cross_paralog_discordant"
                    ),
                    "group_id": f"CID_{int(cid)}",
                    "Family_Alignment_CID": cid,
                    "gof_variant": gof_row["Variant_Key"],
                    "lof_variant": lof_row["Variant_Key"],
                    "gof_gene": gof_row["Gene"],
                    "lof_gene": lof_row["Gene"],
                    "gof_position": gof_row["AA_Position"],
                    "lof_position": lof_row["AA_Position"],
                    "gof_label": gof_row["Mechanism_Label"],
                    "lof_label": lof_row["Mechanism_Label"],
                }

                pair = add_pair_scores(pair, gof_row, lof_row, feature_names)
                rows.append(pair)

    pairs = pd.DataFrame(rows)

    if require_scn1a:
        out_file = OUTPUT_DIR / "scn1a_same_cid_cross_paralog_discordant_pairs.csv"
    else:
        out_file = OUTPUT_DIR / "same_cid_cross_paralog_discordant_pairs.csv"

    pairs.to_csv(out_file, index=False)

    return pairs


def summarize_pairwise_pairs(pairs, challenge_name, feature_names):
    rows = []

    if pairs is None or len(pairs) == 0:
        for feature in feature_names:
            rows.append(
                {
                    "challenge": challenge_name,
                    "feature_set": feature,
                    "n_pairs": 0,
                    "n_unique_groups": 0,
                    "pairwise_accuracy": np.nan,
                    "pairwise_accuracy_ci_low": np.nan,
                    "pairwise_accuracy_ci_high": np.nan,
                    "strict_successes": 0,
                    "strict_failures": 0,
                    "ties": 0,
                    "binomial_p_greater_than_0_5": np.nan,
                }
            )
        return rows

    for feature in feature_names:
        col = f"{feature}_pairwise_success"

        values = pairs[col].dropna().astype(float).values

        n_pairs = len(values)
        n_unique_groups = pairs["group_id"].nunique()

        strict_successes = int((values == 1.0).sum())
        strict_failures = int((values == 0.0).sum())
        ties = int((values == 0.5).sum())

        ci_low, ci_high = bootstrap_ci(values)

        if binomtest is not None and (strict_successes + strict_failures) > 0:
            p_value = binomtest(
                strict_successes,
                strict_successes + strict_failures,
                0.5,
                alternative="greater",
            ).pvalue
        else:
            p_value = np.nan

        rows.append(
            {
                "challenge": challenge_name,
                "feature_set": feature,
                "n_pairs": int(n_pairs),
                "n_unique_groups": int(n_unique_groups),
                "pairwise_accuracy": float(np.mean(values)),
                "pairwise_accuracy_ci_low": ci_low,
                "pairwise_accuracy_ci_high": ci_high,
                "strict_successes": strict_successes,
                "strict_failures": strict_failures,
                "ties": ties,
                "binomial_p_greater_than_0_5": p_value,
            }
        )

    return rows


def main():
    print("Reading metadata:")
    print(META_FILE)

    meta = pd.read_csv(META_FILE)

    print("\nDataset:")
    print("Rows:", len(meta))
    print("\nMechanism counts:")
    print(meta["Mechanism_Label"].value_counts().to_string())
    print("\nGene × mechanism:")
    print(pd.crosstab(meta["Gene"], meta["Mechanism_Label"]).to_string())

    print("\nLoading features:")
    feature_arrays = load_features(meta)
    feature_names = list(feature_arrays.keys())

    print("\nGenerating leave-one-gene-out predictions...")
    pred, gene_metrics = generate_leave_one_gene_predictions(meta, feature_arrays)

    print("\nBuilding SCN1A paralog analogy table...")
    analogy = make_scn1a_paralog_analogy_table(pred)
    analogy_metrics = evaluate_scn1a_analogy_subsets(analogy)

    print("\nBuilding same-position and same-CID challenge pairs...")
    same_position_pairs = build_same_gene_same_position_pairs(pred, feature_names)

    same_cid_pairs = build_same_cid_cross_gene_pairs(
        pred,
        feature_names,
        require_scn1a=False,
    )

    scn1a_same_cid_pairs = build_same_cid_cross_gene_pairs(
        pred,
        feature_names,
        require_scn1a=True,
    )

    pairwise_rows = []
    pairwise_rows.extend(
        summarize_pairwise_pairs(
            same_position_pairs,
            "same_gene_same_position_discordant",
            feature_names,
        )
    )
    pairwise_rows.extend(
        summarize_pairwise_pairs(
            same_cid_pairs,
            "same_cid_cross_paralog_discordant",
            feature_names,
        )
    )
    pairwise_rows.extend(
        summarize_pairwise_pairs(
            scn1a_same_cid_pairs,
            "scn1a_same_cid_cross_paralog_discordant",
            feature_names,
        )
    )

    pairwise_metrics = pd.DataFrame(pairwise_rows)
    pairwise_metrics.to_csv(OUTPUT_DIR / "challenge_pairwise_metrics.csv", index=False)

    qc = {
        "n_variants": int(len(meta)),
        "mechanism_counts": meta["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": meta["Gene"].value_counts().to_dict(),
        "n_scn1a_variants": int((meta["Gene"] == "SCN1A").sum()),
        "n_scn1a_with_same_cid_non_scn1a_analog": int(
            analogy["has_same_cid_non_scn1a_analog"].sum()
        ),
        "n_scn1a_with_mixed_same_cid_non_scn1a_analogs": int(
            analogy["has_mixed_same_cid_non_scn1a_analogs"].sum()
        ),
        "n_same_gene_same_position_discordant_pairs": int(len(same_position_pairs)),
        "n_same_cid_cross_gene_discordant_pairs": int(len(same_cid_pairs)),
        "n_scn1a_same_cid_cross_gene_discordant_pairs": int(len(scn1a_same_cid_pairs)),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "challenge_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("CHALLENGE QC")
    print("=" * 100)
    print(json.dumps(qc, indent=4))

    print("\n" + "=" * 100)
    print("LEAVE-ONE-GENE PERFORMANCE")
    print("=" * 100)
    if len(gene_metrics) > 0:
        print(
            gene_metrics.sort_values(
                ["heldout_gene", "roc_auc"],
                ascending=[True, False],
            ).to_string(index=False)
        )
    else:
        print("No leave-one-gene metrics available.")

    print("\n" + "=" * 100)
    print("SCN1A PARALOG ANALOGY METRICS")
    print("=" * 100)
    if len(analogy_metrics) > 0:
        print(
            analogy_metrics.sort_values(
                ["subset", "roc_auc"],
                ascending=[True, False],
            ).to_string(index=False)
        )
    else:
        print("No SCN1A analogy metrics available.")

    print("\n" + "=" * 100)
    print("PAIRWISE CHALLENGE METRICS")
    print("=" * 100)
    print(
        pairwise_metrics.sort_values(
            ["challenge", "pairwise_accuracy"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()