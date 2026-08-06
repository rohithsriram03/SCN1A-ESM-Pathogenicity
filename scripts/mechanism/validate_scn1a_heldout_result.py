from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_esm2_650M_embeddings"
META_FILE = EMBED_DIR / "scion_embedding_metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "scn1a_heldout_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_FILES = {
    "biochem_only": None,
    "wt_site": EMBED_DIR / "wt_site_embeddings.npy",
    "mutant_site": EMBED_DIR / "mutant_site_embeddings.npy",
    "delta_site": EMBED_DIR / "delta_site_embeddings.npy",
    "wt_mut_site": EMBED_DIR / "wt_mut_site_features.npy",
    "full_site": EMBED_DIR / "full_site_features.npy",
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
        ref = row["Ref_AA"]
        alt = row["Alt_AA"]

        ref_h = HYDROPATHY.get(ref, 0)
        alt_h = HYDROPATHY.get(alt, 0)

        ref_w = MOLECULAR_WEIGHT.get(ref, 0)
        alt_w = MOLECULAR_WEIGHT.get(alt, 0)

        ref_c = CHARGE.get(ref, 0)
        alt_c = CHARGE.get(alt, 0)

        features = [
            ref_h, alt_h, alt_h - ref_h, abs(alt_h - ref_h),
            ref_w, alt_w, alt_w - ref_w, abs(alt_w - ref_w),
            ref_c, alt_c, alt_c - ref_c, abs(alt_c - ref_c),
            int(ref in POLAR), int(alt in POLAR), int(ref in POLAR) != int(alt in POLAR),
            int(ref in AROMATIC), int(alt in AROMATIC), int(ref in AROMATIC) != int(alt in AROMATIC),
            int(ref == "G"), int(alt == "G"),
            int(ref == "P"), int(alt == "P"),
        ]

        features.extend(aa_one_hot(ref))
        features.extend(aa_one_hot(alt))

        rows.append(features)

    return np.array(rows, dtype=float)


def make_model(X_train):
    steps = [("scaler", StandardScaler())]

    if X_train.shape[1] > 80:
        n_components = min(50, X_train.shape[0] - 2, X_train.shape[1])
        steps.append(("pca", PCA(n_components=n_components, random_state=0, svd_solver="randomized")))

    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                solver="liblinear",
                random_state=0,
            ),
        )
    )

    return Pipeline(steps)


def basic_metrics(y, scores):
    preds = (scores >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores),
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro", zero_division=0),
        "gof_f1": f1_score(y, preds, zero_division=0),
    }


def bootstrap_auc_ci(y, scores, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    values = []

    y = np.array(y)
    scores = np.array(scores)

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], scores[idx]))

    values = np.array(values)

    return {
        "auc_boot_mean": float(np.mean(values)),
        "auc_ci_low": float(np.percentile(values, 2.5)),
        "auc_ci_high": float(np.percentile(values, 97.5)),
    }


def bootstrap_auc_difference_ci(y, scores_a, scores_b, n_boot=10000, seed=1):
    rng = np.random.default_rng(seed)
    diffs = []

    y = np.array(y)
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue

        auc_a = roc_auc_score(y[idx], scores_a[idx])
        auc_b = roc_auc_score(y[idx], scores_b[idx])
        diffs.append(auc_a - auc_b)

    diffs = np.array(diffs)

    return {
        "auc_difference_vs_biochem": float(np.mean(diffs)),
        "auc_difference_ci_low": float(np.percentile(diffs, 2.5)),
        "auc_difference_ci_high": float(np.percentile(diffs, 97.5)),
        "bootstrap_fraction_diff_leq_0": float(np.mean(diffs <= 0)),
    }


def permutation_auc_pvalue(y, scores, n_perm=10000, seed=2):
    rng = np.random.default_rng(seed)
    observed = roc_auc_score(y, scores)
    null = []

    y = np.array(y)
    scores = np.array(scores)

    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        null.append(roc_auc_score(y_perm, scores))

    null = np.array(null)

    p_value = (np.sum(null >= observed) + 1) / (len(null) + 1)

    return {
        "permutation_p_auc_greater_than_random": float(p_value),
    }


def main():
    meta = pd.read_csv(META_FILE)

    y = meta["Mechanism_Binary"].astype(int).values

    train_idx = np.where(meta["Gene"].values != "SCN1A")[0]
    test_idx = np.where(meta["Gene"].values == "SCN1A")[0]

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("SCN1A held-out test:")
    print("Train rows:", len(train_idx))
    print("Test rows:", len(test_idx))
    print("SCN1A LOF:", int((y_test == 0).sum()))
    print("SCN1A GOF:", int((y_test == 1).sum()))

    feature_scores = {}
    prediction_table = meta.iloc[test_idx].copy()

    rows = []

    for feature_name, path in FEATURE_FILES.items():
        print("\nRunning:", feature_name)

        if feature_name == "biochem_only":
            X = build_biochem_features(meta)
        else:
            X = np.load(path)

        X_train = X[train_idx]
        X_test = X[test_idx]

        model = make_model(X_train)
        model.fit(X_train, y_train)

        scores = model.predict_proba(X_test)[:, 1]
        feature_scores[feature_name] = scores

        metrics = basic_metrics(y_test, scores)
        metrics.update(bootstrap_auc_ci(y_test, scores))
        metrics.update(permutation_auc_pvalue(y_test, scores))

        metrics["feature_set"] = feature_name
        rows.append(metrics)

        prediction_table[f"{feature_name}_gof_score"] = scores
        prediction_table[f"{feature_name}_predicted_label"] = np.where(scores >= 0.5, "GOF", "LOF")

    # Compare each feature set to biochem-only
    biochem_scores = feature_scores["biochem_only"]

    for row in rows:
        feature_name = row["feature_set"]

        if feature_name == "biochem_only":
            row["auc_difference_vs_biochem"] = 0.0
            row["auc_difference_ci_low"] = 0.0
            row["auc_difference_ci_high"] = 0.0
            row["bootstrap_fraction_diff_leq_0"] = 1.0
        else:
            diff_stats = bootstrap_auc_difference_ci(
                y_test,
                feature_scores[feature_name],
                biochem_scores,
            )
            row.update(diff_stats)

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)

    results.to_csv(OUTPUT_DIR / "scn1a_heldout_validation_metrics.csv", index=False)
    prediction_table.to_csv(OUTPUT_DIR / "scn1a_heldout_predictions.csv", index=False)

    with open(OUTPUT_DIR / "scn1a_heldout_validation_qc.json", "w") as f:
        json.dump(
            {
                "train_rows_non_scn1a": int(len(train_idx)),
                "test_rows_scn1a": int(len(test_idx)),
                "scn1a_lof": int((y_test == 0).sum()),
                "scn1a_gof": int((y_test == 1).sum()),
                "output_dir": str(OUTPUT_DIR),
            },
            f,
            indent=4,
        )

    print("\n" + "=" * 100)
    print("SCN1A HELD-OUT VALIDATION")
    print("=" * 100)
    print(results.to_string(index=False))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()