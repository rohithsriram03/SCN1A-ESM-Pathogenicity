from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCION_TRAINING_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "raw"
    / "scion"
    / "SCION"
    / "app"
    / "training_data.csv"
)

PROCESSED_ALL_VARIANTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "mechanism"
    / "processed"
    / "scion_mechanism_variants_clean.csv"
)

VALID_METADATA_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "scion_esm2_650M_embeddings"
    / "scion_embedding_metadata.csv"
)

ESM_KNN_METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "mechanism"
    / "embedding_only_paralog_analysis"
    / "scn1a_knn_label_transfer_metrics.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "mechanism" / "scion_feature_baselines"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
GLOBAL_KNN_K_VALUES = [1, 3, 5]
N_PERMUTATIONS = 5000


def load_and_align_data():
    print("Reading files:")
    print("SCION features:", SCION_TRAINING_FEATURES_FILE)
    print("All processed variants:", PROCESSED_ALL_VARIANTS_FILE)
    print("Valid metadata:", VALID_METADATA_FILE)

    scion_features = pd.read_csv(SCION_TRAINING_FEATURES_FILE).reset_index(drop=True)
    all_variants = pd.read_csv(PROCESSED_ALL_VARIANTS_FILE).reset_index(drop=True)
    valid_meta = pd.read_csv(VALID_METADATA_FILE).reset_index(drop=True)

    if len(scion_features) != len(all_variants):
        raise ValueError(
            f"Row mismatch: SCION features has {len(scion_features)} rows, "
            f"but processed variants has {len(all_variants)} rows. "
            "These are expected to be row-aligned."
        )

    scion_features["__row_id__"] = np.arange(len(scion_features))
    all_variants["__row_id__"] = np.arange(len(all_variants))

    if "SCION_ID" in all_variants.columns and "SCION_ID" in valid_meta.columns:
        valid_ids = set(valid_meta["SCION_ID"].astype(str))
        valid_mask = all_variants["SCION_ID"].astype(str).isin(valid_ids)
        match_key = "SCION_ID"

    elif "Variant_Key" in all_variants.columns and "Variant_Key" in valid_meta.columns:
        valid_ids = set(valid_meta["Variant_Key"].astype(str))
        valid_mask = all_variants["Variant_Key"].astype(str).isin(valid_ids)
        match_key = "Variant_Key"

    else:
        raise ValueError(
            "Could not find SCION_ID or Variant_Key in both processed variants and valid metadata."
        )

    meta = all_variants.loc[valid_mask].copy().reset_index(drop=True)

    if len(meta) != len(valid_meta):
        print("\nWARNING:")
        print(f"Matched {len(meta)} valid rows using {match_key}, but valid metadata has {len(valid_meta)} rows.")
        print("This may be okay if duplicated keys exist, but check carefully.")

    row_ids = meta["__row_id__"].values

    exclude_cols = {
        "__row_id__",
        "gene",
        "Gene",
        "y",
        "Y",
        "label",
        "Label",
        "Mechanism_Label",
        "Mechanism_Binary",
        "functional_label",
        "Functional_Label",
    }

    candidate_feature_cols = [
        col for col in scion_features.columns
        if col not in exclude_cols
    ]

    X_raw = scion_features.loc[row_ids, candidate_feature_cols].reset_index(drop=True)

    # Convert everything possible to numeric. Non-numeric columns become NaN and will be dropped if fully NaN.
    X = X_raw.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are completely missing after numeric conversion.
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        print("\nDropping all-NaN/non-numeric feature columns:")
        print(all_nan_cols)
        X = X.drop(columns=all_nan_cols)

    # Drop columns with only one unique non-null value.
    constant_cols = []
    for col in X.columns:
        vals = X[col].dropna().unique()
        if len(vals) <= 1:
            constant_cols.append(col)

    if constant_cols:
        print("\nDropping constant feature columns:")
        print(constant_cols)
        X = X.drop(columns=constant_cols)

    if "Mechanism_Binary" not in meta.columns:
        meta["Mechanism_Binary"] = meta["Mechanism_Label"].map({"LOF": 0, "GOF": 1})

    meta["Mechanism_Binary"] = meta["Mechanism_Binary"].astype(int)

    print("\nAligned dataset:")
    print("Rows:", len(meta))
    print("Feature columns:", X.shape[1])
    print("\nMechanism counts:")
    print(meta["Mechanism_Label"].value_counts().to_string())
    print("\nGene × mechanism:")
    print(pd.crosstab(meta["Gene"], meta["Mechanism_Label"]).to_string())

    feature_col_file = OUTPUT_DIR / "scion_feature_columns_used.csv"
    pd.DataFrame({"feature_column": X.columns}).to_csv(feature_col_file, index=False)

    return meta, X


def make_model(model_name, n_train):
    if model_name == "majority_class":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    if model_name == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if model_name == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.04,
                        l2_regularization=1.0,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

    if model_name.startswith("scion_feature_knn_k"):
        k = int(model_name.replace("scion_feature_knn_k", ""))
        k = min(k, n_train)
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("variance", VarianceThreshold()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=k,
                        weights="uniform",
                        metric="minkowski",
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model name: {model_name}")


MODEL_NAMES = [
    "majority_class",
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "scion_feature_knn_k1",
    "scion_feature_knn_k3",
    "scion_feature_knn_k5",
]


def get_positive_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        classes = list(model.classes_)

        if 1 in classes:
            pos_idx = classes.index(1)
        else:
            pos_idx = -1

        return probs[:, pos_idx]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return scores

    return model.predict(X_test).astype(float)


def evaluate_predictions(y_true, y_pred, y_score):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    y_score = np.array(y_score).astype(float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "n": int(len(y_true)),
        "n_lof": int((y_true == 0).sum()),
        "n_gof": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "gof_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "gof_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "gof_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = np.nan

        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["average_precision"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["average_precision"] = np.nan

    return metrics


def fit_and_evaluate_model(meta, X, train_idx, test_idx, model_name, experiment, heldout_gene=None):
    X_train = X.iloc[train_idx].values
    X_test = X.iloc[test_idx].values

    y_train = meta.iloc[train_idx]["Mechanism_Binary"].astype(int).values
    y_test = meta.iloc[test_idx]["Mechanism_Binary"].astype(int).values

    if len(np.unique(y_train)) < 2:
        return {
            "experiment": experiment,
            "heldout_gene": heldout_gene,
            "model": model_name,
            "error": "Training split has only one class.",
        }

    model = make_model(model_name, n_train=len(train_idx))

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = get_positive_scores(model, X_test)

        metrics = evaluate_predictions(y_test, y_pred, y_score)

        metrics.update(
            {
                "experiment": experiment,
                "heldout_gene": heldout_gene,
                "model": model_name,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_lof": int((y_train == 0).sum()),
                "train_gof": int((y_train == 1).sum()),
                "error": "",
            }
        )

        return metrics

    except Exception as e:
        return {
            "experiment": experiment,
            "heldout_gene": heldout_gene,
            "model": model_name,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "error": str(e),
        }


def run_heldout_scn1a(meta, X):
    train_idx = meta.index[meta["Gene"] != "SCN1A"].tolist()
    test_idx = meta.index[meta["Gene"] == "SCN1A"].tolist()

    rows = []

    for model_name in MODEL_NAMES:
        rows.append(
            fit_and_evaluate_model(
                meta,
                X,
                train_idx,
                test_idx,
                model_name=model_name,
                experiment="heldout_scn1a",
                heldout_gene="SCN1A",
            )
        )

    return pd.DataFrame(rows)


def run_leave_one_gene_out(meta, X):
    rows = []

    for gene in sorted(meta["Gene"].unique()):
        train_idx = meta.index[meta["Gene"] != gene].tolist()
        test_idx = meta.index[meta["Gene"] == gene].tolist()

        for model_name in MODEL_NAMES:
            rows.append(
                fit_and_evaluate_model(
                    meta,
                    X,
                    train_idx,
                    test_idx,
                    model_name=model_name,
                    experiment="leave_one_gene_out",
                    heldout_gene=gene,
                )
            )

    return pd.DataFrame(rows)


def summarize_logo(logo_df):
    usable = logo_df[
        (logo_df["error"].fillna("") == "")
        & (logo_df["roc_auc"].notna())
    ].copy()

    summary_rows = []

    for model, group in usable.groupby("model"):
        summary_rows.append(
            {
                "model": model,
                "n_gene_tests": int(group["heldout_gene"].nunique()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "mean_macro_f1": float(group["macro_f1"].mean()),
                "mean_gof_f1": float(group["gof_f1"].mean()),
                "mean_roc_auc": float(group["roc_auc"].mean()),
                "mean_average_precision": float(group["average_precision"].mean()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("mean_roc_auc", ascending=False)


def preprocess_features_for_geometry(X):
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
        ]
    )

    X_proc = pipe.fit_transform(X.values)

    norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return X_proc / norms


def run_global_cross_gene_feature_knn(meta, X, k):
    """
    This is the engineered-feature equivalent of the ESM global cross-gene KNN test.
    It asks: in SCION engineered-feature space, is the closest cross-gene neighbor
    likely to have the same GoF/LoF label?
    """
    X_norm = preprocess_features_for_geometry(X)
    y = meta["Mechanism_Binary"].astype(int).values
    genes = meta["Gene"].astype(str).values

    same_rates = []
    baseline_rates = []

    for i in range(len(meta)):
        candidate_idx = np.where(genes != genes[i])[0]

        sims = X_norm[[i]] @ X_norm[candidate_idx].T
        sims = sims.ravel()

        order = np.argsort(-sims)
        top_idx = candidate_idx[order[: min(k, len(order))]]

        same_rates.append(float(np.mean(y[top_idx] == y[i])))
        baseline_rates.append(float(np.mean(y[candidate_idx] == y[i])))

    observed = float(np.mean(same_rates))
    baseline = float(np.mean(baseline_rates))
    lift = observed / baseline if baseline > 0 else np.nan

    # Permutation test: shuffle labels globally while keeping feature geometry fixed.
    rng = np.random.default_rng(RANDOM_SEED)
    null_values = []

    for _ in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        perm_rates = []

        for i in range(len(meta)):
            candidate_idx = np.where(genes != genes[i])[0]

            sims = X_norm[[i]] @ X_norm[candidate_idx].T
            sims = sims.ravel()

            order = np.argsort(-sims)
            top_idx = candidate_idx[order[: min(k, len(order))]]

            perm_rates.append(float(np.mean(y_perm[top_idx] == y_perm[i])))

        null_values.append(float(np.mean(perm_rates)))

    null_values = np.array(null_values)
    p_value = (np.sum(null_values >= observed) + 1) / (len(null_values) + 1)

    return {
        "feature_space": "scion_engineered_features",
        "k": int(k),
        "n": int(len(meta)),
        "same_mechanism_rate": observed,
        "cross_gene_random_baseline": baseline,
        "lift_over_baseline": lift,
        "permutation_null_mean": float(np.mean(null_values)),
        "permutation_null_std": float(np.std(null_values)),
        "permutation_p_value": float(p_value),
    }


def load_existing_esm_knn_for_comparison():
    if not ESM_KNN_METRICS_FILE.exists():
        return pd.DataFrame()

    esm = pd.read_csv(ESM_KNN_METRICS_FILE)

    keep = esm[
        (esm["candidate_mode"] == "all_non_scn1a")
        & (esm["feature_set"].isin(["wt_mut_site", "mutant_site", "wt_site", "delta_site"]))
    ].copy()

    if len(keep) == 0:
        return pd.DataFrame()

    keep = keep.rename(columns={"feature_set": "model"})
    keep["experiment"] = "heldout_scn1a_existing_esm_knn"
    keep["heldout_gene"] = "SCN1A"
    keep["model"] = "esm_knn_" + keep["model"].astype(str) + "_k" + keep["k"].astype(str)

    cols = [
        "experiment",
        "heldout_gene",
        "model",
        "n",
        "n_lof",
        "n_gof",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "gof_f1",
        "roc_auc",
        "average_precision",
    ]

    cols = [c for c in cols if c in keep.columns]
    return keep[cols].copy()


def main():
    meta, X = load_and_align_data()

    qc = {
        "n_variants": int(len(meta)),
        "mechanism_counts": meta["Mechanism_Label"].value_counts().to_dict(),
        "gene_counts": meta["Gene"].value_counts().to_dict(),
        "n_features_used": int(X.shape[1]),
        "input_scion_training_features_file": str(SCION_TRAINING_FEATURES_FILE),
        "input_processed_all_variants_file": str(PROCESSED_ALL_VARIANTS_FILE),
        "input_valid_metadata_file": str(VALID_METADATA_FILE),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(OUTPUT_DIR / "scion_feature_baseline_qc.json", "w") as f:
        json.dump(qc, f, indent=4)

    print("\n" + "=" * 100)
    print("RUNNING HELD-OUT SCN1A BASELINES")
    print("=" * 100)

    heldout_df = run_heldout_scn1a(meta, X)
    heldout_df.to_csv(OUTPUT_DIR / "heldout_scn1a_scion_feature_baselines.csv", index=False)

    print(
        heldout_df.sort_values("roc_auc", ascending=False, na_position="last")
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("RUNNING LEAVE-ONE-GENE-OUT BASELINES")
    print("=" * 100)

    logo_df = run_leave_one_gene_out(meta, X)
    logo_df.to_csv(OUTPUT_DIR / "leave_one_gene_scion_feature_baselines.csv", index=False)

    logo_summary = summarize_logo(logo_df)
    logo_summary.to_csv(OUTPUT_DIR / "leave_one_gene_scion_feature_baseline_summary.csv", index=False)

    print("\nLEAVE-ONE-GENE SUMMARY")
    print(
        logo_summary.sort_values("mean_roc_auc", ascending=False, na_position="last")
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("RUNNING GLOBAL CROSS-GENE SCION-FEATURE KNN GEOMETRY BASELINE")
    print("=" * 100)

    feature_knn_rows = []
    for k in GLOBAL_KNN_K_VALUES:
        feature_knn_rows.append(run_global_cross_gene_feature_knn(meta, X, k=k))

    feature_knn_df = pd.DataFrame(feature_knn_rows)
    feature_knn_df.to_csv(OUTPUT_DIR / "global_cross_gene_scion_feature_knn.csv", index=False)

    print(feature_knn_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("OPTIONAL: EXISTING ESM KNN HELD-OUT SCN1A RESULTS")
    print("=" * 100)

    esm_knn = load_existing_esm_knn_for_comparison()

    if len(esm_knn):
        esm_knn.to_csv(OUTPUT_DIR / "existing_esm_knn_heldout_scn1a_for_comparison.csv", index=False)
        print(
            esm_knn.sort_values("roc_auc", ascending=False, na_position="last")
            .to_string(index=False)
        )
    else:
        print("No existing ESM KNN metrics file found or no matching rows.")

    print("\n" + "=" * 100)
    print("QC")
    print("=" * 100)
    print(json.dumps(qc, indent=4))

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()