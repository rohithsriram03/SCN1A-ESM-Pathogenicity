from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SCION_DIR = PROJECT_ROOT / "data" / "mechanism" / "raw" / "scion" / "SCION"

FILES_TO_INSPECT = [
    SCION_DIR / "app" / "training_data.csv",
    SCION_DIR / "data" / "clean_tbl.csv",
    SCION_DIR / "data" / "dat_prep.csv",
    SCION_DIR / "data" / "dat_val.csv",
    SCION_DIR / "data" / "raw_tbl.csv",
    SCION_DIR / "data" / "dat_heyne.csv",
    SCION_DIR / "app" / "phenotype.csv",
    SCION_DIR / "app" / "scnviewer_lookup.csv",
    SCION_DIR / "app" / "cid.csv",
]


def inspect_file(path):
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)

    if not path.exists():
        print("MISSING")
        return

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("Could not read:", e)
        return

    print("Shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print(" -", col)

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nPossible label-like columns:")
    label_keywords = [
        "gof", "lof", "effect", "function", "label", "class",
        "target", "phenotype", "pheno", "y", "task", "mutation",
        "variant", "gene", "protein"
    ]

    for col in df.columns:
        col_lower = col.lower()
        if any(k in col_lower for k in label_keywords):
            print(f"\nColumn: {col}")
            print(df[col].value_counts(dropna=False).head(20).to_string())


def main():
    for path in FILES_TO_INSPECT:
        inspect_file(path)


if __name__ == "__main__":
    main()