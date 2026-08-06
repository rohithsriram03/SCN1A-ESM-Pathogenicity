from pathlib import Path
import itertools
import json
import re
import subprocess
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Use the WT+mutant LoRA script because this is the representation we now want to tune.
BASE_SCRIPT = PROJECT_ROOT / "scripts" / "mechanism" / "train_contrastive_lora_esm_wt_mut.py"

TMP_DIR = PROJECT_ROOT / "scripts" / "mechanism" / "_tmp_lora_hyperparam_runs"
SWEEP_DIR = PROJECT_ROOT / "results" / "mechanism" / "lora_wt_mut_hyperparameter_sweep"

TMP_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GRID
# =============================================================================

GRID = {
    "learning_rate": [5e-5, 1e-4, 2e-4],
    "margin": [0.1, 0.2],
    "n_steps": [200, 500],
    "lora_r": [4, 8],
    "batch_size": [1],
    "seed": [42, 7, 123],
}

# To make LoRA alpha proportional to rank.
def lora_alpha_for_rank(r):
    return 2 * r


def make_run_id(config):
    return (
        f"wtmut"
        f"_lr{config['learning_rate']}"
        f"_m{config['margin']}"
        f"_steps{config['n_steps']}"
        f"_r{config['lora_r']}"
        f"_seed{config['seed']}"
    ).replace(".", "p")


def patch_script_for_config(config, run_id):
    src = BASE_SCRIPT.read_text()

    output_dir = SWEEP_DIR / run_id

    replacements = {
        r"RANDOM_SEED\s*=\s*\d+": f"RANDOM_SEED = {config['seed']}",
        r"N_STEPS\s*=\s*\d+": f"N_STEPS = {config['n_steps']}",
        r"LEARNING_RATE\s*=\s*[0-9.eE-]+": f"LEARNING_RATE = {config['learning_rate']}",
        r"MARGIN\s*=\s*[0-9.eE-]+": f"MARGIN = {config['margin']}",
        r"LORA_R\s*=\s*\d+": f"LORA_R = {config['lora_r']}",
        r"LORA_ALPHA\s*=\s*\d+": f"LORA_ALPHA = {lora_alpha_for_rank(config['lora_r'])}",
        r"LORA_DROPOUT\s*=\s*[0-9.eE-]+": f"LORA_DROPOUT = {config['lora_dropout']}",
    }

    for pattern, repl in replacements.items():
        src = re.sub(pattern, repl, src)

    # Force correct project root even if temporary script lives deeper.
    src = re.sub(
        r"PROJECT_ROOT\s*=\s*Path\(__file__\)\.resolve\(\)\.parent\.parent\.parent",
        f'PROJECT_ROOT = Path("{PROJECT_ROOT}")',
        src,
    )

    # Force unique output directory.
    src = re.sub(
        r'OUTPUT_DIR\s*=\s*PROJECT_ROOT\s*/\s*"results"\s*/\s*"mechanism"\s*/\s*"[^"]+"',
        f'OUTPUT_DIR = Path("{output_dir}")',
        src,
    )

    out_script = TMP_DIR / f"{run_id}.py"
    out_script.write_text(src)

    return out_script, output_dir


def run_one_config(config):
    run_id = make_run_id(config)
    output_dir = SWEEP_DIR / run_id
    metrics_file = output_dir / "contrastive_lora_scn1a_knn_metrics.csv"
    log_file = output_dir / "run.log"
    config_file = output_dir / "config.json"

    if metrics_file.exists():
        print(f"Skipping completed run: {run_id}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(config | {"run_id": run_id, "output_dir": str(output_dir)}, f, indent=4)

    script_path, output_dir = patch_script_for_config(config, run_id)

    print("\n" + "=" * 100)
    print("RUNNING:", run_id)
    print("=" * 100)
    print("Output:", output_dir)

    with open(log_file, "w") as log:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
        )

    if proc.returncode != 0:
        print(f"Run failed: {run_id}. Check log: {log_file}")
    else:
        print(f"Run finished: {run_id}")


def build_grid():
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def aggregate_results(configs):
    all_rows = []
    best_rows = []
    manifest_rows = []

    for config in configs:
        run_id = make_run_id(config)
        output_dir = SWEEP_DIR / run_id
        metrics_file = output_dir / "contrastive_lora_scn1a_knn_metrics.csv"
        loss_file = output_dir / "training_loss.csv"
        log_file = output_dir / "run.log"

        status = "complete" if metrics_file.exists() else "missing_or_failed"

        manifest_row = config.copy()
        manifest_row.update(
            {
                "run_id": run_id,
                "status": status,
                "output_dir": str(output_dir),
                "metrics_file": str(metrics_file),
                "loss_file": str(loss_file),
                "log_file": str(log_file),
            }
        )
        manifest_rows.append(manifest_row)

        if not metrics_file.exists():
            continue

        df = pd.read_csv(metrics_file)

        for key, value in config.items():
            df[key] = value

        df["run_id"] = run_id
        all_rows.append(df)

        best = df.sort_values("roc_auc", ascending=False).iloc[0].copy()
        best_rows.append(best)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(SWEEP_DIR / "sweep_manifest.csv", index=False)

    if not all_rows:
        print("No completed runs found.")
        return

    all_df = pd.concat(all_rows, ignore_index=True)
    best_df = pd.DataFrame(best_rows)

    all_df.to_csv(SWEEP_DIR / "all_runs_metrics.csv", index=False)
    best_df.to_csv(SWEEP_DIR / "best_row_by_run.csv", index=False)

    summary = (
        all_df
        .groupby(
            [
                "learning_rate",
                "margin",
                "n_steps",
                "lora_r",
                "lora_dropout",
                "feature_set",
                "k",
            ],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_gof_f1=("gof_f1", "mean"),
            std_gof_f1=("gof_f1", "std"),
            mean_average_precision=("average_precision", "mean"),
            std_average_precision=("average_precision", "std"),
        )
        .sort_values("mean_roc_auc", ascending=False)
    )

    summary.to_csv(SWEEP_DIR / "summary_by_config_feature_k.csv", index=False)

    print("\n" + "=" * 100)
    print("TOP 20 CONFIGS BY MEAN ROC-AUC")
    print("=" * 100)
    print(
        summary.head(20)[
            [
                "learning_rate",
                "margin",
                "n_steps",
                "lora_r",
                "feature_set",
                "k",
                "n_seeds",
                "mean_accuracy",
                "mean_balanced_accuracy",
                "mean_roc_auc",
                "mean_gof_f1",
            ]
        ].to_string(index=False)
    )

    print("\nSaved sweep outputs to:")
    print(SWEEP_DIR)


def main():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Missing base script: {BASE_SCRIPT}")

    configs = build_grid()

    print("Base script:", BASE_SCRIPT)
    print("Sweep directory:", SWEEP_DIR)
    print("Number of runs:", len(configs))

    # Save planned config grid before running anything.
    pd.DataFrame(configs).to_csv(SWEEP_DIR / "planned_grid.csv", index=False)

    for config in configs:
        run_one_config(config)

    aggregate_results(configs)


if __name__ == "__main__":
    main()