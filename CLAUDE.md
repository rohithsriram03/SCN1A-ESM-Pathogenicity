# CLAUDE.md

Guidance for working in this repository. Read it before writing code here.

## What this project is

Predicting pathogenicity of SCN1A missense variants from protein sequence. The
pipeline: map variants to channel domains → build mutant sequences from the
wild-type FASTA → embed with ESM-2 → train classifiers (sklearn baselines and a
LoRA-fine-tuned ESM head) → analyse prediction uncertainty across functional
regions. It is a **research repo**: scripts are experiments, `results/` holds
their outputs, and reproducibility matters more than abstraction.

## Environment

```bash
source venv/bin/activate
pip install -e .          # one-time: makes `import scn1a` work from any script
python scripts/<name>.py  # every script is runnable standalone
```

Python 3.14, PyTorch (MPS on this Mac), transformers + peft, sklearn, pandas,
matplotlib/seaborn, rich. Pinned in `requirements.txt`.

## Layout

```
scn1a/            importable helpers — import these, don't re-implement them
  config.py       paths, SEED, model name, label/region constants
  data.py         FASTA read, mutation, sequence windowing, stratified split
  metrics.py      classification_metrics(), prediction_frame()
  plots.py        publication figures (roc, confusion, boxplot, ...)
  obs.py          rich console: section(), timer(), metrics_table(), save_json()
  runs.py         Run — the structured run folder used by the dashboard
  torch_utils.py  get_device(), trainable_parameters()  (import explicitly)
  dashboard.html  the run-inspector single-page app (served by scripts/dashboard.py)
scripts/          one file per experiment/stage, run top-to-bottom
data/             raw inputs (FASTA, variant spreadsheet) — never written to
results/<name>/   each script writes to its own subdirectory
results/runs/     one folder per training run (config/metrics/history/predictions)
```

## How to write code here

Write what a strong ML engineer would: the shortest code that is still obvious.
Density is a feature, but clarity outranks it — never golf a line at the cost of
readability.

**Comments.** Default to none. Code says *what*; a comment earns its place only
by saying *why* when the code cannot. Keep the ones that state a real constraint:

```python
mutation_token_index = mutation_idx + 1   # token 0 is the CLS/start token
```

Delete the ones that restate the code (`# load the data`, `# small test model`,
banner bars like `# ------`). A short docstring on a shared helper is fine; a
paragraph narrating a script is not.

**Names over comments.** `stratified_split`, `Pathogenic_Probability`,
`mutation_site_embeddings` — a precise name removes the need to explain.

**Don't repeat the pipeline's load-bearing choices.** Import them:

```python
# do this
from scn1a import SEED, results_dir, stratified_split, classification_metrics, prediction_frame
from scn1a.data import sequence_window

# not this — a fourth private copy of get_window / the metrics dict / the split
```

`SEED = 42`, `test_size = 0.2`, stratified splits, and the
`True_Label / Predicted_Label / Pathogenic_Probability / Confidence / Uncertainty`
prediction schema are fixed across the project. Use the helpers so every
experiment stays comparable; if a helper is wrong, fix it once in `scn1a/`.

## Conventions every script follows

- **Paths** come from `scn1a.config`. Never hard-code or recompute
  `Path(__file__)...`. Write outputs under `results_dir("my_experiment")`.
- **Determinism.** Use `SEED` everywhere a seed is accepted.
- **Metrics.** Build them with `classification_metrics(y_true, y_prob)` so the
  keys are always `accuracy, precision, recall, f1, roc_auc`.
- **Predictions.** Build the CSV with `prediction_frame(meta, y_true, y_prob)`.
- **Structure.** Small scripts may run top-to-bottom; anything with reusable
  logic gets functions + `def main(): ...` under `if __name__ == "__main__":`.
- **Style.** 4-space indent, `snake_case` functions, `UPPER_CASE` module
  constants, standard-lib → third-party → local import order, type hints on
  shared helpers.

## Observability

Use `scn1a.obs` instead of bare `print` for anything a human reads while a
script runs:

```python
from scn1a import section, timer, metrics_table, save_json, results_dir
from scn1a.metrics import classification_metrics

section("Training baseline")
with timer("fit"):
    model.fit(X_train, y_train)

metrics = classification_metrics(y_test, model.predict_proba(X_test)[:, 1])
metrics_table(metrics)
save_json(metrics, results_dir("baseline_model") / "metrics.json")
```

`tqdm` for loops over variants/batches. Keep stdout scannable — a reader should
see the stages, timings, and final numbers without scrolling through noise.

## Plotting

Every figure goes through `scn1a.plots`: one style, one palette (pathogenic red
/ benign blue), 300-dpi PNGs. Helpers return a matplotlib `Figure`; `save()`
writes and closes it.

```python
from scn1a import plots, results_dir

plots.set_style()   # once per script
out = results_dir("baseline_model")
plots.save(plots.roc(y_test, y_prob, label="logreg"), out / "roc.png")
plots.save(plots.confusion(y_test, y_pred), out / "confusion.png")
plots.save(plots.boxplot_by_group(df, "Uncertainty", "Mapped_Functional_Category",
                                  order=scn1a.config.REGION_ORDER), out / "uncertainty.png")
```

Available: `roc`, `confusion`, `boxplot_by_group`, `metric_bars`,
`embedding_scatter`. Add new plot types to `plots.py` (returning a `Figure`)
rather than hand-rolling `plt` calls in a script.

## Runs & the dashboard

Model experiments are captured as **runs** so they can be compared point-for-point.
`scripts/train_run.py` trains one model and writes `results/runs/<id>/`:

```
config.json      model type, hyperparameters, seed, split sizes
metrics.json     final test metrics
history.json     {train_loss: [...], eval: [...]}  (empty for the baseline)
predictions.csv  the test rows + prediction columns, one row per variant
status.json      live state (queued/running/done/failed) for the dashboard
```

Both model types share one `stratified_split(SEED)`, so `baseline` and `lora`
runs cover the identical 530-variant test set and align by `Variant_ID` — that
alignment is what makes the comparison honest.

```bash
python scripts/train_run.py --model baseline
python scripts/train_run.py --model lora --epochs 6 --lr 5e-5 --batch-size 2
python scripts/import_existing_runs.py     # wrap the legacy results/ outputs as runs
python scripts/dashboard.py                # inspect at http://localhost:8000
```

The dashboard reads run folders and gives a metric comparison, loss/eval curves,
a live **decision-threshold slider**, a probability histogram, and a per-variant
table (filter to false negatives, or to disagreements against a reference run).
It can also launch a training run (spawns `train_run.py`, polls `status.json`).

**Why LoRA underperforms — the working diagnosis this tooling exposes.** Two
separate failures, both visible in the dashboard:
1. *Operating point.* At threshold 0.5 the LoRA head calls 142/205 pathogenic
   variants benign (recall 0.31). Drop the threshold to ~0.13 and recall recovers
   to ~0.96 — the model ranks better than its default cutoff suggests. The
   eval-per-epoch curve shows recall collapsing as training proceeds while AUC
   drifts up, and `load_best_model_at_end` selecting on `roc_auc` locks in that
   bad operating point.
2. *Feature locality (the ceiling).* Even at its best threshold LoRA's F1 (~0.65)
   trails the baseline (0.71) and its AUC (0.74 vs 0.84) is lower. The baseline
   classifies the **mutation-site embedding** — the single mutated residue — while
   the LoRA head pools a ~1000-residue window in which one substitution is a tiny
   fraction of the signal. New experiments should attack these: threshold/model
   selection by F1, mutation-site pooling instead of whole-window, a larger ESM-2.

## When adding an experiment

1. New file `scripts/<verb>_<subject>.py` (e.g. `train_mlp_classifier.py`).
2. Read inputs via `scn1a.config` paths; write to `results_dir("<experiment>")`.
3. Reuse `data`, `metrics`, `plots`, `obs` — reach for a helper before writing a
   loop, and promote genuinely shared logic into `scn1a/`.
4. Leave prior scripts and their `results/` outputs intact; experiments are a
   record, not something to overwrite.
