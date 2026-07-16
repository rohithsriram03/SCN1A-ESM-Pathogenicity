# SCN1A Protein Language Model Pathogenicity Prediction

## Overview

This project investigates whether protein language models (ESM-2) can improve pathogenicity prediction of neurological missense variants in voltage-gated sodium channel genes.

Current focus:
- SCN1A
- ESM-2 embeddings
- Logistic Regression baseline
- Model uncertainty analysis

## Dataset

- ClinVar
- gnomAD
- UniProt
- MaveDB (future)

## Pipeline

ClinVar
↓

Mutant Protein Generation
↓

ESM-2 Embeddings
↓

Logistic Regression
↓

Pathogenicity Prediction
↓

Uncertainty Analysis
↓

Functional Region Statistics

## Current Results

- Accuracy: 75.3%
- ROC-AUC: 0.84

Significant uncertainty enrichment observed within the Nav1.1 Inactivation Gate (Kruskal-Wallis + Dunn's Test).

## Run dashboard

Train models and inspect them side by side in the browser:

```bash
pip install -e .
python scripts/import_existing_runs.py   # wrap existing baseline + LoRA results as runs
python scripts/dashboard.py              # open http://localhost:8000
```

Each run is a folder under `results/runs/<id>/` (config, metrics, per-epoch
history, and a `predictions.csv` duplicating the test set with the model's
prediction). The dashboard gives a metric comparison, loss/eval curves, a live
decision-threshold slider, a predicted-probability histogram, and a per-variant
table — built to diagnose why LoRA underperforms the frozen-ESM-2 baseline (an
operating-point / model-selection failure on top of a mutation-site vs
whole-window feature-locality ceiling; see [CLAUDE.md](CLAUDE.md)). Launch runs
from the sidebar or the CLI:

```bash
python scripts/train_run.py --model baseline
python scripts/train_run.py --model lora --epochs 6 --lr 5e-5 --batch-size 2
```

## Author

Rohith Sriram
