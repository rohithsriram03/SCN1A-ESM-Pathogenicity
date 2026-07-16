"""Train one model and write a structured run folder the dashboard can read.

    python scripts/train_run.py --model baseline
    python scripts/train_run.py --model lora --epochs 6 --lr 5e-5 --batch-size 2

Both model types share one stratified split (SEED), so their predictions align
by Variant_ID and can be compared row-for-row in the dashboard.

    baseline  logistic regression on the frozen ESM-2 mutation-site embedding
    lora      LoRA-fine-tuned ESM-2 sequence classifier over the residue window
"""
import argparse

import numpy as np
import pandas as pd

from scn1a import config
from scn1a.data import sequence_window, stratified_split
from scn1a.metrics import classification_metrics, prediction_frame
from scn1a.obs import console, metrics_table, section
from scn1a.runs import Run

ID_COLS = ["Variant_ID", "Gene", "AA_Position", "Ref_AA", "Alt_AA",
           "ClinVar_Label", "Mapped_Region", "Mapped_Functional_Category"]


def load_dataset():
    """Metadata frame + aligned mutation-site embeddings + labels.

    embedding_metadata.csv, the .npy matrix, and the Success rows of
    SCN1A_mutant_sequences.csv are all produced by the same filter in the same
    order, so they align positionally. Variant_ID is not unique, so never merge.
    """
    meta = pd.read_csv(config.RESULTS / "embeddings" / "embedding_metadata.csv")
    X = np.load(config.RESULTS / "embeddings" / "mutation_site_embeddings.npy")
    seqs = (pd.read_csv(config.RESULTS / "SCN1A_mutant_sequences.csv")
            .query("Mutation_Status == 'Success'")
            .dropna(subset=["Mutant_Sequence", "Binary_Label", "AA_Position"])
            .reset_index(drop=True))
    assert len(seqs) == len(meta) == len(X), (len(seqs), len(meta), len(X))
    assert (seqs["Variant_ID"].values == meta["Variant_ID"].values).all()

    df = meta.copy()
    df["Mutant_Sequence"] = seqs["Mutant_Sequence"].values
    return df, X, df["Binary_Label"].astype(int).values


def predictions(df, test_idx, y_test, prob) -> pd.DataFrame:
    out = prediction_frame(df.iloc[test_idx][ID_COLS].reset_index(drop=True), y_test, prob)
    out["Correct"] = (out["True_Label"] == out["Predicted_Label"]).astype(int)
    return out


def train_baseline(df, X, y, train_idx, test_idx, params):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")
    model.fit(X[train_idx], y[train_idx])
    prob = model.predict_proba(X[test_idx])[:, 1]
    return prob, {"train_loss": [], "eval": []}


def train_lora(run, df, X, y, train_idx, test_idx, params):
    import torch
    from torch.nn import CrossEntropyLoss
    from transformers import (AutoTokenizer, EsmForSequenceClassification,
                              Trainer, TrainerCallback, TrainingArguments)
    from peft import LoraConfig, TaskType, get_peft_model

    windows = df["Mutant_Sequence"].combine(
        df["AA_Position"], lambda s, p: sequence_window(s, p)[0]).tolist()
    tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, idx):
            self.idx = idx

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            row = self.idx[i]
            item = tokenizer(windows[row], truncation=True, max_length=1022,
                             padding="max_length", return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in item.items()}
            item["labels"] = torch.tensor(int(y[row]), dtype=torch.long)
            return item

    counts = np.bincount(y[train_idx], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            loss = CrossEntropyLoss(weight=weights.to(outputs["logits"].device))(
                outputs["logits"], labels)
            return (loss, outputs) if return_outputs else loss

    class StatusCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            run.set_status("running", f"epoch {int(state.epoch)}/{params['epochs']}",
                           state.epoch / params["epochs"])

    def compute_metrics(pred):
        prob = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:, 1]
        return classification_metrics(pred.label_ids, prob)

    model = get_peft_model(
        EsmForSequenceClassification.from_pretrained(config.ESM_MODEL, num_labels=2),
        LoraConfig(task_type=TaskType.SEQ_CLS, r=params["r"], lora_alpha=params["alpha"],
                   lora_dropout=params["dropout"], target_modules=["query", "value"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(run.dir / "hf"),
        eval_strategy="epoch", save_strategy="no",
        learning_rate=params["lr"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        num_train_epochs=params["epochs"], weight_decay=0.01,
        logging_steps=25, report_to="none")

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=Dataset(train_idx), eval_dataset=Dataset(test_idx),
        compute_metrics=compute_metrics, callbacks=[StatusCallback()])
    run.set_status("running", f"training (0/{params['epochs']} epochs)", 0.0)
    trainer.train()

    logits = trainer.predict(Dataset(test_idx)).predictions
    prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    log = trainer.state.log_history
    history = {
        "train_loss": [{"step": e["step"], "epoch": e["epoch"], "loss": e["loss"]}
                       for e in log if "loss" in e],
        "eval": [{"epoch": e["epoch"], "loss": e["eval_loss"],
                  **{k[5:]: e[k] for k in e if k.startswith("eval_") and k != "eval_loss"}}
                 for e in log if "eval_loss" in e],
    }
    return prob, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "lora"], required=True)
    p.add_argument("--run-id", default=None, help="reuse an existing (pre-created) run id")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()

    params = {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
              "r": args.r, "alpha": args.alpha, "dropout": args.dropout}

    run = Run(args.run_id) if args.run_id else Run.create(args.model)
    run.dir.mkdir(parents=True, exist_ok=True)
    run.set_status("running", "loading data")

    try:
        section(f"Run {run.id} — {args.model}")
        df, X, y = load_dataset()
        train_idx, test_idx = stratified_split(np.arange(len(y)), labels=y)
        y_test = y[test_idx]

        run.write_json("config.json", {
            "run_id": run.id, "model_type": args.model, "seed": config.SEED,
            "feature": "mutation_site_embedding" if args.model == "baseline" else "residue_window",
            "params": params if args.model == "lora" else {},
            "n_train": len(train_idx), "n_test": len(test_idx),
        })

        prob, history = (train_baseline(df, X, y, train_idx, test_idx, params)
                         if args.model == "baseline"
                         else train_lora(run, df, X, y, train_idx, test_idx, params))

        metrics = classification_metrics(y_test, prob)
        run.write_json("metrics.json", metrics)
        run.write_json("history.json", history)
        run.write_predictions(predictions(df, test_idx, y_test, prob))

        metrics_table(metrics, title=f"{run.id} test metrics")
        run.set_status("done", "complete", 1.0)
        console.log(f"run written to {run.dir}")
    except Exception as exc:
        run.set_status("failed", str(exc))
        raise


if __name__ == "__main__":
    main()
