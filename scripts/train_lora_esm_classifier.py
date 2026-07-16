from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from transformers import AutoTokenizer, EsmForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "results" / "SCN1A_mutant_sequences.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "lora_esm_classifier"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_AA_LENGTH = 1000


def get_window(sequence, position, max_len=1000):
    pos0 = int(position) - 1
    half = max_len // 2
    start = max(0, pos0 - half)
    end = min(len(sequence), start + max_len)
    start = max(0, end - max_len)
    return sequence[start:end]


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            max_length=1022,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in item.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }


def main():
    df = pd.read_csv(INPUT_CSV)

    df = df[df["Mutation_Status"] == "Success"].copy()
    df = df.dropna(subset=["Mutant_Sequence", "Binary_Label", "AA_Position"])

    df["Window_Sequence"] = df.apply(
        lambda row: get_window(
            row["Mutant_Sequence"],
            row["AA_Position"],
            MAX_AA_LENGTH
        ),
        axis=1
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Binary_Label"]
    )

    class_counts = train_df["Binary_Label"].value_counts().sort_index()
    total = class_counts.sum()

    class_weights = torch.tensor(
        [
            total / (2 * class_counts[0]),
            total / (2 * class_counts[1])
        ],
        dtype=torch.float
    )

    print("Class counts:")
    print(class_counts)
    print("Class weights:", class_weights)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = EsmForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()

    train_dataset = ProteinDataset(
        train_df["Window_Sequence"].tolist(),
        train_df["Binary_Label"].astype(int).tolist(),
        tokenizer
    )

    test_dataset = ProteinDataset(
        test_df["Window_Sequence"].tolist(),
        test_df["Binary_Label"].astype(int).tolist(),
        tokenizer
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=6,
        weight_decay=0.01,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        report_to="none"
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)
    labels = test_df["Binary_Label"].astype(int).values

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs),
    }

    print("\nLoRA ESM metrics:")
    print(metrics)

    results = test_df.copy()
    results["Predicted_Label"] = preds
    results["Pathogenic_Probability"] = probs
    results["Confidence"] = np.maximum(probs, 1 - probs)
    results["Uncertainty"] = 1 - results["Confidence"]

    results.to_csv(OUTPUT_DIR / "lora_predictions.csv", index=False)

    with open(OUTPUT_DIR / "lora_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSaved LoRA results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()