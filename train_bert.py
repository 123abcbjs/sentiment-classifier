import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from evaluate import LABELS, save_evaluation


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "bert"
OUTPUT_DIR = ROOT / "outputs" / "bert"
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class CommentDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = [LABEL_TO_ID[label] for label in frame["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="微调中文 BERT 四分类模型")
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    return parser.parse_args()


def compute_metrics(result):
    predictions = np.argmax(result.predictions, axis=1)
    return {
        "accuracy": accuracy_score(result.label_ids, predictions),
        "macro_f1": f1_score(result.label_ids, predictions, average="macro"),
    }


def main() -> int:
    args = parse_args()
    required = [SPLIT_DIR / f"{name}.csv" for name in ("train", "valid", "test")]
    if not all(path.exists() for path in required):
        print("错误：请先运行 validate_data.py 生成数据划分。")
        return 1

    train, valid, test = [pd.read_csv(path) for path in required]
    print(f"CUDA available: {torch.cuda.is_available()}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    train_dataset = CommentDataset(train, tokenizer, args.max_length)
    valid_dataset = CommentDataset(valid, tokenizer, args.max_length)
    test_dataset = CommentDataset(test, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        seed=42,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    prediction_output = trainer.predict(test_dataset)
    predicted_ids = np.argmax(prediction_output.predictions, axis=1)
    probabilities = torch.softmax(
        torch.tensor(prediction_output.predictions), dim=1
    ).numpy()
    true_labels = [ID_TO_LABEL[index] for index in prediction_output.label_ids]
    predictions = [ID_TO_LABEL[index] for index in predicted_ids]
    metrics = save_evaluation(
        test["text"].tolist(), true_labels, predictions, OUTPUT_DIR, probabilities
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR / "best_model")
    tokenizer.save_pretrained(MODEL_DIR / "best_model")
    (MODEL_DIR / "label_mapping.json").write_text(
        json.dumps({"label_to_id": LABEL_TO_ID}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
