import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from evaluate import LABELS, save_evaluation


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "bert"
OUTPUT_DIR = ROOT / "outputs" / "bert"

# BERT 训练参数，想改就直接改这里，不用命令行传参。
MODEL_NAME = "bert-base-chinese"
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 96

label_to_id = {}
id_to_label = {}
for index, label in enumerate(LABELS):
    label_to_id[label] = index
    id_to_label[index] = label


class CommentDataset(Dataset):
    # 自定义数据集，给 Trainer 使用。
    def __init__(self, data, tokenizer, max_length):
        self.encodings = tokenizer(
            data["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = []
        for label in data["label"]:
            self.labels.append(label_to_id[label])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {}
        for key, value in self.encodings.items():
            item[key] = torch.tensor(value[index])
        item["labels"] = torch.tensor(self.labels[index])
        return item


def compute_metrics(result):
    pred_ids = np.argmax(result.predictions, axis=1)
    scores = {
        "accuracy": accuracy_score(result.label_ids, pred_ids),
        "macro_f1": f1_score(result.label_ids, pred_ids, average="macro"),
    }
    return scores


def main():
    # 1.读取参数和检查数据
    train_path = SPLIT_DIR / "train.csv"
    valid_path = SPLIT_DIR / "valid.csv"
    test_path = SPLIT_DIR / "test.csv"
    if not train_path.exists() or not valid_path.exists() or not test_path.exists():
        print("错误：请先运行 validate_data.py 生成数据划分。")
        return 1

    # 2.加载数据
    df_train = pd.read_csv(train_path)
    df_valid = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    print(f"CUDA available: {torch.cuda.is_available()}")

    # 3.加载 tokenizer 和 BERT 模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # 4.构建 Dataset
    train_dataset = CommentDataset(df_train, tokenizer, MAX_LENGTH)
    valid_dataset = CommentDataset(df_valid, tokenizer, MAX_LENGTH)
    test_dataset = CommentDataset(df_test, tokenizer, MAX_LENGTH)

    # 5.训练参数
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
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

    # 6.开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # 7.在测试集上预测
    pred_output = trainer.predict(test_dataset)
    pred_ids = np.argmax(pred_output.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(pred_output.predictions), dim=1).numpy()

    y_true = []
    for index in pred_output.label_ids:
        y_true.append(id_to_label[index])

    y_pred = []
    for index in pred_ids:
        y_pred.append(id_to_label[index])

    result = save_evaluation(
        df_test["text"].tolist(),
        y_true,
        y_pred,
        OUTPUT_DIR,
        y_proba,
    )

    # 8.保存模型和标签映射
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR / "best_model")
    tokenizer.save_pretrained(MODEL_DIR / "best_model")
    (MODEL_DIR / "label_mapping.json").write_text(
        json.dumps({"label_to_id": label_to_id}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
