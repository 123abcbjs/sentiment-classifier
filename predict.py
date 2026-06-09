import argparse
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluate import LABELS


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="预测单条中文评论的情感类别")
    parser.add_argument("text", help="需要预测的评论")
    parser.add_argument("--model", choices=["baseline", "bert"], default="baseline")
    return parser.parse_args()


def predict_baseline(text: str):
    path = ROOT / "models" / "baseline" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError("请先运行 train_baseline.py。")
    model = joblib.load(path)
    probabilities = model.predict_proba([text])[0]
    return dict(zip(model.classes_, probabilities))


def predict_bert(text: str):
    path = ROOT / "models" / "bert" / "best_model"
    if not path.exists():
        raise FileNotFoundError("请先运行 train_bert.py。")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=0).tolist()
    return {model.config.id2label[index]: value for index, value in enumerate(probabilities)}


def main() -> int:
    args = parse_args()
    try:
        scores = predict_baseline(args.text) if args.model == "baseline" else predict_bert(args.text)
    except FileNotFoundError as exc:
        print(f"错误：{exc}")
        return 1
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{label:10s} {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
