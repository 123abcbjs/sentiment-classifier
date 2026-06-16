from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent

TEXT = "画面清晰，色彩也很舒服"
MODEL_NAME = "baseline"


def predict_baseline(text):
    # 1.加载 TF-IDF 模型
    path = ROOT / "models" / "baseline" / "model.joblib"
    if not path.exists():
        raise FileNotFoundError("请先运行 train_baseline.py。")

    model = joblib.load(path)

    # 2.预测概率
    y_proba = model.predict_proba([text])[0]
    result = {}
    for label, score in zip(model.classes_, y_proba):
        result[label] = score
    return result


def predict_bert(text):
    # 1.加载 BERT 模型
    path = ROOT / "models" / "bert" / "best_model"
    if not path.exists():
        raise FileNotFoundError("请先运行 train_bert.py。")

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    # 2.编码文本并预测
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        y_proba = torch.softmax(logits, dim=0).tolist()

    # 3.整理成 {标签: 概率}
    result = {}
    for index, score in enumerate(y_proba):
        result[model.config.id2label[index]] = score
    return result



def main():
    # 1.读取固定的预测文本和模型类型
    text = TEXT
    model_name = MODEL_NAME

    if model_name not in ["baseline", "bert"]:
        print("MODEL_NAME 只能写 baseline 或 bert")
        return 1

    # 2.选择模型预测
    try:
        if model_name == "baseline":
            scores = predict_baseline(text)
        else:
            scores = predict_bert(text)
    except FileNotFoundError as e:
        print(f"错误：{e}")
        return 1

    # 3.按照概率从高到低打印
    for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        print(f"{label:10s} {score:.4f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
