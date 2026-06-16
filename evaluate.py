import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABELS = ["positive", "neutral", "negative", "sarcasm"]


def save_evaluation(texts, y_true, y_pred, output_dir, probabilities=None):
    # 1.创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2.计算分类指标
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3.保存预测错误的样本，方便后面分析
    error_data = pd.DataFrame({"text": texts, "true_label": y_true, "pred_label": y_pred})
    if probabilities is not None:
        error_data["confidence"] = probabilities.max(axis=1)

    error_data[error_data["true_label"] != error_data["pred_label"]].to_csv(
        output_dir / "error_samples.csv", index=False, encoding="utf-8-sig"
    )

    # 4.画混淆矩阵
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )

    for row in range(len(LABELS)):
        for col in range(len(LABELS)):
            ax.text(col, row, matrix[row, col], ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    return result
