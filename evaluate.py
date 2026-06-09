import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


LABELS = ["positive", "neutral", "negative", "sarcasm"]


def save_evaluation(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    output_dir: Path,
    probabilities=None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    errors = pd.DataFrame({"text": texts, "true_label": y_true, "pred_label": y_pred})
    if probabilities is not None:
        errors["confidence"] = probabilities.max(axis=1)
    errors[errors["true_label"] != errors["pred_label"]].to_csv(
        output_dir / "error_samples.csv", index=False, encoding="utf-8-sig"
    )

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
    return metrics
