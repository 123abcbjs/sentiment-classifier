from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate import save_evaluation


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "baseline"
OUTPUT_DIR = ROOT / "outputs" / "baseline"


def main() -> int:
    train_path = SPLIT_DIR / "train.csv"
    test_path = SPLIT_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("错误：请先运行 validate_data.py 生成数据划分。")
        return 1

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=30000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
            ),
        ]
    )
    model.fit(train["text"], train["label"])
    predictions = model.predict(test["text"])
    probabilities = model.predict_proba(test["text"])
    metrics = save_evaluation(
        test["text"].tolist(),
        test["label"].tolist(),
        predictions.tolist(),
        OUTPUT_DIR,
        probabilities,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"模型已保存：{MODEL_DIR / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
