from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluate import save_evaluation


ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "baseline"
OUTPUT_DIR = ROOT / "outputs" / "baseline"


def main():
    # 1.检查数据划分文件
    train_path = SPLIT_DIR / "train.csv"
    test_path = SPLIT_DIR / "test.csv"
    if not train_path.exists() or not test_path.exists():
        print("错误：请先运行 validate_data.py 生成数据划分。")
        return 1

    # 2.加载训练集和测试集
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 3.创建 TF-IDF + 逻辑回归
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
                LogisticRegression(
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    # 4.训练和预测
    model.fit(df_train["text"], df_train["label"])
    y_pred = model.predict(df_test["text"])
    y_proba = model.predict_proba(df_test["text"])

    # 5.保存评估结果
    result = save_evaluation(
        df_test["text"].tolist(),
        df_test["label"].tolist(),
        y_pred.tolist(),
        OUTPUT_DIR,
        y_proba,
    )

    # 6.保存模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(f"模型已保存：{MODEL_DIR / 'model.joblib'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
