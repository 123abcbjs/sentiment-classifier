import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "generated_comments.jsonl"
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
LABELS = ["positive", "neutral", "negative", "sarcasm"]
ASPECTS = ["剧情", "画质", "配音", "节奏", "角色"]


def parse_args():
    parser = argparse.ArgumentParser(description="清洗、检查并划分评论数据")
    parser.add_argument("--review-size", type=int, default=200)
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    text = re.sub(r"\s+", "", text)
    return text.strip("，。！？,.!? ")


def assert_no_overlap(*frames: pd.DataFrame):
    sets = [set(frame["text"]) for frame in frames]
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            if sets[left] & sets[right]:
                raise ValueError("数据划分之间存在重复文本。")


def main() -> int:
    args = parse_args()
    if not RAW_FILE.exists():
        print(f"错误：找不到原始数据 {RAW_FILE}")
        return 1

    rows = []
    invalid_json = 0
    for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_json += 1
    raw_count = len(rows)
    df = pd.DataFrame(rows)
    required = {"text", "label", "aspect"}
    if not required.issubset(df.columns):
        print(f"错误：数据字段必须包含 {sorted(required)}")
        return 1

    df["text"] = df["text"].map(clean_text)
    df = df[
        df["label"].isin(LABELS)
        & df["aspect"].isin(ASPECTS)
        & df["text"].str.len().between(6, 60)
    ].copy()
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))

    counts = df.groupby(["label", "aspect"]).size()
    missing = [
        f"{label}/{aspect}"
        for label in LABELS
        for aspect in ASPECTS
        if counts.get((label, aspect), 0) == 0
    ]
    if missing:
        print(f"错误：以下组合没有数据：{', '.join(missing)}")
        return 1

    DATA_DIR.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "comments.csv", index=False, encoding="utf-8-sig")
    review_size = min(args.review_size, len(df))
    df.sample(review_size, random_state=42).to_csv(
        DATA_DIR / "review_sample.csv", index=False, encoding="utf-8-sig"
    )

    train, temp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    valid, test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["label"]
    )
    assert_no_overlap(train, valid, test)
    for name, frame in (("train", train), ("valid", valid), ("test", test)):
        frame.to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    report = {
        "raw_rows": raw_count,
        "invalid_json_lines": invalid_json,
        "removed_by_validation": raw_count - before_dedup,
        "duplicates_removed": before_dedup - len(df),
        "final_rows": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "aspect_distribution": df["aspect"].value_counts().to_dict(),
        "text_length": df["text"].str.len().describe().round(2).to_dict(),
        "split_sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
    }
    (DATA_DIR / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
