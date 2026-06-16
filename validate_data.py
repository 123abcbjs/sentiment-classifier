import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "generated_comments.jsonl"
DATA_DIR = ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"

label_list = ["positive", "neutral", "negative", "sarcasm"]
aspect_list = ["剧情", "画质", "配音", "节奏", "角色"]


# 人工抽查样本数量，想改就直接改这里。
REVIEW_SIZE = 200


def delete_url(text):
    # 不用正则，手动从字符串中删除 http、https、www 开头的网址。
    text = str(text)
    result = ""
    i = 0
    stop_chars = " \t\r\n，。！？；;、"

    while i < len(text):
        if text.startswith("http://", i) or text.startswith("https://", i) or text.startswith("www.", i):
            while i < len(text) and text[i] not in stop_chars:
                i += 1
            continue
        result += text[i]
        i += 1

    return result


def clean_text(text):
    # 1.去网址
    text = delete_url(text)
    # 2.去掉所有空白字符
    new_text = ""
    for ch in text:
        if not ch.isspace():
            new_text += ch
    # 3.去掉开头结尾常见标点
    new_text = new_text.strip("，。！？,.!? ")
    return new_text


def assert_no_overlap(*frames):
    # 检查训练集、验证集、测试集之间不能有重复文本。
    set_list = []
    for frame in frames:
        set_list.append(set(frame["text"]))

    for left in range(len(set_list)):
        for right in range(left + 1, len(set_list)):
            if set_list[left] & set_list[right]:
                raise ValueError("数据划分之间存在重复文本。")


def main():
    # 1.读取参数和原始数据
    if not RAW_FILE.exists():
        print(f"错误：找不到原始数据 {RAW_FILE}")
        return 1

    rows = []
    invalid_json = 0
    lines = RAW_FILE.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            invalid_json += 1

    raw_count = len(rows)
    data = pd.DataFrame(rows)

    # 2.检查字段
    required = {"text", "label", "aspect"}
    if not required.issubset(data.columns):
        print(f"错误：数据字段必须包含 {sorted(required)}")
        return 1

    # 3.清洗文本、过滤非法数据
    data["text"] = data["text"].map(clean_text)
    data = data[
        data["label"].isin(label_list)
        & data["aspect"].isin(aspect_list)
        & data["text"].str.len().between(6, 60)
    ].copy()

    before_dedup = len(data)
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    data.insert(0, "id", range(1, len(data) + 1))

    # 4.检查每个组合是否都有数据
    counts = data.groupby(["label", "aspect"]).size()
    missing = []
    for label in label_list:
        for aspect in aspect_list:
            if counts.get((label, aspect), 0) == 0:
                missing.append(f"{label}/{aspect}")

    if missing:
        print(f"错误：以下组合没有数据：{', '.join(missing)}")
        return 1

    # 5.保存完整数据和人工抽查样本
    DATA_DIR.mkdir(exist_ok=True)
    SPLIT_DIR.mkdir(exist_ok=True)
    data.to_csv(DATA_DIR / "comments.csv", index=False, encoding="utf-8-sig")

    review_size = min(REVIEW_SIZE, len(data))
    data.sample(review_size, random_state=42).to_csv(
        DATA_DIR / "review_sample.csv", index=False, encoding="utf-8-sig"
    )

    # 6.划分训练集、验证集、测试集
    df_train, temp = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["label"]
    )
    df_valid, df_test = train_test_split(
        temp, test_size=0.5, random_state=42, stratify=temp["label"]
    )

    assert_no_overlap(df_train, df_valid, df_test)

    split_data = (("train", df_train), ("valid", df_valid), ("test", df_test))
    for name, frame in split_data:
        frame.to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")

    # 7.保存质量报告
    report = {
        "raw_rows": raw_count,
        "invalid_json_lines": invalid_json,
        "removed_by_validation": raw_count - before_dedup,
        "duplicates_removed": before_dedup - len(data),
        "final_rows": len(data),
        "label_distribution": data["label"].value_counts().to_dict(),
        "aspect_distribution": data["aspect"].value_counts().to_dict(),
        "text_length": data["text"].str.len().describe().round(2).to_dict(),
        "split_sizes": {"train": len(df_train), "valid": len(df_valid), "test": len(df_test)},
    }

    (DATA_DIR / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

