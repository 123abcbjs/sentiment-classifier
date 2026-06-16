import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "generated_comments.jsonl"

label_list = ["positive", "neutral", "negative", "sarcasm"]
aspect_list = ["剧情", "画质", "配音", "节奏", "角色"]


# 生成数据的参数，想改就直接改这里，不用命令行传参。
PER_COMBINATION = 100
BATCH_SIZE = 20
MAX_RETRIES = 4


def normalize_text(text):
    # 1.转成字符串
    text = str(text)
    # 2.去掉所有空白字符，不使用正则
    new_text = ""
    for ch in text:
        if not ch.isspace():
            new_text += ch
    # 3.去掉开头结尾常见标点
    new_text = new_text.strip("，。！？,.!? ")
    return new_text


def load_existing():
    # 读取已经生成过的数据，方便断点续跑。
    item_list = []
    text_set = set()
    count_dict = Counter()

    if not RAW_FILE.exists():
        return item_list, text_set, count_dict

    lines = RAW_FILE.read_text(encoding="utf-8").splitlines()
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = normalize_text(item.get("text", ""))
        label = item.get("label")
        aspect = item.get("aspect")
        key = (label, aspect)

        if text and text not in text_set and label in label_list and aspect in aspect_list:
            item["text"] = text
            item_list.append(item)
            text_set.add(text)
            count_dict[key] += 1

    return item_list, text_set, count_dict


def build_prompt(label, aspect, amount, seed_words):
    label_meaning = {
        "positive": "明确表达喜欢、认可或满意",
        "neutral": "只陈述客观信息，不表达明显好恶",
        "negative": "明确表达不满、批评或失望",
        "sarcasm": "表面像夸奖，实际表达讽刺或不满",
    }
    prompt = f"""请生成 {amount} 条中文短评论，用于文本分类实验。
所有评论的标签必须是 {label}，含义为：{label_meaning[label]}。
所有评论主要评价对象必须是：{aspect}。

要求：
1. 每条 8-45 个中文字符，像普通用户随手写的评论。
2. 句式和措辞必须多样，不要编号，不要解释标签。
3. 不要出现“小红书”“数据集”“标签”等词。
4. 避免与这些提示词形成固定模板：{seed_words}。
5. 只返回 JSON 对象，格式为：
{{"items":[{{"text":"评论正文","label":"{label}","aspect":"{aspect}"}}]}}
"""
    return prompt


def request_batch(client, label, aspect, amount, seed_words):
    # 调用 DeepSeek 生成一小批评论。
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": build_prompt(label, aspect, amount, seed_words)}],
        temperature=1.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return data.get("items", [])


def main():
    # 1.加载环境变量
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY。")
        return 1

    # 2.读取参数和已有数据
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _, existing_texts, count_dict = load_existing()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    random.seed(42)

    # 3.按照 标签 x 评价对象 生成数据
    with RAW_FILE.open("a", encoding="utf-8") as output:
        for label in label_list:
            for aspect in aspect_list:
                key = (label, aspect)
                fail_count = 0

                while count_dict[key] < PER_COMBINATION:
                    need_num = PER_COMBINATION - count_dict[key]
                    amount = min(BATCH_SIZE, need_num + 5)
                    seed_words = random.sample(sorted(existing_texts), min(3, len(existing_texts)))

                    try:
                        batch = request_batch(client, label, aspect, amount, seed_words)
                    except Exception as e:
                        fail_count += 1
                        print(f"{label}/{aspect} 请求失败（{fail_count}/{MAX_RETRIES}）：{e}")
                        if fail_count >= MAX_RETRIES:
                            return 1
                        time.sleep(2 ** fail_count)
                        continue

                    # 4.清洗这一批结果，去重后写入 jsonl
                    add_num = 0
                    for item in batch:
                        text = normalize_text(item.get("text", ""))
                        if count_dict[key] >= PER_COMBINATION:
                            continue
                        if not (6 <= len(text) <= 60):
                            continue
                        if text in existing_texts:
                            continue

                        clean_item = {"text": text, "label": label, "aspect": aspect}
                        output.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                        output.flush()
                        existing_texts.add(text)
                        count_dict[key] += 1
                        add_num += 1

                    print(f"{label}/{aspect}: +{add_num}, {count_dict[key]}/{PER_COMBINATION}")

                    if add_num == 0:
                        fail_count += 1
                    else:
                        fail_count = 0

                    if fail_count >= MAX_RETRIES:
                        print(f"错误：{label}/{aspect} 连续未生成有效新评论。")
                        return 1

    print(f"生成完成：{RAW_FILE}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
