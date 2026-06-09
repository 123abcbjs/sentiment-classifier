import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "generated_comments.jsonl"
LABELS = ["positive", "neutral", "negative", "sarcasm"]
ASPECTS = ["剧情", "画质", "配音", "节奏", "角色"]


def parse_args():
    parser = argparse.ArgumentParser(description="使用 DeepSeek 生成四分类中文评论数据")
    parser.add_argument("--per-combination", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip("，。！？,.!? ")


def load_existing() -> tuple[list[dict], set[str], Counter]:
    items, texts, counts = [], set(), Counter()
    if not RAW_FILE.exists():
        return items, texts, counts
    for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = normalize_text(item.get("text", ""))
        key = (item.get("label"), item.get("aspect"))
        if text and text not in texts and key[0] in LABELS and key[1] in ASPECTS:
            item["text"] = text
            items.append(item)
            texts.add(text)
            counts[key] += 1
    return items, texts, counts


def build_prompt(label: str, aspect: str, amount: int, seed_words: list[str]) -> str:
    definitions = {
        "positive": "明确表达喜欢、认可或满意",
        "neutral": "只陈述客观信息，不表达明显好恶",
        "negative": "明确表达不满、批评或失望",
        "sarcasm": "表面像夸奖，实际表达讽刺或不满",
    }
    return f"""请生成 {amount} 条中文短评论，用于文本分类实验。
所有评论的标签必须是 {label}，含义为：{definitions[label]}。
所有评论主要评价对象必须是：{aspect}。

要求：
1. 每条 8-45 个中文字符，像普通用户随手写的评论。
2. 句式和措辞必须多样，不要编号，不要解释标签。
3. 不要出现“小红书”“数据集”“标签”等词。
4. 避免与这些提示词形成固定模板：{seed_words}。
5. 只返回 JSON 对象，格式为：
{{"items":[{{"text":"评论正文","label":"{label}","aspect":"{aspect}"}}]}}
"""


def request_batch(client, label: str, aspect: str, amount: int, seed_words: list[str]):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": build_prompt(label, aspect, amount, seed_words)}],
        temperature=1.2,
        max_tokens=2500,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    payload = json.loads(content)
    return payload.get("items", [])


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY。")
        return 1

    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _, existing_texts, counts = load_existing()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    random.seed(42)

    with RAW_FILE.open("a", encoding="utf-8") as output:
        for label in LABELS:
            for aspect in ASPECTS:
                key = (label, aspect)
                failures = 0
                while counts[key] < args.per_combination:
                    needed = args.per_combination - counts[key]
                    amount = min(args.batch_size, needed + 5)
                    seed_words = random.sample(sorted(existing_texts), min(3, len(existing_texts)))
                    try:
                        batch = request_batch(client, label, aspect, amount, seed_words)
                    except Exception as exc:
                        failures += 1
                        print(f"{label}/{aspect} 请求失败（{failures}/{args.max_retries}）：{exc}")
                        if failures >= args.max_retries:
                            return 1
                        time.sleep(2**failures)
                        continue

                    added = 0
                    for item in batch:
                        text = normalize_text(item.get("text", ""))
                        if (
                            counts[key] >= args.per_combination
                            or not (6 <= len(text) <= 60)
                            or text in existing_texts
                        ):
                            continue
                        clean_item = {"text": text, "label": label, "aspect": aspect}
                        output.write(json.dumps(clean_item, ensure_ascii=False) + "\n")
                        output.flush()
                        existing_texts.add(text)
                        counts[key] += 1
                        added += 1
                    print(
                        f"{label}/{aspect}: +{added}, "
                        f"{counts[key]}/{args.per_combination}"
                    )
                    failures = failures + 1 if added == 0 else 0
                    if failures >= args.max_retries:
                        print(f"错误：{label}/{aspect} 连续未生成有效新评论。")
                        return 1
    print(f"生成完成：{RAW_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
