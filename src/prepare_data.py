"""切分数据：读取原始数据，划分 train/val/test,写入 data/。"""

# src/prepare_data.py
import json, os, random
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer



SYSTEM_PROMPT = (
    "You are the customer support assistant of an online store. "
    "Resolve the customer's request with clear, actionable steps. "
    "When a specific detail is unknown (order number, website section, phone number, "
    "amount, date), write a placeholder in double curly braces such as {{Order Number}} "
    "instead of inventing it. Be polite and concise."
)
SEED = 42
N_TEST, N_VAL, N_PREF, N_SFT_MAX = 10, 5, 75, 8000   # 前三个是每个意图的条数

def to_example(instruction, response, intent, category):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": instruction}],
        "completion": [{"role": "assistant", "content": response}],
        "intent": intent, "category": category,
    }

def main():
    random.seed(SEED)
    raw = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    # 去重：同一句用户问题只留一条，避免同句同时出现在训练集和测试集
    seen, by_intent = set(), defaultdict(list)
    for r in raw:
        key = " ".join(r["instruction"].lower().split())
        if key not in seen:
            seen.add(key)
            by_intent[r["intent"]].append(r)

    splits = defaultdict(list)
    for intent in sorted(by_intent):
        items = by_intent[intent]
        random.shuffle(items)
        a, b, c = N_TEST, N_TEST + N_VAL, N_TEST + N_VAL + N_PREF
        splits["test"] += items[:a]
        splits["val"] += items[a:b]
        splits["pref"] += items[b:c]
        splits["sft"] += items[c:]
    random.shuffle(splits["sft"])
    splits["sft"] = splits["sft"][:N_SFT_MAX]

    out = {k: Dataset.from_list([to_example(r["instruction"], r["response"], r["intent"], r["category"])
                                 for r in v]) for k, v in splits.items()}

    # 自己写的分布外问题，每行 {"instruction": "...", "lang": "en|fr"}，没有参考答案。
    # lang 写进 intent 列，评测时按 intent 分组即可分层统计（en 同语言泛化，fr 跨语言迁移）
    path = "data/ood_raw.jsonl"
    if os.path.exists(path):
        with open(path) as f:
            ood = [json.loads(line) for line in f if line.strip()]
        out["ood"] = Dataset.from_list(
            [to_example(o["instruction"], "", "OOD", o["lang"]) for o in ood])

    dd = DatasetDict(out)
    dd.save_to_disk("data/support")
    for k, v in dd.items():
        print(k, len(v))


    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    ds = load_from_disk("data/support")["sft"].select(range(2000))
    texts = [tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False) for ex in ds]
    lens = [len(tok(t).input_ids) for t in texts]
    print(np.percentile(lens, [50, 90, 99, 100]))   # 一般 99 分位远小于 1024

if __name__ == "__main__":
    main()