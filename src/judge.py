"""裁判模型客户端：通过 OpenAI 兼容接口调用裁判模型（读取 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL 。"""
# src/judge.py
import json, os, re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL"), api_key=os.environ["JUDGE_API_KEY"])
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

# 两个裁判共用的“编造事实”定义：占位符视为对系统真实数据的引用，不算编造；
# 只有直接写出的具体值（value）或纯文字的店铺规则/能力/已完成动作（claim）才算
# 注意：拼进模板后会被 .format()，所以字面花括号要写四层 {{{{...}}}}
INVENTED_RULES = """Placeholders: text in double braces such as {{{{Order Number}}}} or {{{{Cancellation Period}}}} is a reference
to the store's real data, filled in by the system. Using a placeholder is never an invented fact, whatever it stands for.

Invented value: a concrete detail the customer did not give, written out instead of as a placeholder:
phone number, email, URL, business hours, fee or price, a time such as "3-5 business days",
a list of payment methods. Hedging ("typically", "usually") does not excuse it.

Invented claim: a store-specific rule, policy, capability or completed action stated in plain text with no
placeholder, e.g. "orders can only be cancelled before shipping", "we offer payment plans", "I have cancelled your order".

Not invented: generic steps any store has ("log in, open My Orders, choose Cancel"); hedged conditionals
without a concrete value ("if it has not shipped, it can usually be cancelled").
"""

CRITERIA = """In order of importance:
1. It addresses what the customer actually asked.
2. It gives concrete, actionable steps.
3. It does not invent facts; unknown details must be placeholders like {{{{Order Number}}}}.
4. Professional, polite tone without padding. Length alone earns no points.

"""

POINTWISE = """You are grading a reply written by an online store's customer-support assistant.

Customer message:
{question}

Reference reply by a support expert (guidance only, the candidate does not need to match it):
{reference}

Candidate reply:
{candidate}

Score the candidate from 1 to 10. """ + CRITERIA + INVENTED_RULES + """
The reference may itself contain invented items; that does not excuse them in the candidate.
If the candidate contains any invented value or claim, the score is lower than 5: 4 if it is a minor aside,
3 if the customer is likely to act on it (visit the URL, call the number, wait that long, rely on the rule),
2 if the main answer depends on it.

Return only JSON: {{"score": <integer 1-10>, "invented": [{{"text": "<quoted span>", "kind": "value" | "claim"}}], "reason": "<one sentence>"}}
Use an empty list for "invented" if nothing is invented."""

PAIRWISE = """You are comparing two replies from an online store's customer-support assistant.

Customer message:
{question}

Reply A:
{a}

Reply B:
{b}

Which reply serves the customer better? """ + CRITERIA + INVENTED_RULES + """
A reply with an invented item the customer would act on loses, unless the other fails criterion 1;
if both invent, prefer the one whose invented items matter less. Do not prefer a reply for its length or formatting.

Return only JSON: {{"winner": "A" | "B" | "tie", "invented_a": [], "invented_b": [], "reason": "<one sentence>"}}
"invented_a" / "invented_b" list which kinds of invented items each reply has: "value", "claim", both, or empty."""

def _extract_json(content):
    """从文本里取第一个含 score 或 winner 的 JSON 对象：逐个 '{' 位置 raw_decode，找不到返回 None"""
    dec = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(content, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and ("score" in obj or "winner" in obj):
            return obj
    return None

def _call(prompt: str):
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, 
                reasoning_effort="low",
                extra_body={"thinking": {"type": "enabled"}}, 
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}])
            out = _extract_json(r.choices[0].message.content or "")
            if out is not None:
                return out
            print(f"judge warning: finish_reason={r.choices[0].finish_reason}, no JSON parsed")
        except Exception as e:
            print("judge error:", e)
    return None

def parse_invented(raw):
    """规范化裁判返回的 invented 列表：只保留 {"text": str, "kind": "value"|"claim"}（kind 去空白转小写）。
    被丢弃的条目打印一行警告；缺失/None 或整体非 list 不算丢弃"""
    if not isinstance(raw, list):
        return []
    out = []
    for it in raw:
        kind = str(it.get("kind")).strip().lower() if isinstance(it, dict) else None
        if isinstance(it, dict) and isinstance(it.get("text"), str) and it["text"] and kind in ("value", "claim"):
            out.append({"text": it["text"], "kind": kind})
        else:
            print(f"judge warning: dropped invented entry {repr(it)[:120]}")
    return out

def kinds(items):
    """invented 条目里出现过的 kind 集合"""
    return {it["kind"] for it in items}

def score_with_reason(question, reference, candidate):
    """返回 (score, reason, invented)；裁判失败时为 (None, None, [])"""
    out = _call(POINTWISE.format(question=question, reference=reference or "(none)",
                                 candidate=candidate))
    if not out or "score" not in out:
        return None, None, []
    try:
        sc = int(out["score"])
    except (TypeError, ValueError):
        return None, None, []   # 非整数分数：不要抛异常，否则会中断整个线程池
    return sc, out.get("reason"), parse_invented(out.get("invented"))

def score(question, reference, candidate):
    return score_with_reason(question, reference, candidate)[0]

def _winner(r):
    """裁判返回的 winner 规范成 "A" / "B" / "tie"；缺失或非法时为 None"""
    return {"a": "A", "b": "B", "tie": "tie"}.get(str(r.get("winner", "")).strip().lower())

def parse_kinds(raw):
    """pairwise 返回的编造类型列表规范化：只保留 "value"/"claim"（去空白转小写），去重排序。
    非 list 当作 []；未知类型打印一行警告后丢弃"""
    if not isinstance(raw, list):
        return []
    out = set()
    for k in raw:
        k2 = str(k).strip().lower()
        if k2 in ("value", "claim"):
            out.add(k2)
        else:
            print(f"judge warning: dropped invented kind {repr(k)[:120]}")
    return sorted(out)

def merge_invented(r1, r2):
    """合并两种顺序的编造类型，返回 (kinds_a, kinds_b)，均为排序后的 list。
    r1 为原顺序；r2 为交换顺序，其 invented_a 属于 b，invented_b 属于 a。格式非法的当作 []"""
    inv_a = set(parse_kinds(r1.get("invented_a"))) | set(parse_kinds(r2.get("invented_b")))
    inv_b = set(parse_kinds(r1.get("invented_b"))) | set(parse_kinds(r2.get("invented_a")))
    return sorted(inv_a), sorted(inv_b)

def compare(question, a, b):
    """两种顺序各判一次，返回 dict：
    score: +1 = a 更好，-1 = b 更好，两种顺序的结论取平均；任一次失败为 None（不算平局）
    ab / ba: 原顺序、交换顺序的原始槽位结论（ba 里的 "A" 指 b），未翻转
    reason_ab / reason_ba: 两次判决的理由
    invented_a / invented_b: 两种顺序合并后的编造类型 ["claim", "value"] 的子集；某次失败时只含另一次的结果"""
    r1 = _call(PAIRWISE.format(question=question, a=a, b=b)) or {}
    r2 = _call(PAIRWISE.format(question=question, a=b, b=a)) or {}
    ab, ba = _winner(r1), _winner(r2)
    v1 = {"A": 1, "B": -1, "tie": 0}.get(ab)
    v2 = {"A": -1, "B": 1, "tie": 0}.get(ba)
    score = None if v1 is None or v2 is None else (v1 + v2) / 2
    invented_a, invented_b = merge_invented(r1, r2)
    return {"score": score, "ab": ab, "ba": ba,
            "reason_ab": r1.get("reason"), "reason_ba": r2.get("reason"),
            "invented_a": invented_a, "invented_b": invented_b}



if __name__ == "__main__":
    print(POINTWISE.format(question="aaa", reference="sdasdad", candidate="faffff"))