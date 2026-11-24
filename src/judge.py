"""裁判模型客户端：通过 OpenAI 兼容接口调用裁判模型（读取 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL 。"""
# src/judge.py
import json, os, re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL"), api_key=os.environ["JUDGE_API_KEY"])
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

# 两个裁判共用的“编造事实”定义：value（具体数值）与 claim（规则/流程/能力）
# 注意：拼进模板后会被 .format()，所以字面花括号要写四层 {{{{...}}}}
INVENTED_RULES = """Invented value: a concrete detail the customer did not give, written literally instead of as a
placeholder. Examples: a phone number, email, URL, business hours, a fee or price, a processing time
or deadline like "3-5 business days" or "within 24 hours", a list of supported payment methods.
A placeholder in its place ({{{{Customer Support Phone Number}}}}, {{{{Website URL}}}}) is fine.

Invented claim: a store-specific rule, process, capability or completed action stated as fact.
Examples: "orders can only be cancelled before shipping", "a {{{{Cancellation Fee}}}} applies",
"you can cancel within {{{{Cancellation Period}}}}", "click the 'Cancel Order' button under
'{{{{Online Order Interaction}}}}'", "we offer payment plans / store credit / live chat",
"I have cancelled your order". Wrapping a rule, process or capability in a placeholder does not make it acceptable.
The placeholder hides the value, but the rule or capability is still made up.
A step naming a specific button, page or menu through a non-value placeholder is a claim.

Not invented: a placeholder standing for one value the store certainly has ({{{{Customer Support Phone Number}}}},
{{{{Website URL}}}}); generic navigation wording ("My Orders", "Account Settings", "Forgot PIN"); a generic
3-5 step flow ("log in, open My Orders, select the order, choose Cancel"); hedged statements such as
"if your order has not shipped yet, it can usually be cancelled" or "if a cancellation window applies, ...".
"""

POINTWISE = """You are grading a reply written by an online store's customer-support assistant.

Customer message:
{question}

Reference reply by a support expert (guidance only, the candidate does not need to match it):
{reference}

Candidate reply:
{candidate}

Score the candidate from 1 to 10. In order of importance:
1. It addresses what the customer actually asked.
2. It gives concrete, actionable steps.
3. It does not invent specific facts; unknown details must be placeholders like {{{{Order Number}}}}.
4. Professional, polite tone without padding. Length alone earns no points.

Hard rule: the candidate must not invent facts. Two kinds count:

""" + INVENTED_RULES + """
If the candidate contains any invented value or invented claim, the score is lower than 5:
Score 4 if it is a minor aside, 3 if the customer is likely to act on it
(e.g. visit the URL, call the number, wait that long, rely on the rule), 2 if the main answer depends on it.
Using placeholders elsewhere does not excuse an invented item. Name each invented item in "invented".

Return only JSON: {{"score": <integer 1-10>, "invented": [{{"text": "<quoted span>", "kind": "value" or "claim"}}], "reason": "<one sentence>"}}
Use an empty list for "invented" if nothing is invented."""

PAIRWISE = """You are comparing two replies from an online store's customer-support assistant.

Customer message:
{question}

Reply A:
{a}

Reply B:
{b}

Which reply serves the customer better? Judge helpfulness, correctness and whether it invents
specific facts. Do not prefer a reply because it is longer.
Return only JSON: {{"winner": "A" or "B" or "tie", "reason": "<one sentence>"}}"""

def _call(prompt: str):
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, 
                reasoning_effort="low",
                extra_body={"thinking": {"type": "enabled"}}, 
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}])
            m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            print("judge error:", e)
    return None

def parse_invented(raw):
    """规范化裁判返回的 invented 列表：只保留 {"text": str, "kind": "value"|"claim"}，其余丢弃"""
    if not isinstance(raw, list):
        return []
    return [{"text": it["text"], "kind": it["kind"]} for it in raw
            if isinstance(it, dict) and isinstance(it.get("text"), str)
            and it.get("kind") in ("value", "claim")]

def kinds(items):
    """invented 条目里出现过的 kind 集合"""
    return {it["kind"] for it in items}

def score_with_reason(question, reference, candidate):
    """返回 (score, reason, invented)；裁判失败时为 (None, None, [])"""
    out = _call(POINTWISE.format(question=question, reference=reference or "(none)",
                                 candidate=candidate))
    if not out or "score" not in out:
        return None, None, []
    return int(out["score"]), out.get("reason"), parse_invented(out.get("invented"))

def score(question, reference, candidate):
    return score_with_reason(question, reference, candidate)[0]

def _winner(r):
    """裁判返回的 winner 规范成 "A" / "B" / "tie"；缺失或非法时为 None"""
    return {"a": "A", "b": "B", "tie": "tie"}.get(str(r.get("winner", "")).strip().lower())

def compare(question, a, b):
    """两种顺序各判一次，返回 dict：
    score: +1 = a 更好，-1 = b 更好，两种顺序的结论取平均；任一次失败为 None（不算平局）
    ab / ba: 原顺序、交换顺序的原始槽位结论（ba 里的 "A" 指 b），未翻转
    reason_ab / reason_ba: 两次判决的理由"""
    r1 = _call(PAIRWISE.format(question=question, a=a, b=b)) or {}
    r2 = _call(PAIRWISE.format(question=question, a=b, b=a)) or {}
    ab, ba = _winner(r1), _winner(r2)
    v1 = {"A": 1, "B": -1, "tie": 0}.get(ab)
    v2 = {"A": -1, "B": 1, "tie": 0}.get(ba)
    score = None if v1 is None or v2 is None else (v1 + v2) / 2
    return {"score": score, "ab": ab, "ba": ba,
            "reason_ab": r1.get("reason"), "reason_ba": r2.get("reason")}



if __name__ == "__main__":
    print(POINTWISE.format(question="aaa", reference="sdasdad", candidate="faffff"))