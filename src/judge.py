"""裁判模型客户端：通过 OpenAI 兼容接口调用裁判模型（读取 JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL 。"""
# src/judge.py
import json, os, re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(base_url=os.environ.get("JUDGE_BASE_URL"), api_key=os.environ["JUDGE_API_KEY"])
JUDGE_MODEL = os.environ["JUDGE_MODEL"]

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

Hard rule: if the candidate states any concrete detail that the customer did not give and that is
not a placeholder (e.g. phone number, email, URL, business hours, fee or price, processing time
or deadline like "3-5 business days" or "within 24 hours", a list of supported payment methods),
the score is lower than 5 :  Score 4 if it is a minor aside, 3 if the customer is likely to act on it
(e.g. visit the URL, call the number, wait that long), 2 if the main answer depends on it. 
Using placeholders elsewhere does not excuse an invented detail. Generic navigation wording ("My Orders", "Account Settings", "Forgot PIN") is not invented.
 Name the invented detail in the reason.

Return only JSON: {{"score": <integer 1-10>, "reason": "<one sentence>"}}"""

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

def score_with_reason(question, reference, candidate):
    """返回 (score, reason)；裁判失败时为 (None, None)"""
    out = _call(POINTWISE.format(question=question, reference=reference or "(none)",
                                 candidate=candidate))
    if not out or "score" not in out:
        return None, None
    return int(out["score"]), out.get("reason")

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