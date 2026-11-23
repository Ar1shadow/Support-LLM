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
Return only JSON: {{"winner": "A" or "B" or "tie"}}"""

def _call(prompt: str):
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            print("judge error:", e)
    return None

def score(question, reference, candidate):
    out = _call(POINTWISE.format(question=question, reference=reference or "(none)",
                                 candidate=candidate))
    return int(out["score"]) if out and "score" in out else None

def compare(question, a, b):
    """+1 = a 更好，-1 = b 更好，两种顺序的结论取平均"""
    r1 = _call(PAIRWISE.format(question=question, a=a, b=b)) or {}
    r2 = _call(PAIRWISE.format(question=question, a=b, b=a)) or {}
    v1 = {"A": 1, "B": -1}.get(r1.get("winner"), 0)
    v2 = {"A": -1, "B": 1}.get(r2.get("winner"), 0)
    return (v1 + v2) / 2



if __name__ == "__main__":
    print(POINTWISE.format(question="aaa", reference="sdasdad", candidate="faffff"))