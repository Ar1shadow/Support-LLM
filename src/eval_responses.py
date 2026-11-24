"""评测生成结果：汇总规则指标与裁判打分，结果写入 results/。"""
# src/eval_responses.py
import argparse, json, statistics
from concurrent.futures import ThreadPoolExecutor
from judge import score_with_reason, kinds
from metrics import bootstrap_ci, invented_specifics, uses_placeholder, n_words

p = argparse.ArgumentParser()
p.add_argument("--gen", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

recs = [json.loads(l) for l in open(args.gen)]
resp = [r["responses"][0] for r in recs]  #only the first resp.
with ThreadPoolExecutor(16) as ex:
    judged = list(ex.map(lambda r: score_with_reason(r["question"], r["reference"], r["responses"][0]), recs))
scores = [s for s, _, _ in judged]
reasons = [why for _, why, _ in judged]
invented = [inv for _, _, inv in judged]
valid_kinds = [kinds(inv) for s, _, inv in judged if s is not None]
valid = [s for s in scores if s is not None]

summary = {
    "n": len(recs),
    "n_valid": len(valid),
    "judge_mean": statistics.mean(valid),
    "judge_ci95": bootstrap_ci(valid),
    "invented_rate": statistics.mean(invented_specifics(t) for t in resp),
    "judge_value_rate": statistics.mean("value" in k for k in valid_kinds),
    "judge_claim_rate": statistics.mean("claim" in k for k in valid_kinds),
    "judge_any_invented_rate": statistics.mean(bool(k) for k in valid_kinds),
    "placeholder_rate": statistics.mean(uses_placeholder(t) for t in resp),
    "mean_words": statistics.mean(n_words(t) for t in resp),
}
print(json.dumps(summary, indent=2))
json.dump({"summary": summary, "scores": scores, "reasons": reasons,
           "invented": invented},
          open(args.out, "w"), indent=2, ensure_ascii=False)