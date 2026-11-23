"""评测生成结果：汇总规则指标与裁判打分，结果写入 results/。"""
# src/eval_responses.py
import argparse, json, random, statistics
from concurrent.futures import ThreadPoolExecutor
from judge import score
from metrics import invented_specifics, uses_placeholder, n_words

def bootstrap_ci(xs, iters=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]

p = argparse.ArgumentParser()
p.add_argument("--gen", required=True)
p.add_argument("--out", required=True)
args = p.parse_args()

recs = [json.loads(l) for l in open(args.gen)]
resp = [r["responses"][0] for r in recs]  #only the first resp.
with ThreadPoolExecutor(16) as ex:
    scores = list(ex.map(lambda r: score(r["question"], r["reference"], r["responses"][0]), recs))
valid = [s for s in scores if s is not None]

summary = {
    "n": len(recs),
    "judge_mean": statistics.mean(valid),
    "judge_ci95": bootstrap_ci(valid),
    "invented_rate": statistics.mean(invented_specifics(t) for t in resp),
    "placeholder_rate": statistics.mean(uses_placeholder(t) for t in resp),
    "mean_words": statistics.mean(n_words(t) for t in resp),
}
print(json.dumps(summary, indent=2))
json.dump({"summary": summary, "scores": scores}, open(args.out, "w"), indent=2)