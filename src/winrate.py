"""两两对比胜率：用裁判模型比较两组回复，汇总统计并把逐条判决写入 results/。"""
# src/winrate.py
# 用法: python src/winrate.py results/gen_dpo.jsonl results/gen_sft.jsonl [--out results/xxx.json]
#       --out 默认 results/winrate_<A 文件名>_vs_<B 文件名>.json
import argparse, json, os, statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from judge import compare
from metrics import bootstrap_ci

p = argparse.ArgumentParser()
p.add_argument("a")                     # A 组生成结果；score > 0 表示 A 更好
p.add_argument("b")                     # B 组生成结果
p.add_argument("--out", default=None)
args = p.parse_args()
stem = lambda path: os.path.splitext(os.path.basename(path))[0]
out = args.out or f"results/winrate_{stem(args.a)}_vs_{stem(args.b)}.json"

a = [json.loads(l) for l in open(args.a)]
b = [json.loads(l) for l in open(args.b)]
assert [r["question"] for r in a] == [r["question"] for r in b]
with ThreadPoolExecutor(16) as ex:
    verdicts = list(ex.map(lambda p: compare(p[0]["question"], p[0]["responses"][0], p[1]["responses"][0]),
                           zip(a, b)))

def rates(scores):
    """有效 score 列表 → 胜 / 负 / 平比例和均值"""
    n = len(scores)
    return {"n": n,
            "win": sum(s > 0 for s in scores) / n,
            "lose": sum(s < 0 for s in scores) / n,
            "tie": sum(s == 0 for s in scores) / n,
            "mean_score": statistics.mean(scores)}

valid = [v for v in verdicts if v["score"] is not None]   # 两次裁判都成功的样本
scores = [v["score"] for v in valid]
assert scores, "all judge calls failed; check JUDGE_* env vars and the judge errors above"

overall = rates(scores)
decisive = overall["win"] + overall["lose"]
flip = {"A": "B", "B": "A", "tie": "tie"}
slots = [w for v in valid for w in (v["ab"], v["ba"]) if w != "tie"]   # 所有非平局判决选的槽位
summary = {
    "n": len(verdicts),
    "n_valid": len(valid),
    "n_failed": len(verdicts) - len(valid),
    "win": overall["win"],
    "lose": overall["lose"],
    "tie": overall["tie"],
    "net_win": overall["win"] - overall["lose"],
    "win_rate_excl_ties": overall["win"] / decisive if decisive else None,
    "mean_score": overall["mean_score"],           # score ∈ [-1, 1]
    "score_ci95": bootstrap_ci(scores),
    # 交换顺序后结论是否一致（翻转槽位后指向同一回答，或都是 tie）
    "consistency": statistics.mean(v["ab"] == flip[v["ba"]] for v in valid),
    # 非平局判决中选槽位 A 的比例，≈0.5 说明裁判没有位置偏差
    "slot_A_rate": statistics.mean(w == "A" for w in slots) if slots else None,
}

by_intent = defaultdict(list)
for r, v in zip(a, verdicts):
    if v["score"] is not None:
        by_intent[r.get("intent")].append(v["score"])
per_intent = {k: rates(s) for k, s in sorted(by_intent.items(), key=lambda kv: str(kv[0]))}

items = [{"question": x["question"], "intent": x.get("intent"),
          "a": x["responses"][0], "b": y["responses"][0], **v}
         for x, y, v in zip(a, b, verdicts)]

print(f"A wins {summary['win']:.1%}, B wins {summary['lose']:.1%}, tie {summary['tie']:.1%}")
print(json.dumps(summary, indent=2))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump({"a": args.a, "b": args.b, "summary": summary, "per_intent": per_intent, "items": items},
          open(out, "w"), indent=2, ensure_ascii=False)
print(f"saved to {out}")
