"""两两对比胜率：用裁判模型比较两组回复。"""
# src/winrate.py  
# 用法: python src/winrate.py results/gen_dpo.jsonl results/gen_sft.jsonl
import json, statistics, sys
from concurrent.futures import ThreadPoolExecutor
from judge import compare

a = [json.loads(l) for l in open(sys.argv[1])]
b = [json.loads(l) for l in open(sys.argv[2])]
assert [r["question"] for r in a] == [r["question"] for r in b]
with ThreadPoolExecutor(16) as ex:
    v = list(ex.map(lambda p: compare(p[0]["question"], p[0]["responses"][0], p[1]["responses"][0]),
                    zip(a, b)))
win = statistics.mean(x > 0 for x in v)
lose = statistics.mean(x < 0 for x in v)
print(f"A wins {win:.1%}, B wins {lose:.1%}, tie {1 - win - lose:.1%}")