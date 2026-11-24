"""规则指标：编造信息检测、占位符检测、回复长度统计；均值的 bootstrap 置信区间。"""
# src/metrics.py
import random, re, statistics

def bootstrap_ci(xs, iters=2000, seed=0):
    """均值的 95% bootstrap 置信区间"""
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]

PATTERNS = {
    "phone": r"\+?\d[\d\-\s().]{7,}\d",
    "email": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "url": r"https?://\S+|www\.\S+",
    "money": r"[$€£]\s?\d+",
    "order_id": r"#\s?\d{4,}",
}
PLACEHOLDER = r"\{\{[^}]*\}\}"

def invented_specifics(text: str) -> bool:
    stripped = re.sub(PLACEHOLDER, "", text)
    return any(re.search(p, stripped) for p in PATTERNS.values())

def uses_placeholder(text: str) -> bool:
    return re.search(PLACEHOLDER, text) is not None

def n_words(text: str) -> int:
    return len(text.split())