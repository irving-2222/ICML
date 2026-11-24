"""
悬念 / 惊喜的计算（参考 Ely, Frankel & Kamenica 2015, "Suspense and Surprise"）。

信念 belief：观众对"结局"各选项的概率，dict{选项: 概率}。
- 惊喜 surprise_t  = || b_t - b_{t-1} ||          看完第 t 个镜头后，信念实际变了多少（回看）
- 悬念 suspense_t  = sqrt( E[ || b' - b_t ||^2 ] ) 观众想象下一个镜头时，信念"可能"变多大（前瞻）
- 不确定性 entropy = 归一化熵，是悬念的一个便宜近似
"""
from __future__ import annotations

import itertools
import math
import random


def entropy(p: dict[str, float]) -> float:
    vals = list(p.values())
    if len(vals) < 2:
        return 0.0
    h = -sum(v * math.log(v) for v in vals if v > 0)
    return h / math.log(len(vals))


def l2(p: dict[str, float], q: dict[str, float]) -> float:
    return math.sqrt(sum((p[k] - q[k]) ** 2 for k in p))


def rollout_suspense(cur: dict[str, float], posts: list[dict[str, float]]) -> float | None:
    if not posts:
        return None
    return math.sqrt(sum(l2(x, cur) ** 2 for x in posts) / len(posts))


def expected_rating(probs: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in probs.items())


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cohens_dz(diffs: list[float]) -> float:
    s = std(diffs)
    return mean(diffs) / s if s > 0 else float("nan")


def sign_flip_test(diffs: list[float], n_mc: int = 20000, seed: int = 0) -> float:
    """配对差值的单侧符号翻转置换检验，H1: mean(diff) > 0。n<=16 时精确枚举。"""
    diffs = [d for d in diffs if d is not None and not math.isnan(d)]
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = sum(diffs) / n
    if n <= 16:
        cnt = total = 0
        for signs in itertools.product([1, -1], repeat=n):
            total += 1
            if sum(s * d for s, d in zip(signs, diffs)) / n >= obs - 1e-12:
                cnt += 1
        return cnt / total
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n_mc):
        if sum(d if rng.random() < 0.5 else -d for d in diffs) / n >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n_mc + 1)
