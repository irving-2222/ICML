"""
分析真人打分，并与观众模型（7B / 72B）对比。

用法（在 bomb_pilot/ 目录下）：
    python human_eval/analyze_human.py
    python human_eval/analyze_human.py --exclude R05 R11     # 剔除不认真的打分人

输入：human_eval/responses/*.csv（打分人发回来的文件，直接全部放进去）
输出：human_eval/report/report_human.md + 曲线图

回答三个问题：
  1. 真人身上，两个版本是否也有预期的差别？（悬念/恐惧预期/惊喜）
  2. 真人的"紧张"更接近哪个模型指标：恐惧预期、悬念、不确定性，还是直接打分？
  3. 7B 和 72B，哪个观众模型更像真人？
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.common import load_story  # noqa: E402

EPS = 1e-3
STORIES = ["configs/stories/cafe_bomb.json", "configs/stories/poisoned_wine.json", "configs/stories/thin_ice.json"]


# ---------------------------------------------------------------- 基础计算
def kl(p: dict, q: dict) -> float:
    """KL(p || q)，p 为新信念、q 为旧信念；加平滑防止 log(0)。"""
    P = {k: p[k] + EPS for k in p}
    Q = {k: q[k] + EPS for k in p}
    zp, zq = sum(P.values()), sum(Q.values())
    return sum(P[k] / zp * math.log((P[k] / zp) / (Q[k] / zq)) for k in p)


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def entropy(p: dict) -> float:
    v = [x for x in p.values() if x > 0]
    return -sum(x * math.log(x) for x in v) / math.log(len(p)) if len(p) > 1 else 0.0


def pearson(x, y):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)]
    if len(pts) < 3:
        return float("nan"), len(pts)
    mx, my = mean([a for a, _ in pts]), mean([b for _, b in pts])
    sxy = sum((a - mx) * (b - my) for a, b in pts)
    sx = math.sqrt(sum((a - mx) ** 2 for a, _ in pts))
    sy = math.sqrt(sum((b - my) ** 2 for _, b in pts))
    return (sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")), len(pts)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not math.isnan(a) and not math.isnan(b)]
    if len(pts) < 3:
        return float("nan")
    return pearson(ranks([a for a, _ in pts]), ranks([b for _, b in pts]))[0]


def strat_perm_test(groups: list[tuple[list, list]], n: int = 20000, seed: int = 0) -> float:
    """分层置换检验（单侧，H1: A 组均值 > B 组）。每层 = 一个故事，只在层内打乱版本标签。"""
    def stat(gs):
        diffs = [mean(a) - mean(b) for a, b in gs if a and b]
        return mean(diffs)
    obs = stat(groups)
    if math.isnan(obs):
        return float("nan")
    rng = random.Random(seed)
    cnt = 0
    for _ in range(n):
        gs = []
        for a, b in groups:
            pool = a + b
            rng.shuffle(pool)
            gs.append((pool[:len(a)], pool[len(a):]))
        if stat(gs) >= obs - 1e-12:
            cnt += 1
    return (cnt + 1) / (n + 1)


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ---------------------------------------------------------------- 读取真人数据
def load_human(resp_dir: Path, exclude: set[str]):
    rows = []
    for fp in sorted(glob.glob(str(resp_dir / "*.csv"))):
        with open(fp, encoding="utf-8-sig") as f:
            rows += list(csv.DictReader(f))
    vids = defaultdict(list)
    for r in rows:
        if r["rater"] in exclude:
            continue
        vids[(r["rater"], r["story"], r["version"], int(r["seed"]))].append(r)
    out = []
    for (rater, story, version, seed), rs in vids.items():
        rs.sort(key=lambda r: int(r["t"]))
        keys = [k[2:] for k in rs[0] if k.startswith("p_")]
        beliefs = [{k: float(r["p_" + k]) for k in keys} for r in rs]

        def num(r, k):
            return float(r[k]) if r.get(k) not in (None, "") else None
        out.append({
            "rater": rater, "story": story, "version": version, "seed": seed,
            "beliefs": beliefs,
            "kl": [None] + [kl(beliefs[t], beliefs[t - 1]) for t in range(1, len(beliefs))],
            "tension": [num(r, "tension") for r in rs],
            "surprise": [num(r, "surprise") for r in rs],
            "curiosity": [num(r, "curiosity") for r in rs],
            "rt": [num(r, "rt_ms") for r in rs],
            "comprehension": rs[-1].get("comprehension", ""),
            "complete": len(rs),
        })
    return out


def quality(hv: list[dict]) -> list[dict]:
    flags = defaultdict(list)
    for v in hv:
        if len(v["comprehension"]) < 4:
            flags[v["rater"]].append("描述过短")
        rts = [x for x in v["rt"] if x]
        if rts and sorted(rts)[len(rts) // 2] < 2500:
            flags[v["rater"]].append("作答过快")
        ts = [x for x in v["tension"] if x is not None]
        if len(ts) >= 5 and len(set(ts)) == 1:
            flags[v["rater"]].append("紧张度全相同")
    return [{"rater": k, "flags": sorted(set(f))} for k, f in sorted(flags.items())]


# ---------------------------------------------------------------- 读取模型结果
def load_model(viewer_dir: Path, story_ids):
    res = defaultdict(list)
    for sid in story_ids:
        for fp in glob.glob(str(viewer_dir / sid / "*.json")):
            r = json.load(open(fp, encoding="utf-8"))
            m = r["modes"]["video"]
            q = m["questions"][next(iter(m["questions"]))]
            b = q["beliefs"]
            res[(r["story"], r["version"])].append({
                "beliefs": b,
                "kl": [None] + [kl(b[t], b[t - 1]) for t in range(1, len(b))],
                "entropy": [entropy(x) for x in b],
                "suspense": q.get("suspense_rollout"),
                "rating": m.get("direct_rating"),
            })
    return res


def curve(items, fn, T):
    out = []
    for t in range(T + 1):
        vals = []
        for it in items:
            s = fn(it)
            if s is not None and t < len(s) and s[t] is not None:
                vals.append(s[t])
        out.append(mean(vals) if vals else None)
    return out


# ---------------------------------------------------------------- 主流程
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", default=str(HERE / "responses"))
    p.add_argument("--models", nargs="+", default=["7B=outputs/bomb_pilot/viewer",
                                                   "72B=outputs/bomb_pilot_qwen72b/viewer"])
    p.add_argument("--exclude", nargs="*", default=[])
    p.add_argument("--out", default=str(HERE / "report"))
    args = p.parse_args()

    stories = {s.id: s for s in (load_story(x) for x in STORIES)}
    hv = load_human(Path(args.responses), set(args.exclude))
    if not hv:
        raise SystemExit(f"没有找到真人数据：{args.responses}/*.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    harm = {sid: s.questions[0].harm_option for sid, s in stories.items()}
    L = ["# 真人打分分析报告\n"]

    raters = sorted({v["rater"] for v in hv})
    L.append(f"有效打分人 {len(raters)} 位，共 {len(hv)} 段短片的打分。" +
             (f"已剔除：{', '.join(args.exclude)}。" if args.exclude else "") + "\n")
    cnt = defaultdict(int)
    for v in hv:
        cnt[(v["story"], v["version"])] += 1
    L.append("| 故事 | 手提包版（surprise）人数 | 揭示版（suspense）人数 |\n|---|---|---|")
    for sid in stories:
        L.append(f"| {sid} | {cnt[(sid, 'surprise')]} | {cnt[(sid, 'suspense')]} |")
    q = quality(hv)
    if q:
        L.append("\n**可能不认真的打分人**（建议人工复核后决定是否用 `--exclude` 剔除）：")
        for x in q:
            L.append(f"- {x['rater']}：{'、'.join(x['flags'])}")
    L.append("")

    # ---------- 问题 1：真人身上两个版本是否有差别
    def win(sid):
        s = stories[sid]
        return range(s.reveal_index + 1, s.final_index + 1)

    tests = {
        "紧张度（揭示后到结局前，平均）": ("suspense", "surprise", lambda v: mean([v["tension"][t] for t in win(v["story"])])),
        "恐惧预期（揭示后到结局前，平均受伤概率）": ("suspense", "surprise", lambda v: mean([v["beliefs"][t][harm[v["story"]]] for t in win(v["story"])])),
        "想知道后续的程度（揭示后到结局前）": ("suspense", "surprise", lambda v: mean([v["curiosity"][t] for t in win(v["story"]) if t < stories[v["story"]].final_index + 1])),
        "结局镜头的意外程度（打分）": ("surprise", "suspense", lambda v: v["surprise"][stories[v["story"]].final_index + 1]),
        "结局镜头的惊喜（KL，由信念计算）": ("surprise", "suspense", lambda v: v["kl"][stories[v["story"]].final_index + 1]),
    }
    L.append("## 1. 真人身上，两个版本是否有预期的差别？\n")
    L.append("每行比较两组不同的人（组间比较）。差值为正 = 方向符合预期。p 为分层置换检验（单侧）。\n")
    L.append("| 指标 | 比较 | " + " | ".join(stories) + " | 汇总差值 | p |")
    L.append("|---|---|" + "---|" * len(stories) + "---|---|")
    summary = {}
    for name, (A, B, fn) in tests.items():
        groups, cells = [], []
        for sid in stories:
            a = [fn(v) for v in hv if v["story"] == sid and v["version"] == A]
            b = [fn(v) for v in hv if v["story"] == sid and v["version"] == B]
            a = [x for x in a if x is not None and not math.isnan(x)]
            b = [x for x in b if x is not None and not math.isnan(x)]
            groups.append((a, b))
            cells.append(fmt(mean(a) - mean(b)) if a and b else "-")
        d = mean([mean(a) - mean(b) for a, b in groups if a and b])
        pv = strat_perm_test(groups)
        summary[name] = (d, pv)
        ptxt = "<0.001" if pv < 0.001 else fmt(pv)
        L.append(f"| {name} | {A} − {B} | " + " | ".join(cells) + f" | {fmt(d)} | {ptxt} |")
    L.append("")

    # ---------- 问题 2/3：模型指标与真人的一致性
    hum = defaultdict(list)
    for v in hv:
        hum[(v["story"], v["version"])].append(v)
    keys = [(sid, ver) for sid in stories for ver in ("surprise", "suspense")]
    T = {sid: len(stories[sid].versions["suspense"]) for sid in stories}

    def hcurve(fn):
        return {k: curve(hum[k], fn, T[k[0]]) for k in keys if hum[k]}

    H = {
        "tension": hcurve(lambda v: v["tension"]),
        "surprise": hcurve(lambda v: v["surprise"]),
        "harm": {k: curve(hum[k], lambda v, s=k[0]: [b[harm[s]] for b in v["beliefs"]], T[k[0]]) for k in keys if hum[k]},
        "kl": hcurve(lambda v: v["kl"]),
    }

    def pts(hc, mc, t_from=1):
        xs, ys = [], []
        for k in keys:
            if k not in hc or k not in mc:
                continue
            for t in range(t_from, T[k[0]] + 1):
                xs.append(hc[k][t] if t < len(hc[k]) else None)
                ys.append(mc[k][t] if t < len(mc[k]) else None)
        return xs, ys

    L.append("## 2. 真人的感受更接近哪个模型指标？\n")
    L.append("每个点 = 一个（故事, 版本, 镜头）上真人平均值 vs 模型平均值。r = Pearson 相关，ρ = Spearman 相关，越接近 1 越一致。\n")
    for spec in args.models:
        label, d = spec.split("=", 1)
        vdir = ROOT / d
        if not vdir.exists():
            L.append(f"**{label}**：未找到 `{d}`，跳过（需要把该模型的 viewer/ 文件夹推送上来）。\n")
            continue
        M = load_model(vdir, list(stories))
        mc = {
            "恐惧预期（受伤概率）": {k: curve(M[k], lambda r, s=k[0]: [b[harm[s]] for b in r["beliefs"]], T[k[0]]) for k in keys if M[k]},
            "悬念（rollout）": {k: curve(M[k], lambda r: r["suspense"], T[k[0]]) for k in keys if M[k]},
            "不确定性（熵）": {k: curve(M[k], lambda r: r["entropy"], T[k[0]]) for k in keys if M[k]},
            "直接打分（紧张度）": {k: curve(M[k], lambda r: r["rating"], T[k[0]]) for k in keys if M[k]},
            "惊喜（KL）": {k: curve(M[k], lambda r: r["kl"], T[k[0]]) for k in keys if M[k]},
        }
        L.append(f"### 观众模型 {label}\n")
        L.append("| 真人指标 | 模型指标 | r | ρ | 点数 |\n|---|---|---|---|---|")
        for hname, hkey, mnames in [
            ("紧张度（打分）", "tension", ["恐惧预期（受伤概率）", "悬念（rollout）", "不确定性（熵）", "直接打分（紧张度）"]),
            ("意外程度（打分）", "surprise", ["惊喜（KL）", "直接打分（紧张度）"]),
            ("惊喜（KL，真人信念）", "kl", ["惊喜（KL）"]),
            ("受伤概率（真人信念）", "harm", ["恐惧预期（受伤概率）"]),
        ]:
            for mn in mnames:
                x, y = pts(H[hkey], mc[mn])
                r, n = pearson(x, y)
                L.append(f"| {hname} | {mn} | {fmt(r, 2)} | {fmt(spearman(x, y), 2)} | {n} |")
        L.append("")

    L.append("## 怎么读\n")
    L.append("- **第 1 节**：真人数据本身是否支持\"揭示版更紧张、手提包版结局更意外\"。毒酒故事要特别看：真人是否觉得揭示版更紧张（恐惧预期）。")
    L.append("- **第 2 节**：真人\"紧张度\"那几行里，哪个模型指标的相关最高，就说明真人的紧张更接近哪种概念；同一指标下 7B 和 72B 谁更高，谁就更像真人。")
    L.append("- 真人只看了每个故事一次（组间设计），数据量较小，结论以方向和量级为主。")
    (out / "report_human.md").write_text("\n".join(L), encoding="utf-8")
    plot(H, stories, out)
    print(f"报告已写入 {out / 'report_human.md'}")


def plot(H, stories, out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"surprise": "#2b6cb0", "suspense": "#c53030"}
    for sid, s in stories.items():
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        for ax, (key, title) in zip(axes, [("tension", "Human tension (1-5)"), ("surprise", "Human surprise (1-5)"),
                                           ("harm", "Human P(harm)"), ("kl", "Human surprise (KL)")]):
            for ver in ("surprise", "suspense"):
                c = H[key].get((sid, ver))
                if not c:
                    continue
                xs = [t for t, v in enumerate(c) if v is not None]
                ax.plot(xs, [c[t] for t in xs], marker="o", color=colors[ver], label=ver)
            ax.axvline(s.reveal_index + 1, color="gray", ls="--", lw=1)
            ax.axvline(s.final_index + 1, color="black", ls=":", lw=1)
            ax.set_title(title)
            ax.set_xlabel("shots watched (t)")
            ax.legend(fontsize=8)
        fig.suptitle(f"{s.title} ({sid}) - human ratings")
        fig.tight_layout()
        fig.savefig(out / f"human_{sid}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
