"""
第 4 步：汇总观众结果，画曲线，检验假设，生成报告。

假设（t = 已看的镜头数）：
  H1 悬念：  在 reveal 之后、结局之前的窗口里，suspense 版本的悬念 > surprise 版本
  H2 惊喜：  看到结局镜头时，surprise 版本的惊喜 > suspense 版本
  H3 传达：  关键信息（炸弹等）在 reveal 镜头里被看到的概率高，在其他镜头里低
  H4 对照：  直接让 VLM 打"紧张度"分，能否同样区分两个版本（期望：不如基于信念的方法）
另外输出"传达差距"：同一版本 video 模式和 text 模式的信念差距。
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from .common import base_arg_parser, load_config, load_stories, out_dir, result_path, setup_logging, video_path
from .metrics import cohens_dz, l2, mean, sign_flip_test, std

log = setup_logging("analyze")


def window(story) -> range:
    return range(story.reveal_index + 1, story.final_index + 1)


def suspense_series(res: dict, qid: str, mode: str = "video") -> tuple[list, str]:
    q = res["modes"][mode]["questions"][qid]
    if "suspense_rollout" in q:
        return q["suspense_rollout"], "rollout"
    return q["entropy"], "entropy"


def series_stats(runs: list[list]) -> tuple[list[float], list[float]]:
    T = max(len(r) for r in runs)
    ms, ss = [], []
    for t in range(T):
        vals = [r[t] for r in runs if t < len(r) and r[t] is not None]
        ms.append(mean(vals) if vals else float("nan"))
        ss.append(std(vals) if vals else 0.0)
    return ms, ss


def plot_story(story, data, cfg, sus_ver, sur_ver, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = story.questions[0]
    colors = {sur_ver: "#2b6cb0", sus_ver: "#c53030"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("P(%s)  [video]" % q.harm_option, lambda r: [b[q.harm_option] for b in r["modes"]["video"]["questions"][q.id]["beliefs"]]),
        ("Uncertainty (entropy)", lambda r: r["modes"]["video"]["questions"][q.id]["entropy"]),
        ("Suspense (rollout)", lambda r: r["modes"]["video"]["questions"][q.id].get("suspense_rollout")),
        ("Surprise", lambda r: r["modes"]["video"]["questions"][q.id]["surprise"]),
        ("Direct rating 1-5 (baseline)", lambda r: r["modes"]["video"].get("direct_rating")),
    ]
    for ax, (title, fn) in zip(axes.flat, panels):
        for ver in (sur_ver, sus_ver):
            runs = [fn(r) for r in data.get(ver, [])]
            runs = [r for r in runs if r]
            if not runs:
                continue
            m, s = series_stats(runs)
            xs = list(range(len(m)))
            ax.plot(xs, m, marker="o", color=colors[ver], label=ver)
            ax.fill_between(xs, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                            color=colors[ver], alpha=0.15)
        ax.set_title(title)
        ax.axvline(story.reveal_index + 1, color="gray", ls="--", lw=1)
        ax.axvline(story.final_index + 1, color="black", ls=":", lw=1)
        ax.set_xlabel("shots watched (t)")
        ax.legend(fontsize=8)

    # 传达差距：suspense 版本 video vs text
    ax = axes.flat[5]
    for mode, ls in (("video", "-"), ("text", "--")):
        runs = [[b[q.harm_option] for b in r["modes"][mode]["questions"][q.id]["beliefs"]]
                for r in data.get(sus_ver, []) if mode in r["modes"]]
        if runs:
            m, _ = series_stats(runs)
            ax.plot(range(len(m)), m, ls=ls, marker="o", color=colors[sus_ver], label=f"{sus_ver} [{mode}]")
    ax.set_title("Communication gap: video vs text plan")
    ax.axvline(story.reveal_index + 1, color="gray", ls="--", lw=1)
    ax.set_xlabel("shots watched (t)")
    ax.legend(fontsize=8)

    fig.suptitle(f"{story.title}  ({story.id})   gray dashed = reveal shot, dotted = ending shot")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    p = base_arg_parser(__doc__)
    args = p.parse_args()
    cfg = load_config(args)
    stories = load_stories(cfg)
    acfg = cfg.get("analysis", {})
    sus_ver = acfg.get("suspense_version", "suspense")
    sur_ver = acfg.get("surprise_version", "surprise")
    rep_dir = out_dir(cfg) / "report"
    rep_dir.mkdir(parents=True, exist_ok=True)

    h1, h2, h4, rows = [], [], [], []
    probe_rows, gap_rows = [], []
    sus_kind = "entropy"
    per_story = {}

    for st in stories:
        q = st.questions[0]
        data = defaultdict(list)
        for ver in st.versions:
            for seed in cfg["seeds"]:
                rp = result_path(cfg, st.id, ver, seed)
                if rp.exists():
                    with open(rp, encoding="utf-8") as f:
                        data[ver].append(json.load(f))
        if not data.get(sus_ver) or not data.get(sur_ver):
            log.warning("%s 缺少结果，跳过", st.id)
            continue
        plot_story(st, data, cfg, sus_ver, sur_ver, rep_dir / f"{st.id}.png")

        by_seed = {ver: {r["seed"]: r for r in data[ver]} for ver in (sus_ver, sur_ver)}
        seeds = sorted(set(by_seed[sus_ver]) & set(by_seed[sur_ver]))
        w = window(st)
        tf = st.final_index + 1
        st_h1, st_h2, st_h4 = [], [], []
        for s in seeds:
            a, b = by_seed[sus_ver][s], by_seed[sur_ver][s]
            sa, sus_kind = suspense_series(a, q.id)
            sb, _ = suspense_series(b, q.id)
            d1 = mean([sa[t] for t in w]) - mean([sb[t] for t in w])
            d2 = b["modes"]["video"]["questions"][q.id]["surprise"][tf] - a["modes"]["video"]["questions"][q.id]["surprise"][tf]
            st_h1.append(d1)
            st_h2.append(d2)
            if a["modes"]["video"].get("direct_rating") and b["modes"]["video"].get("direct_rating"):
                ra, rb = a["modes"]["video"]["direct_rating"], b["modes"]["video"]["direct_rating"]
                st_h4.append(mean([ra[t] for t in w]) - mean([rb[t] for t in w]))
            rows.append({"story": st.id, "seed": s, "H1_suspense_diff": d1, "H2_surprise_diff": d2,
                         "H4_rating_diff": st_h4[-1] if st_h4 else None,
                         "label_mass_min": min(a.get("label_mass_min") or 1, b.get("label_mass_min") or 1)})
        h1 += st_h1
        h2 += st_h2
        h4 += st_h4
        per_story[st.id] = {"H1_mean": mean(st_h1), "H2_mean": mean(st_h2),
                            "H4_mean": mean(st_h4) if st_h4 else None, "n_seeds": len(seeds)}

        # H3 传达检查（只看 video 模式下每个镜头单独判断）
        for ver in st.versions:
            for r in data[ver]:
                for pid, pr in (r.get("probes") or {}).items():
                    exp_pos = [i for i, sid in enumerate(r["shots"]) if sid in pr["expected_shots"]]
                    other = [i for i in range(len(r["shots"])) if i not in exp_pos]
                    probe_rows.append({
                        "story": st.id, "version": ver, "seed": r["seed"], "fact": pid,
                        "p_yes_expected": mean([pr["p_yes"][i] for i in exp_pos]) if exp_pos else None,
                        "p_yes_other": mean([pr["p_yes"][i] for i in other]) if other else None,
                    })
                # 传达差距
                if "text" in r["modes"]:
                    bv = r["modes"]["video"]["questions"][q.id]["beliefs"]
                    bt = r["modes"]["text"]["questions"][q.id]["beliefs"]
                    gaps = [l2(x, y) for x, y in zip(bv, bt)]
                    gap_rows.append({"story": st.id, "version": ver, "seed": r["seed"],
                                     "gap_mean": mean(gaps[1:]), "gap_at_reveal": gaps[st.reveal_index + 1]})

    if not rows:
        log.error("没有可分析的结果。请先运行 run_viewer。")
        return

    summary = {
        "suspense_metric": sus_kind,
        "H1": {"mean_diff": mean(h1), "dz": cohens_dz(h1), "p_one_sided": sign_flip_test(h1), "n": len(h1),
               "frac_positive": mean([1.0 if d > 0 else 0.0 for d in h1])},
        "H2": {"mean_diff": mean(h2), "dz": cohens_dz(h2), "p_one_sided": sign_flip_test(h2), "n": len(h2),
               "frac_positive": mean([1.0 if d > 0 else 0.0 for d in h2])},
        "H4": ({"mean_diff": mean(h4), "dz": cohens_dz(h4), "p_one_sided": sign_flip_test(h4), "n": len(h4)}
               if h4 else None),
        "per_story": per_story,
    }
    with open(rep_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "pairs": rows, "probes": probe_rows, "comm_gap": gap_rows},
                  f, ensure_ascii=False, indent=2)
    with open(rep_dir / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # 人工评测模板：每个短片每个镜头打 1-5 分紧张度
    with open(rep_dir / "human_eval_template.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rater_id", "video_file", "shot_index(1-based)", "suspense_1to5", "surprise_1to5"])
        for st in stories:
            for ver in st.versions:
                for seed in cfg["seeds"]:
                    vp = video_path(cfg, st.id, ver, seed)
                    for i in range(len(st.versions[ver])):
                        wr.writerow(["", vp.name if vp.exists() else f"{st.id}/{ver}_seed{seed}.mp4", i + 1, "", ""])

    write_report(rep_dir, summary, per_story, probe_rows, gap_rows, stories, sus_ver, sur_ver)
    log.info("报告已写入 %s", rep_dir / "report.md")


def verdict(h: dict | None, alpha: float = 0.05) -> str:
    if not h or h["n"] == 0:
        return "无数据"
    if h["mean_diff"] > 0 and h["p_one_sided"] < alpha:
        return "✅ 支持"
    if h["mean_diff"] > 0:
        return "🟡 方向对但不显著"
    return "❌ 不支持"


def write_report(rep_dir, summary, per_story, probe_rows, gap_rows, stories, sus_ver, sur_ver):
    L = []
    L.append("# 炸弹实验报告\n")
    L.append(f"悬念指标：`{summary['suspense_metric']}`；每个配对 = 同一故事、同一 seed 下的两个版本。\n")
    L.append("## 假设检验（汇总所有故事）\n")
    L.append("| 假设 | 含义 | 平均差值 | 效应量 dz | 单侧 p | 正向比例 | 结论 |")
    L.append("|---|---|---|---|---|---|---|")
    for key, desc in (("H1", f"窗口内悬念：{sus_ver} − {sur_ver}"),
                      ("H2", f"结局惊喜：{sur_ver} − {sus_ver}")):
        h = summary[key]
        L.append(f"| {key} | {desc} | {fmt(h['mean_diff'])} | {fmt(h['dz'], 2)} | {fmt(h['p_one_sided'])} | "
                 f"{fmt(h['frac_positive'], 2)} | {verdict(h)} |")
    h4 = summary.get("H4")
    if h4:
        L.append(f"| H4（对照） | 直接打分：{sus_ver} − {sur_ver} | {fmt(h4['mean_diff'])} | {fmt(h4['dz'], 2)} | "
                 f"{fmt(h4['p_one_sided'])} | - | {verdict(h4)} |")
    L.append("")
    L.append("## 分故事结果\n")
    L.append("| 故事 | seeds | H1 平均差 | H2 平均差 | H4 平均差 | 曲线图 |")
    L.append("|---|---|---|---|---|---|")
    for sid, v in per_story.items():
        L.append(f"| {sid} | {v['n_seeds']} | {fmt(v['H1_mean'])} | {fmt(v['H2_mean'])} | {fmt(v['H4_mean'])} | "
                 f"[{sid}.png]({sid}.png) |")
    L.append("")
    if probe_rows:
        L.append("## H3 传达检查（关键信息是否真的被画出来）\n")
        L.append("p_yes_expected 应该高（>0.5），p_yes_other 应该低。若 reveal 镜头的 p_yes_expected 低，"
                 "说明视频模型没把炸弹画清楚——这时 H1 不成立不是观众模型的问题，而是生成问题。\n")
        L.append("| 故事 | 版本 | seed | 事实 | p_yes(应出现的镜头) | p_yes(其他镜头) |")
        L.append("|---|---|---|---|---|---|")
        for r in probe_rows:
            L.append(f"| {r['story']} | {r['version']} | {r['seed']} | {r['fact']} | "
                     f"{fmt(r['p_yes_expected'])} | {fmt(r['p_yes_other'])} |")
        L.append("")
    if gap_rows:
        L.append("## 传达差距（video vs text 信念的 L2 距离）\n")
        L.append("差距大 = 观众看视频得到的信息和读分镜文字得到的不一样（画面没传达到计划的信息）。\n")
        L.append("| 故事 | 版本 | seed | 平均差距 | reveal 镜头处差距 |")
        L.append("|---|---|---|---|---|")
        for r in gap_rows:
            L.append(f"| {r['story']} | {r['version']} | {r['seed']} | {fmt(r['gap_mean'])} | {fmt(r['gap_at_reveal'])} |")
        L.append("")
    L.append("## 怎么读这份报告\n")
    L.append("- **H1 + H2 都支持** → 观众 Agent 能感知信息结构带来的悬念/惊喜差异，idea 的关键前提成立。")
    L.append("- **H1 支持、H4 不支持或更弱** → 最理想：说明\"基于信念\"的测量比\"直接问 VLM\"更好，这是论文的一个核心论据。")
    L.append("- **H3 失败** → 先换更强的生成模型（如 wan22_a14b）或改 reveal 镜头 prompt，再看 H1。")
    L.append("- **全不支持且 H3 通过** → 观众模型本身不行：试 72B 观众、增加抽帧、调整问题措辞；仍不行则考虑换方向。")
    L.append("- 若日志里 `label_mass` 很低（<0.5），说明模型没按\"只回答字母\"的格式作答，需要调整 prompt。")
    (rep_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
