"""
第 3 步：观众 Agent 逐镜头"看片"，记录信念曲线。

对每个短片（故事 × 版本 × seed）计算：
  1. beliefs[t]        看完前 t 个镜头后，对结局各选项的概率（t=0 表示还没看，只读了开场介绍）
  2. entropy[t]        信念的不确定性（悬念的便宜近似）
  3. surprise[t]       看完第 t 个镜头，信念变化多大（惊喜）
  4. suspense_rollout  想象 K 种下一镜头，信念可能变化多大（悬念，前瞻）
  5. direct_rating     对照：直接问"有多紧张 1-5"
  6. probes            传达检查：关键信息（炸弹等）在每个镜头里能否被看到
modes:
  video = 看真实视频帧；text = 只读分镜文字（计划层）。两者差距 = "画面没传达到"的部分。
"""
from __future__ import annotations

import json
import time

from .common import (Question, Story, base_arg_parser, load_config, load_stories, manifest_path,
                     result_path, setup_logging, shard, write_json)
from .metrics import entropy, expected_rating, l2, rollout_suspense
from .video_io import sample_frames
from .viewer import MockViewer, QwenVLViewer, image, text

log = setup_logging("run_viewer")

LETTERS = "ABCDEFGH"
IMAGINE_PROMPT = (
    "\n\nBefore the next shot appears: in one short sentence, describe what you expect "
    "the NEXT shot of this film to show."
)
RATING_PROMPT = (
    "\n\nOn a scale from 1 (completely calm) to 5 (extremely tense), how much suspense do you "
    "feel right now? Answer with only the number."
)


def question_block(q: Question, order: list[str]) -> dict:
    lines = [f"{LETTERS[i]}. {q.options[k]}" for i, k in enumerate(order)]
    return text(f"\n\nQuestion: {q.text}\n" + "\n".join(lines) + "\nAnswer with only the letter of your choice.")


def belief(viewer, prefix: list[dict], q: Question, n_perm: int, masses: list[float]) -> dict[str, float]:
    """选项轮换 n_perm 次取平均，抵消模型对某个字母位置的偏好。"""
    keys = list(q.options)
    n = len(keys)
    acc = {k: 0.0 for k in keys}
    shifts = list(range(min(max(1, n_perm), n)))
    for s in shifts:
        order = keys[s:] + keys[:s]
        probs, mass = viewer.choice_probs(prefix + [question_block(q, order)], list(LETTERS[:n]))
        masses.append(mass)
        for i, k in enumerate(order):
            acc[k] += probs[LETTERS[i]]
    z = sum(acc.values())
    return {k: v / z for k, v in acc.items()}


def probe_yes(viewer, frames, question: str, masses: list[float]) -> float:
    """单个镜头的是/否检查，Yes/No 位置互换后取平均。"""
    head = [text("Here is one shot from a film:")] + [image(f) for f in frames]
    p1, m1 = viewer.choice_probs(head + [text(f"\n\nQuestion: {question}\nA. Yes\nB. No\n"
                                              "Answer with only the letter.")], ["A", "B"])
    p2, m2 = viewer.choice_probs(head + [text(f"\n\nQuestion: {question}\nA. No\nB. Yes\n"
                                              "Answer with only the letter.")], ["A", "B"])
    masses.extend([m1, m2])
    return 0.5 * (p1["A"] + p2["B"])


def evaluate_video(viewer, story: Story, version: str, seed: int, cfg: dict) -> dict:
    vcfg = cfg["viewer"]
    with open(manifest_path(cfg, story.id, version, seed), encoding="utf-8") as f:
        manifest = json.load(f)
    seq = story.sequence(version)
    T = len(seq)
    frames = [sample_frames(c, int(vcfg["frames_per_shot"]), int(vcfg["max_side"])) for c in manifest["clips"]]
    n_perm = int(vcfg.get("n_option_permutations", 3))
    masses: list[float] = []

    def prefix(t: int, mode: str) -> list[dict]:
        content = [text(story.intro)]
        for i in range(t):
            content.append(text(f"\nShot {i + 1}:"))
            if mode == "video":
                content += [image(fr) for fr in frames[i]]
            else:
                content.append(text(" " + seq[i].description))
        return content

    result = {
        "story": story.id, "version": version, "seed": seed,
        "shots": [s.id for s in seq], "reveal_index": story.reveal_index, "final_index": story.final_index,
        "viewer_model": vcfg.get("model_id"), "modes": {},
    }

    roll = vcfg.get("rollout", {})
    for mode in vcfg.get("modes", ["video"]):
        mres: dict = {"questions": {}}
        do_roll = roll.get("enabled", False) and mode == "video"

        # 想象下一个镜头：对每个 t 只采样一次，所有问题共用
        imagined: list[list[str]] = []
        if do_roll:
            for t in range(T):
                imagined.append(viewer.generate_texts(
                    prefix(t, mode) + [text(IMAGINE_PROMPT)], int(roll.get("num_samples", 8)),
                    int(roll.get("max_new_tokens", 48)), float(roll.get("temperature", 1.0))))
            mres["imagined_next_shots"] = imagined

        for q in story.questions:
            beliefs = [belief(viewer, prefix(t, mode), q, n_perm, masses) for t in range(T + 1)]
            qres = {
                "beliefs": beliefs,
                "entropy": [entropy(b) for b in beliefs],
                "surprise": [None] + [l2(beliefs[t], beliefs[t - 1]) for t in range(1, T + 1)],
            }
            if do_roll:
                sus = []
                for t in range(T):
                    posts = [belief(viewer, prefix(t, mode) + [text(f"\nShot {t + 1} (imagined): {s}")],
                                    q, n_perm, masses) for s in imagined[t]]
                    sus.append(rollout_suspense(beliefs[t], posts))
                qres["suspense_rollout"] = sus + [None]  # 看完最后一个镜头后故事结束，悬念无定义
            mres["questions"][q.id] = qres

        if vcfg.get("direct_rating", False):
            ratings = [None]
            for t in range(1, T + 1):
                probs, m = viewer.choice_probs(prefix(t, mode) + [text(RATING_PROMPT)], list("12345"))
                masses.append(m)
                ratings.append(expected_rating(probs))
            mres["direct_rating"] = ratings
        result["modes"][mode] = mres

    if vcfg.get("probe_facts", False) and story.probe_facts:
        result["probes"] = {
            pf.id: {
                "question": pf.question,
                "expected_shots": pf.expected_shots,
                "p_yes": [probe_yes(viewer, frames[i], pf.question, masses) for i in range(T)],
            }
            for pf in story.probe_facts
        }

    result["label_mass_mean"] = sum(masses) / len(masses) if masses else None
    result["label_mass_min"] = min(masses) if masses else None
    return result


def main():
    p = base_arg_parser(__doc__)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load_config(args)
    stories = load_stories(cfg)

    units = [(st, v, int(s)) for st in stories for v in st.versions for s in cfg["seeds"]]
    mine = shard(units, args.rank, args.world_size)
    todo = []
    for st, v, s in mine:
        if not manifest_path(cfg, st.id, v, s).exists():
            log.warning("缺少 manifest（先运行 assemble）：%s/%s seed=%d", st.id, v, s)
            continue
        if args.overwrite or not result_path(cfg, st.id, v, s).exists():
            todo.append((st, v, s))
    log.info("rank %d/%d: 待评估 %d 个短片", args.rank, args.world_size, len(todo))
    if not todo:
        return

    viewer = MockViewer() if args.dry_run else QwenVLViewer(cfg["viewer"])
    for i, (st, v, s) in enumerate(todo, 1):
        t0 = time.time()
        res = evaluate_video(viewer, st, v, s, cfg)
        write_json(result_path(cfg, st.id, v, s), res)
        log.info("[%d/%d] %s/%s seed=%d 完成，用时 %.0fs，选项概率质量均值 %.2f",
                 i, len(todo), st.id, v, s, time.time() - t0, res["label_mass_mean"] or -1)


if __name__ == "__main__":
    main()
