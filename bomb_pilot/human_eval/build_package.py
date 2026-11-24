"""
生成真人打分用的离线网页包。

用法（在 bomb_pilot/ 目录下）：
    python human_eval/build_package.py                 # 默认 36 个打分人
    python human_eval/build_package.py --n_raters 24

产出：
    human_eval/package/          网页包（index.html + data.js + clips/ + 打分说明）
    human_eval/package.zip       打包好的压缩文件，直接发给打分人
    human_eval/assignments.csv   编号分配表：每个编号看哪 3 段（组织者自己留着，不要发给打分人）

分配原则：
    - 每人只看 3 段，每个故事 1 段（看过炸弹版再看手提包版就不会"意外"了，所以同一故事只能看一次）
    - 每人看到的版本是混合的（不会 3 段全是同一版本），故事顺序也轮换
    - 每 6 个人为一轮，每个（故事, 版本, seed）恰好被看 1 次；人数取 6 的倍数最均衡
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from src.common import load_story  # noqa: E402


def make_assignments(story_ids: list[str], versions: list[str], seeds: list[int], n: int) -> list[dict]:
    combos = [(v, s) for s in seeds for v in versions]  # 每个故事 6 种（版本 × seed）
    orders = list(itertools.permutations(range(len(story_ids))))
    out = []
    for r in range(n):
        vids = []
        for i, sid in enumerate(story_ids):
            v, s = combos[(r + i) % len(combos)]
            vids.append({"story": sid, "version": v, "seed": s})
        order = orders[(r // len(combos)) % len(orders)]
        out.append({"rater": f"R{r + 1:02d}", "videos": [vids[j] for j in order]})
    return out


def encode(src: Path, dst: Path, crf: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None or crf <= 0:
        shutil.copy2(src, dst)
        return
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", str(crf),
           "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stories", nargs="+", default=["configs/stories/cafe_bomb.json",
                                                    "configs/stories/poisoned_wine.json",
                                                    "configs/stories/thin_ice.json"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--clips_dir", default="outputs/bomb_pilot/clips")
    p.add_argument("--n_raters", type=int, default=36)
    p.add_argument("--crf", type=int, default=28, help="视频压缩质量（越大文件越小）；0 表示直接复制不压缩")
    p.add_argument("--out", default=str(HERE / "package"))
    args = p.parse_args()

    zh = json.loads((HERE / "stories_zh.json").read_text(encoding="utf-8"))
    clips_dir = (ROOT / args.clips_dir) if not Path(args.clips_dir).is_absolute() else Path(args.clips_dir)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "clips").mkdir(parents=True)

    stories_js, story_ids, versions = {}, [], None
    for sp in args.stories:
        story = load_story(sp)
        if story.id not in zh:
            raise SystemExit(f"stories_zh.json 里缺少故事 {story.id} 的中文文本")
        t = zh[story.id]
        missing = set(story.questions[0].options) - set(t["options"])
        if missing:
            raise SystemExit(f"{story.id} 的中文选项缺少: {missing}")
        stories_js[story.id] = {
            "intro": t["intro"], "question": t["question"],
            "options": {k: t["options"][k] for k in story.questions[0].options},
            "versions": story.versions,
            "reveal_index": story.reveal_index, "final_index": story.final_index,
            "harm_option": story.questions[0].harm_option,
        }
        story_ids.append(story.id)
        versions = versions or list(story.versions)
        for sid in story.used_shot_ids():
            for s in args.seeds:
                src = clips_dir / story.id / f"{sid}_seed{s}.mp4"
                if not src.exists():
                    raise SystemExit(f"缺少片段: {src}")
                encode(src, out / "clips" / story.id / f"{sid}_seed{s}.mp4", args.crf)
        print(f"[{story.id}] 片段已处理")

    assignments = make_assignments(story_ids, versions, args.seeds, args.n_raters)
    data = {"stories": stories_js, "assignments": assignments}
    (out / "data.js").write_text("window.STUDY = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n",
                                 encoding="utf-8")
    shutil.copy2(HERE / "template" / "index.html", out / "index.html")
    shutil.copy2(HERE / "instructions_for_raters.md", out / "打分说明.md")

    with open(HERE / "assignments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "打分人姓名（组织者填写）", "第1段", "第2段", "第3段"])
        for a in assignments:
            w.writerow([a["rater"], ""] + [f'{v["story"]}/{v["version"]}/seed{v["seed"]}' for v in a["videos"]])

    zip_path = shutil.make_archive(str(HERE / "package"), "zip", root_dir=out.parent, base_dir=out.name)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"完成：{out}（{size:.0f} MB）\n压缩包：{zip_path}\n分配表：{HERE / 'assignments.csv'}")

    # 检查分配是否均衡
    cnt = {}
    for a in assignments:
        for v in a["videos"]:
            k = (v["story"], v["version"])
            cnt[k] = cnt.get(k, 0) + 1
    print("每个（故事, 版本）被看的次数：", {f"{k[0]}/{k[1]}": c for k, c in sorted(cnt.items())})


if __name__ == "__main__":
    main()
