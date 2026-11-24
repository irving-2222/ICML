"""
第 1 步：逐镜头生成视频片段。

每个 (故事, 镜头, seed) 是一个任务；两个版本共享的镜头只生成一次。
多卡并行：每张卡起一个进程，按 rank 分任务（见 scripts/run_generate.sh）。
已存在的片段会跳过，可随时中断后重跑。
"""
from __future__ import annotations

import time

from .common import base_arg_parser, clip_path, load_config, load_stories, setup_logging, shard, write_json
from .generators import MockGenerator, WanGenerator
from .video_io import save_video

log = setup_logging("generate")


def main():
    p = base_arg_parser(__doc__)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    cfg = load_config(args)
    stories = load_stories(cfg)

    jobs = []
    for st in stories:
        for sid in st.used_shot_ids():
            for seed in cfg["seeds"]:
                jobs.append((st, st.shots[sid], int(seed)))
    mine = shard(jobs, args.rank, args.world_size)
    todo = [j for j in mine if args.overwrite or not clip_path(cfg, j[0].id, j[1].id, j[2]).exists()]
    log.info("rank %d/%d: 共 %d 个任务，本进程 %d 个，待生成 %d 个",
             args.rank, args.world_size, len(jobs), len(mine), len(todo))
    if not todo:
        return

    if args.dry_run:
        gen = MockGenerator()
    else:
        gen = WanGenerator(cfg["generator"], prompts=[shot.prompt for _, shot, _ in todo])

    for i, (st, shot, seed) in enumerate(todo, 1):
        out = clip_path(cfg, st.id, shot.id, seed)
        t0 = time.time()
        frames = gen.generate(shot.prompt, seed)
        save_video(frames, out, fps=gen.fps)
        dt = time.time() - t0
        write_json(out.with_suffix(".json"), {
            "story": st.id, "shot": shot.id, "seed": seed, "prompt": shot.prompt,
            "seconds": round(dt, 1), "num_frames": len(frames),
        })
        log.info("[%d/%d] %s/%s seed=%d 完成，用时 %.0fs", i, len(todo), st.id, shot.id, seed, dt)


if __name__ == "__main__":
    main()
