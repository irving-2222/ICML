"""
第 2 步：把镜头片段按版本拼成完整短片。

- 写一个 manifest（镜头顺序 + 片段路径），观众 Agent 按镜头读取；
- 同时拼一个完整 mp4，方便人看 / 后续做人工评测。
同一 seed 下，两个版本的公共镜头是同一个文件 -> 只有 reveal 镜头不同。
"""
from __future__ import annotations

from .common import (base_arg_parser, clip_path, load_config, load_stories, manifest_path,
                     setup_logging, video_path, write_json)
from .video_io import read_frames, save_video

log = setup_logging("assemble")


def main():
    p = base_arg_parser(__doc__)
    p.add_argument("--no_mp4", action="store_true", help="只写 manifest，不拼 mp4")
    args = p.parse_args()
    cfg = load_config(args)

    n_ok = n_missing = 0
    for st in load_stories(cfg):
        for version, seq in st.versions.items():
            for seed in cfg["seeds"]:
                clips = [clip_path(cfg, st.id, sid, seed) for sid in seq]
                missing = [str(c) for c in clips if not c.exists()]
                if missing:
                    log.warning("%s/%s seed=%d 缺少 %d 个片段，跳过", st.id, version, seed, len(missing))
                    n_missing += 1
                    continue
                write_json(manifest_path(cfg, st.id, version, seed), {
                    "story": st.id, "version": version, "seed": seed,
                    "shots": seq, "clips": [str(c) for c in clips],
                    "reveal_index": st.reveal_index, "final_index": st.final_index,
                })
                if not args.no_mp4:
                    all_frames, fps = [], 16.0
                    for c in clips:
                        frames, fps = read_frames(c)
                        all_frames.extend(frames)
                    save_video(all_frames, video_path(cfg, st.id, version, seed), fps=round(fps))
                n_ok += 1
    log.info("完成：%d 个短片已拼好，%d 个因缺片段跳过", n_ok, n_missing)


if __name__ == "__main__":
    main()
