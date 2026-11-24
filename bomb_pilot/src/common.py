"""公共工具：配置加载、故事数据结构、输出路径。"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def base_arg_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment.yaml"))
    p.add_argument("--output_dir", default=None, help="覆盖配置文件里的 output_dir")
    p.add_argument("--dry_run", action="store_true", help="用假模型跑通整个流程（不需要 GPU）")
    return p


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.output_dir:
        out = args.output_dir
    elif args.dry_run:
        out = cfg.get("dry_run_output_dir", "outputs/dry_run")
    else:
        out = cfg["output_dir"]
    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    cfg["output_dir"] = str(out)
    return cfg


# ---------------------------------------------------------------------------
# 故事数据结构
# ---------------------------------------------------------------------------
@dataclass
class Shot:
    id: str
    prompt: str        # 给视频生成模型的完整 prompt（已填入角色描述和风格）
    description: str   # 给"观众"读的纯文字描述（计划层对照用）


@dataclass
class Question:
    id: str
    text: str
    options: dict[str, str]  # 选项键 -> 选项文字（保持顺序）
    harm_option: str         # 画图时重点关注的"出事"选项


@dataclass
class ProbeFact:
    id: str
    question: str             # 是/否问题
    expected_shots: list[str]  # 这个事实"应该"出现在哪些镜头里


@dataclass
class Story:
    id: str
    title: str
    intro: str
    shots: dict[str, Shot]
    versions: dict[str, list[str]]
    questions: list[Question]
    probe_facts: list[ProbeFact]
    reveal_index: int   # 两个版本不同的那个镜头的位置（0 开始）
    final_index: int    # 结局镜头的位置（0 开始）

    def sequence(self, version: str) -> list[Shot]:
        return [self.shots[s] for s in self.versions[version]]

    def used_shot_ids(self) -> list[str]:
        seen: list[str] = []
        for seq in self.versions.values():
            for s in seq:
                if s not in seen:
                    seen.append(s)
        return seen


def load_story(path: str | Path) -> Story:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    chars: dict[str, str] = raw.get("characters", {})
    style: str = raw.get("style", "")

    def fill(text: str) -> str:
        for k, v in chars.items():
            text = text.replace("{" + k + "}", v)
        return text

    shots = {}
    for sid, s in raw["shots"].items():
        prompt = fill(s["prompt"])
        if style:
            prompt = f"{prompt} {style}"
        shots[sid] = Shot(id=sid, prompt=prompt, description=s["description"])

    questions = [
        Question(id=q["id"], text=q["text"], options=dict(q["options"]), harm_option=q["harm_option"])
        for q in raw["questions"]
    ]
    probes = [
        ProbeFact(id=p["id"], question=p["question"], expected_shots=list(p["expected_shots"]))
        for p in raw.get("probe_facts", [])
    ]
    story = Story(
        id=raw["id"],
        title=raw.get("title", raw["id"]),
        intro=raw["intro"],
        shots=shots,
        versions={k: list(v) for k, v in raw["versions"].items()},
        questions=questions,
        probe_facts=probes,
        reveal_index=int(raw["reveal_index"]),
        final_index=int(raw["final_index"]),
    )
    _validate(story)
    return story


def _validate(story: Story) -> None:
    lengths = {len(v) for v in story.versions.values()}
    if len(lengths) != 1:
        raise ValueError(f"[{story.id}] 各版本镜头数必须相同（保证时长一致），现在是 {lengths}")
    n = lengths.pop()
    for ver, seq in story.versions.items():
        for sid in seq:
            if sid not in story.shots:
                raise ValueError(f"[{story.id}] 版本 {ver} 引用了不存在的镜头 {sid}")
    if not (0 <= story.reveal_index < story.final_index < n):
        raise ValueError(f"[{story.id}] 需要 0 <= reveal_index < final_index < 镜头数")
    for q in story.questions:
        if q.harm_option not in q.options:
            raise ValueError(f"[{story.id}] 问题 {q.id} 的 harm_option 不在选项里")
    # 检查：两个版本除了 reveal_index 位置外应完全相同（严格对照）
    seqs = list(story.versions.values())
    for i in range(n):
        if i == story.reveal_index:
            continue
        if len({seq[i] for seq in seqs}) != 1:
            logging.getLogger("common").warning(
                "[%s] 位置 %d 在不同版本中镜头不同，实验将不再是严格单变量对照", story.id, i
            )


def load_stories(cfg: dict) -> list[Story]:
    return [load_story(p) for p in cfg["stories"]]


# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------
def out_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"])


def clip_path(cfg: dict, story_id: str, shot_id: str, seed: int) -> Path:
    return out_dir(cfg) / "clips" / story_id / f"{shot_id}_seed{seed}.mp4"


def manifest_path(cfg: dict, story_id: str, version: str, seed: int) -> Path:
    return out_dir(cfg) / "videos" / story_id / f"{version}_seed{seed}.json"


def video_path(cfg: dict, story_id: str, version: str, seed: int) -> Path:
    return out_dir(cfg) / "videos" / story_id / f"{version}_seed{seed}.mp4"


def result_path(cfg: dict, story_id: str, version: str, seed: int) -> Path:
    return out_dir(cfg) / "viewer" / story_id / f"{version}_seed{seed}.json"


def shard(items: list, rank: int, world_size: int) -> list:
    return items[rank::world_size]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
