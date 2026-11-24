"""视频读写与抽帧。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(frame) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.asarray(frame.convert("RGB"))
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)
    return arr


def save_video(frames, path: str | Path, fps: int) -> None:
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    imageio.mimwrite(str(tmp), [_to_uint8(f) for f in frames], fps=fps, codec="libx264", quality=8)
    tmp.replace(path)


def read_frames(path: str | Path) -> tuple[list[np.ndarray], float]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    fps = float(reader.get_meta_data().get("fps", 16))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return frames, fps


def sample_frames(path: str | Path, n: int, max_side: int) -> list[Image.Image]:
    """均匀抽 n 帧（取每段的中点），并把长边缩放到 max_side。"""
    frames, _ = read_frames(path)
    if not frames:
        raise RuntimeError(f"视频没有帧: {path}")
    total = len(frames)
    idx = [min(total - 1, int((i + 0.5) * total / n)) for i in range(n)]
    out = []
    for i in idx:
        img = Image.fromarray(frames[i]).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
        out.append(img)
    return out
