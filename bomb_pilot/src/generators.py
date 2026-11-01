"""视频生成器：本地 Wan（diffusers）+ 用于跑通流程的 Mock。"""
from __future__ import annotations

import gc
import hashlib
import logging
from functools import wraps

import numpy as np

log = logging.getLogger("generators")


def _patch_metax_sdpa(torch) -> None:
    """MetaX torch 2.4 的 SDPA 不接受 enable_gqa；仅省略关闭 GQA 时的参数。"""
    version = str(torch.__version__)
    if not (version.startswith("2.4.") and "metax" in version):
        return
    native_sdpa = torch.nn.functional.scaled_dot_product_attention
    if getattr(native_sdpa, "_bomb_metax_sdpa_compat", False):
        return

    @wraps(native_sdpa)
    def compatible_sdpa(*args, **kwargs):
        if kwargs.get("enable_gqa") is False:
            kwargs.pop("enable_gqa")
        return native_sdpa(*args, **kwargs)

    compatible_sdpa._bomb_metax_sdpa_compat = True
    torch.nn.functional.scaled_dot_product_attention = compatible_sdpa
    log.info("MetaX torch 2.4 SDPA 兼容：省略 enable_gqa=False，保留原注意力计算")


class MockGenerator:
    """不需要 GPU：生成颜色随 prompt 变化的小视频，只用来测试流程。"""

    fps = 8

    def generate(self, prompt: str, seed: int):
        h = int(hashlib.md5(f"{prompt}|{seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        base = rng.random(3)
        frames = []
        for t in range(16):
            img = np.ones((64, 96, 3), dtype=np.float32) * base
            img[:, : (t * 6) % 96] *= 0.5  # 一个移动的色块，模拟"运动"
            frames.append(img)
        return frames


class WanGenerator:
    """
    通过 diffusers 本地调用 Wan2.1 / Wan2.2。

    省显存技巧（precompute_text_embeds=true）：
      1) 先只加载 T5 文本编码器，把本进程要用的所有 prompt 编码好，放到 CPU；
      2) 释放 T5（约 11GB），再加载不带文本编码器的主 pipeline。
    若这条路径因 diffusers 版本差异失败，会自动回退到普通加载方式。
    """

    def __init__(self, gcfg: dict, prompts: list[str]):
        import torch

        _patch_metax_sdpa(torch)
        from diffusers import AutoencoderKLWan, WanPipeline

        self.torch = torch
        preset_name = gcfg["preset"]
        self.p = gcfg["presets"][preset_name]
        self.neg = " ".join(gcfg.get("negative_prompt", "").split())
        self.max_seq = int(gcfg.get("max_sequence_length", 512))
        model_id = self.p["model_id"]
        self.fps = int(self.p["fps"])
        log.info("加载视频模型 %s（preset=%s）", model_id, preset_name)

        self.embeds: dict[str, tuple] = {}
        if self.p.get("precompute_text_embeds", True):
            try:
                self.embeds = self._encode_all(WanPipeline, model_id, sorted(set(prompts)))
            except Exception as e:  # noqa: BLE001
                log.warning("预计算文本向量失败，回退到普通模式: %r", e)
                self.embeds = {}
                gc.collect()
                torch.cuda.empty_cache()

        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        extra = {"text_encoder": None} if self.embeds else {}
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16, **extra)

        if self.p.get("flow_shift"):
            from diffusers import UniPCMultistepScheduler

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config, flow_shift=float(self.p["flow_shift"])
            )

        offload = self.p.get("cpu_offload", "auto")
        if offload == "auto":
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            offload = total_gb < float(self.p.get("min_gpu_mem_gb_no_offload", 0))
            log.info("GPU 显存 %.1f GB -> cpu_offload=%s", total_gb, offload)
        if offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe

    def _encode_all(self, WanPipeline, model_id: str, prompts: list[str]) -> dict:
        torch = self.torch
        none_kwargs = {"transformer": None, "vae": None}
        if self.p.get("has_transformer_2"):
            none_kwargs["transformer_2"] = None
        tp = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, **none_kwargs)
        tp.text_encoder.to("cuda")
        out = {}
        with torch.no_grad():
            for p in prompts:
                pe, npe = tp.encode_prompt(
                    prompt=p,
                    negative_prompt=self.neg,
                    do_classifier_free_guidance=True,
                    num_videos_per_prompt=1,
                    max_sequence_length=self.max_seq,
                    device=torch.device("cuda"),
                )
                out[p] = (pe.cpu(), npe.cpu())
        del tp
        gc.collect()
        torch.cuda.empty_cache()
        log.info("已预计算 %d 条 prompt 的文本向量", len(out))
        return out

    def generate(self, prompt: str, seed: int):
        torch = self.torch
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        common = dict(
            height=int(self.p["height"]),
            width=int(self.p["width"]),
            num_frames=int(self.p["num_frames"]),
            num_inference_steps=int(self.p["num_inference_steps"]),
            generator=g,
            **self.p.get("call_kwargs", {}),
        )
        with torch.no_grad():
            if prompt in self.embeds:
                pe, npe = self.embeds[prompt]
                out = self.pipe(prompt_embeds=pe.to("cuda"), negative_prompt_embeds=npe.to("cuda"), **common)
            else:
                out = self.pipe(
                    prompt=prompt, negative_prompt=self.neg, max_sequence_length=self.max_seq, **common
                )
        return out.frames[0]
