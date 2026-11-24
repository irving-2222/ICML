"""
观众 Agent：用本地 VLM（默认 Qwen2.5-VL）模拟"第一次看片的观众"。

只提供两个底层能力：
  - choice_probs(content, labels): 读下一个 token 的概率，得到观众在各选项上的概率（信念）
  - generate_texts(content, n):    采样生成文字（用于"想象下一个镜头"）
content 是一个列表，元素为 {"type": "text", "text": str} 或 {"type": "image", "image": PIL.Image}
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

log = logging.getLogger("viewer")

DEFAULT_SYSTEM = (
    "You are an ordinary audience member watching a short film for the first time, shot by shot. "
    "You only know what has been shown so far. Answer honestly from your own expectations as a viewer."
)


def text(t: str) -> dict:
    return {"type": "text", "text": t}


def image(img) -> dict:
    return {"type": "image", "image": img}


class QwenVLViewer:
    def __init__(self, vcfg: dict):
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # 旧版 transformers
            from transformers import Qwen2_5_VLForConditionalGeneration as AutoVLM

        self.torch = torch
        model_id = vcfg["model_id"]
        log.info("加载观众模型 %s", model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, **vcfg.get("processor_kwargs", {}))
        self.model = AutoVLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=vcfg.get("device_map", "auto"),
            attn_implementation=vcfg.get("attn_implementation", "sdpa"),
        ).eval()
        self.tok = self.processor.tokenizer
        self.system = vcfg.get("system_prompt", DEFAULT_SYSTEM)
        self._label_cache: dict[str, int] = {}

    # ------------------------------------------------------------------
    def _build_inputs(self, content: list[dict]):
        msg, images = [], []
        for it in content:
            if it["type"] == "image":
                msg.append({"type": "image"})
                images.append(it["image"])
            else:
                msg.append({"type": "text", "text": it["text"]})
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": msg},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt")
        return inputs.to(self.model.device)

    def _label_id(self, label: str) -> int:
        if label not in self._label_cache:
            ids = self.tok.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                log.warning("选项标签 %r 被切成 %d 个 token，只用最后一个", label, len(ids))
            self._label_cache[label] = ids[-1]
        return self._label_cache[label]

    def choice_probs(self, content: list[dict], labels: list[str]) -> tuple[dict[str, float], float]:
        """返回 (在 labels 上重新归一化的概率, labels 原始概率质量之和)。质量太低说明模型没按格式回答。"""
        torch = self.torch
        inputs = self._build_inputs(content)
        with torch.no_grad():
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
        logits = out.logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        ids = torch.tensor([self._label_id(l) for l in labels], device=logp.device)
        lp = logp[ids]
        mass = float(lp.exp().sum())
        probs = torch.softmax(lp, dim=-1).tolist()
        return dict(zip(labels, probs)), mass

    def generate_texts(self, content: list[dict], n: int, max_new_tokens: int = 48,
                       temperature: float = 1.0) -> list[str]:
        torch = self.torch
        inputs = self._build_inputs(content)
        L = inputs["input_ids"].shape[1]
        gen_kwargs = dict(do_sample=True, temperature=temperature, top_p=0.95, max_new_tokens=max_new_tokens)
        with torch.no_grad():
            try:
                out = self.model.generate(**inputs, num_return_sequences=n, **gen_kwargs)
                seqs = out[:, L:]
            except Exception as e:  # noqa: BLE001  部分版本不支持多图 + num_return_sequences
                log.debug("num_return_sequences 失败，改为逐条生成: %r", e)
                seqs = [self.model.generate(**inputs, **gen_kwargs)[0, L:] for _ in range(n)]
        texts = [self.tok.decode(s, skip_special_tokens=True).strip() for s in seqs]
        return [t.split("\n")[0] for t in texts]


class MockViewer:
    """不需要 GPU：返回由输入内容决定的伪随机结果，只用来测试流程。"""

    def choice_probs(self, content, labels):
        key = "|".join(
            it["text"] if it["type"] == "text" else hashlib.md5(it["image"].tobytes()).hexdigest()[:8]
            for it in content
        )
        seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        p = np.random.default_rng(seed).dirichlet(np.ones(len(labels)))
        return dict(zip(labels, p.tolist())), 0.9

    def generate_texts(self, content, n, max_new_tokens=48, temperature=1.0):
        return [f"mock imagined next shot #{i}" for i in range(n)]
