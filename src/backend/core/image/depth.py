"""Depth Anything V2 Small（Apache-2.0，可商用；同系列 Base / Large 的权重许可有冲突，不用）。

只给循环拼接找切段位置用：估出长图每个像素的相对远近。不是生成器，不进 models.json；
占显存不到 0.5GB，但和驻留的生图模型同时在也可能挤爆 8GB，所以用 exclusive() 独占，用完即释放。
"""
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import ensure_snapshot, models_root

REPO_ID = "depth-anything/Depth-Anything-V2-Small-hf"
# 模型输入会被缩到约 518 像素，宽图整张喂会糊成一片：按窗口分块估，重叠一半用三角权重混合
WINDOW = 1024
STRIDE = 512


def _model_dir() -> Path:
    return models_root() / "depth" / "Depth-Anything-V2-Small-hf"


def estimate_ring_depth(image: Image.Image,
                        cancel_event: threading.Event | None = None) -> np.ndarray:
    """返回 (高, 宽) 的相对逆深度，0~1，越大越近。image 按环处理：首尾相接。"""
    ensure_snapshot(REPO_ID, _model_dir())
    with GeneratorFactory.exclusive("深度估计"):
        return _run(image, cancel_event)


def _run(image: Image.Image, cancel_event: threading.Event | None) -> np.ndarray:
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(str(_model_dir()))
    model = AutoModelForDepthEstimation.from_pretrained(str(_model_dir())).to(device).eval()
    try:
        image = image.convert("RGB")
        width, height = image.size
        window = min(WINDOW, width)
        stride = max(1, window // 2)
        # 把开头一个窗口接到末尾：首尾交界处也有完整的左右上下文
        padded = Image.new("RGB", (width + window, height))
        padded.paste(image, (0, 0))
        padded.paste(image.crop((0, 0, window, height)), (width, 0))
        depth = np.zeros((height, padded.width), np.float32)
        weight = np.zeros_like(depth)
        ramp = 1.0 - np.abs(np.linspace(-1, 1, window)) + 1e-3
        for x in range(0, padded.width - window + 1, stride):
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("深度估计已取消")
            crop = padded.crop((x, 0, x + window, height))
            inputs = processor(images=crop, return_tensors="pt").to(device)
            with torch.no_grad():
                predicted = model(**inputs).predicted_depth
            part = torch.nn.functional.interpolate(
                predicted.unsqueeze(1), size=(height, window), mode="bicubic",
                align_corners=False).squeeze().float().cpu().numpy()
            # 每块的相对深度尺度不同：按分位归一后再叠，块与块之间才接得上
            low, high = np.percentile(part, [2, 98])
            part = np.clip((part - low) / max(high - low, 1e-6), 0, 1)
            depth[:, x:x + window] += part * ramp
            weight[:, x:x + window] += ramp
        depth /= np.maximum(weight, 1e-6)
        # 开头只被单侧的块覆盖，末尾补上的那段才有两侧混合：用它替换开头
        depth[:, :stride] = depth[:, width:width + stride]
        return depth[:, :width]
    finally:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
