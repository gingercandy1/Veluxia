"""图片类步骤的 runner（ADR 0004）：文生图、去背景、放大、裁透明边。

生成器的 generate() 名义上是协程、实际是同步阻塞调用，这里用 asyncio.run 驱动，
所以执行器必须跑在没有事件循环的线程里（任务里用 asyncio.to_thread 调用）。
"""
import asyncio
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType


class GeneratorStepRunner(StepRunner):
    """一组条目共用一次租约：open() 取租约并加载模型，run() 逐条生成。"""
    factory_type = FactoryType.Image

    def __init__(self):
        self._generator = None

    @contextmanager
    def open(self, params: dict[str, Any], cancel_event: threading.Event):
        model_name = params.get("model_name")
        if not model_name:
            raise ValueError(f"步骤 {self.type_name} 缺少 model_name 参数")
        with GeneratorFactory.acquire(self.factory_type, model_name) as generator:
            generator.cancel_event = cancel_event
            generator.ensure_model_loaded()
            self._generator = generator
            try:
                yield
            finally:
                self._generator = None

    def run(self, ctx: StepContext) -> list[Path]:
        if self._generator is None:
            raise RuntimeError(f"步骤 {self.type_name} 未在 open() 内调用")
        self._generator.check_cancelled()
        self._generator.parse_params(self.build_params(ctx))
        path = asyncio.run(self._generator.generate())
        if not path or not Path(path).is_file():
            # 生成器失败时只打印日志并返回 None，这里必须转成异常，否则会被当成成功
            raise RuntimeError(f"{self.type_name} 没有生成文件，请查看后端日志")
        return [_move_into(Path(path), ctx.out_dir / f"{ctx.step_id}.png")]

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        raise NotImplementedError


class ImageGenerateRunner(GeneratorStepRunner):
    type_name = "image.generate"

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k not in ("model_name", "prompt_suffix")}
        suffix = ctx.params.get("prompt_suffix", "")
        params["content"] = ", ".join(p for p in (ctx.prompt, suffix) if p)
        if ctx.negative_prompt:
            params["negative_prompt"] = ctx.negative_prompt
        return params


class ImageInputRunner(GeneratorStepRunner):
    """以上游产物为输入的图片步骤（去背景、放大）。"""

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        if len(ctx.inputs) != 1:
            raise ValueError(f"{self.type_name} 需要恰好一张输入图，实际 {len(ctx.inputs)} 张")
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params["input_path"] = str(ctx.inputs[0])
        return params


class RemoveBackgroundRunner(ImageInputRunner):
    type_name = "image.remove_bg"


class UpscaleRunner(ImageInputRunner):
    type_name = "image.upscale"


class TrimRunner(StepRunner):
    """裁掉透明边（纯 CPU，不占租约）：去背景后主体四周常留大片透明，导入引擎前裁掉。"""
    type_name = "image.trim"

    def run(self, ctx: StepContext) -> list[Path]:
        if len(ctx.inputs) != 1:
            raise ValueError(f"{self.type_name} 需要恰好一张输入图，实际 {len(ctx.inputs)} 张")
        padding = int(ctx.params.get("padding", 4))
        with Image.open(ctx.inputs[0]) as image:
            trimmed = trim_transparent(image.convert("RGBA"), padding)
        path = ctx.out_dir / f"{ctx.step_id}.png"
        trimmed.save(path)
        return [path]


def trim_transparent(image: Image.Image, padding: int = 0) -> Image.Image:
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("图片完全透明，去背景可能把主体也去掉了")
    left, top, right, bottom = box
    return image.crop((max(left - padding, 0), max(top - padding, 0),
                       min(right + padding, image.width), min(bottom + padding, image.height)))


def _move_into(source: Path, target: Path) -> Path:
    """生成器写在默认输出目录，移进资源包并用固定文件名，manifest 里的路径才稳定。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return target


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    ImageGenerateRunner(),
    RemoveBackgroundRunner(),
    UpscaleRunner(),
    TrimRunner(),
)
