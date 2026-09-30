"""图片类步骤的 runner（ADR 0004）：文生图、去背景、放大、裁透明边、统一尺寸。"""
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.core.collection.generator_runner import GeneratorStepRunner, move_into
from src.backend.core.collection.steps import StepContext, StepRunner


class ImageGenerateRunner(GeneratorStepRunner):
    type_name = "image.generate"

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items()
                  if k not in ("model_name", "prompt", "prompt_suffix")}
        suffix = ctx.params.get("prompt_suffix", "")
        # 参数 prompt 换掉条目主提示词：同一条目要画两样东西时（背景 + 地面），
        # 第二样的描述放在字段里，用 "{字段}" 引过来；风格锁照样加，两张图才统一
        if "prompt" in ctx.params:
            subject = [ctx.params["prompt"], ctx.style_prompt]
        else:
            subject = [ctx.prompt]
        params["content"] = ", ".join(p.strip() for p in (*subject, suffix) if p.strip())
        if ctx.negative_prompt:
            params["negative_prompt"] = ctx.negative_prompt
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        paths = super().run(ctx)
        ctx.meta.update(describe_image(paths[0]))
        return paths


class ImageInputRunner(GeneratorStepRunner):
    """以上游产物为输入的图片步骤（去背景、放大）。

    上游产出多张图时（如分层的 N 个图层）逐张处理，产物按输入顺序命名为 <步骤>_1.png …；
    只有一张时仍叫 <步骤>.png，已有资源包里的文件名不变。
    """

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        return self._params(ctx, single_input(self.type_name, ctx))

    def _params(self, ctx: StepContext, source: Path) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params["input_path"] = str(source)
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        if len(ctx.inputs) <= 1:
            paths = super().run(ctx)
        else:
            paths = []
            for index, source in enumerate(ctx.inputs, start=1):
                params = self._params(ctx, source)
                path = self.generate_once(params)
                target = ctx.out_dir / f"{ctx.step_id}_{index}{self.output_suffix}"
                paths.append(move_into(path, target))
            ctx.meta.update(self.describe(params))
        ctx.meta.update(describe_image(paths[0]))
        return paths


class RemoveBackgroundRunner(ImageInputRunner):
    type_name = "image.remove_bg"


class UpscaleRunner(ImageInputRunner):
    type_name = "image.upscale"


class TrimRunner(StepRunner):
    """裁掉透明边（纯 CPU，不占租约）：去背景后主体四周常留大片透明，导入引擎前裁掉。"""
    type_name = "image.trim"

    def run(self, ctx: StepContext) -> list[Path]:
        padding = int(ctx.params.get("padding", 4))
        with Image.open(single_input(self.type_name, ctx)) as image:
            trimmed = trim_transparent(image.convert("RGBA"), padding)
        path = ctx.out_dir / f"{ctx.step_id}.png"
        trimmed.save(path)
        ctx.meta.update(describe_image(path))
        return [path]


class ResizeRunner(StepRunner):
    """等比缩放后居中放进 size×size 的透明画布（纯 CPU）：图标要统一尺寸，引擎里才能直接对齐。"""
    type_name = "image.resize"

    def run(self, ctx: StepContext) -> list[Path]:
        size = int(ctx.params.get("size", 256))
        if size <= 0:
            raise ValueError(f"{self.type_name} 的 size 必须大于 0：{size}")
        with Image.open(single_input(self.type_name, ctx)) as image:
            resized = fit_square(image.convert("RGBA"), size)
        path = ctx.out_dir / f"{ctx.step_id}.png"
        resized.save(path)
        ctx.meta.update(describe_image(path))
        return [path]


def single_input(type_name: str, ctx: StepContext) -> Path:
    if len(ctx.inputs) != 1:
        raise ValueError(f"{type_name} 需要恰好一张输入图，实际 {len(ctx.inputs)} 张")
    return ctx.inputs[0]


def trim_transparent(image: Image.Image, padding: int = 0) -> Image.Image:
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("图片完全透明，去背景可能把主体也去掉了")
    left, top, right, bottom = box
    return image.crop((max(left - padding, 0), max(top - padding, 0),
                       min(right + padding, image.width), min(bottom + padding, image.height)))


def fit_square(image: Image.Image, size: int) -> Image.Image:
    scale = size / max(image.width, image.height)
    width, height = max(round(image.width * scale), 1), max(round(image.height * scale), 1)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS),
                 ((size - width) // 2, (size - height) // 2))
    return canvas


def describe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"size": [image.width, image.height]}


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    ImageGenerateRunner(),
    RemoveBackgroundRunner(),
    UpscaleRunner(),
    TrimRunner(),
    ResizeRunner(),
)
