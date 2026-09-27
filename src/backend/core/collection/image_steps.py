"""图片类步骤的 runner（ADR 0004）：文生图、去背景、放大、裁透明边、统一尺寸、色调对齐、空气透视、横向无缝、图层合成。"""
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.core.collection.generator_runner import GeneratorStepRunner
from src.backend.core.collection.steps import StepContext, StepRunner


class ImageGenerateRunner(GeneratorStepRunner):
    type_name = "image.generate"

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k not in ("model_name", "prompt_suffix")}
        suffix = ctx.params.get("prompt_suffix", "")
        params["content"] = ", ".join(p for p in (ctx.prompt, suffix) if p)
        if ctx.negative_prompt:
            params["negative_prompt"] = ctx.negative_prompt
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        paths = super().run(ctx)
        ctx.meta.update(describe_image(paths[0]))
        return paths


class ImageInputRunner(GeneratorStepRunner):
    """以上游产物为输入的图片步骤（去背景、放大）。"""

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params["input_path"] = str(single_input(self.type_name, ctx))
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        paths = super().run(ctx)
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


class ColorMatchRunner(StepRunner):
    """把第一张输入的色调对齐到第二张参考图（纯 CPU）：视差各层分开生成，色温、色相各不相同，
    叠在一起就穿帮。

    只迁移 LAB 的 a/b 色彩通道、不动亮度 L：近景本该比远景暗、对比强，连亮度一起对齐会把景深压平。
    """
    type_name = "image.color_match"

    def run(self, ctx: StepContext) -> list[Path]:
        if len(ctx.inputs) != 2:
            raise ValueError(f"{self.type_name} 需要目标图和参考图两张输入，实际 {len(ctx.inputs)} 张")
        strength = float(ctx.params.get("strength", 0.7))
        if not 0 <= strength <= 1:
            raise ValueError(f"{self.type_name} 的 strength 必须在 0 到 1 之间：{strength}")
        target_path, reference_path = ctx.inputs
        with Image.open(target_path) as target, Image.open(reference_path) as reference:
            matched = match_color(target.convert("RGBA"), reference.convert("RGBA"), strength)
        path = ctx.out_dir / f"{ctx.step_id}.png"
        matched.save(path)
        ctx.meta.update(describe_image(path))
        return [path]


class AtmosphereRunner(StepRunner):
    """空气透视（纯 CPU）：把图层颜色往远景的雾色推，越远推得越多。

    色调对齐只让各层"同一种光"，但对比度、饱和度一样，叠起来远近拉不开、显得平。
    真实远处物体因大气散射会变淡、变灰、趋向天空色；混向远景平均色一步就同时压低了
    对比和饱和度。第一张输入是图层，第二张是远景参考；haze 是混合比例（0 不变，1 全成雾色）。
    """
    type_name = "image.atmosphere"

    def run(self, ctx: StepContext) -> list[Path]:
        if len(ctx.inputs) != 2:
            raise ValueError(f"{self.type_name} 需要图层和远景参考两张输入，实际 {len(ctx.inputs)} 张")
        haze = float(ctx.params.get("haze", 0.25))
        if not 0 <= haze <= 1:
            raise ValueError(f"{self.type_name} 的 haze 必须在 0 到 1 之间：{haze}")
        layer_path, reference_path = ctx.inputs
        with Image.open(layer_path) as layer, Image.open(reference_path) as reference:
            result = apply_haze(layer.convert("RGBA"), reference.convert("RGBA"), haze)
        path = ctx.out_dir / f"{ctx.step_id}.png"
        result.save(path)
        ctx.meta.update(describe_image(path))
        return [path]


class TileHorizontalRunner(StepRunner):
    """把图片左右边缘做成无缝衔接（纯 CPU）：视差背景层在引擎里横向平铺滚动，接缝不能露出来。

    做法同 audio.loop：右端 overlap 宽度淡出、叠到左端的淡入上，再去掉右端这段，
    平铺时右边缘接着的正是"被叠进左端的那段右端"，两侧都连续。
    """
    type_name = "image.tile_x"

    def run(self, ctx: StepContext) -> list[Path]:
        overlap_ratio = float(ctx.params.get("overlap", 0.125))
        with Image.open(single_input(self.type_name, ctx)) as image:
            if not 0 < overlap_ratio < 0.5:
                raise ValueError(f"{self.type_name} 的 overlap 必须在 0 到 0.5 之间：{overlap_ratio}")
            tiled = tile_horizontal(image.convert("RGBA"), round(image.width * overlap_ratio))
        path = ctx.out_dir / f"{ctx.step_id}.png"
        tiled.save(path)
        ctx.meta.update(describe_image(path))
        return [path]


class CompositeRunner(StepRunner):
    """按输入顺序从后往前叠图层（纯 CPU）：分层背景看不出整体效果，合成一张预览当封面。"""
    type_name = "image.composite"

    def run(self, ctx: StepContext) -> list[Path]:
        if len(ctx.inputs) < 2:
            raise ValueError(f"{self.type_name} 至少需要两张输入图，实际 {len(ctx.inputs)} 张")
        with Image.open(ctx.inputs[0]) as base_image:
            result = base_image.convert("RGBA")
        for layer_path in ctx.inputs[1:]:
            with Image.open(layer_path) as layer:
                # 各层分别生成放大，尺寸按理一致；万一不同就拉到底图尺寸，不让预览失败
                result = Image.alpha_composite(
                    result, layer.convert("RGBA").resize(result.size, Image.Resampling.LANCZOS))
        path = ctx.out_dir / f"{ctx.step_id}.png"
        result.save(path)
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


def match_color(image: Image.Image, reference: Image.Image, strength: float) -> Image.Image:
    import numpy as np
    from PIL import ImageCms

    srgb, lab = ImageCms.createProfile("sRGB"), ImageCms.createProfile("LAB")
    to_lab = ImageCms.buildTransform(srgb, lab, "RGB", "LAB")
    to_rgb = ImageCms.buildTransform(lab, srgb, "LAB", "RGB")

    def lab_pixels(source: Image.Image):
        pixels = np.asarray(ImageCms.applyTransform(source.convert("RGB"), to_lab), dtype=np.float32)
        # 只统计不透明像素：去背景后大片透明区域的底色不属于画面，会把均值拉偏
        opaque = np.asarray(source.getchannel("A")) > 127
        if not opaque.any():
            raise ValueError("图片完全透明，无法统计色调")
        return pixels, pixels[opaque]

    pixels, target_opaque = lab_pixels(image)
    _, reference_opaque = lab_pixels(reference)
    for channel in (1, 2):
        target_mean, target_std = target_opaque[:, channel].mean(), target_opaque[:, channel].std()
        reference_mean, reference_std = (reference_opaque[:, channel].mean(),
                                         reference_opaque[:, channel].std())
        scale = reference_std / target_std if target_std > 1e-3 else 1.0
        matched = (pixels[..., channel] - target_mean) * scale + reference_mean
        pixels[..., channel] += strength * (matched - pixels[..., channel])
    lab_image = Image.fromarray(np.round(np.clip(pixels, 0, 255)).astype(np.uint8), "LAB")
    result = ImageCms.applyTransform(lab_image, to_rgb).convert("RGBA")
    result.putalpha(image.getchannel("A"))
    return result


def apply_haze(image: Image.Image, reference: Image.Image, haze: float) -> Image.Image:
    import numpy as np

    reference_pixels = np.asarray(reference, dtype=np.float32)
    opaque = reference_pixels[..., 3] > 127
    if not opaque.any():
        raise ValueError("远景参考图完全透明，无法取雾色")
    haze_color = reference_pixels[..., :3][opaque].mean(axis=0)
    pixels = np.asarray(image, dtype=np.float32)
    # 只动 RGB：alpha 不变，轮廓和透明区保持原样
    rgb = pixels[..., :3] * (1 - haze) + haze_color * haze
    result = np.concatenate([rgb, pixels[..., 3:]], axis=-1)
    return Image.fromarray(np.round(np.clip(result, 0, 255)).astype(np.uint8), "RGBA")


def tile_horizontal(image: Image.Image, overlap: int) -> Image.Image:
    import numpy as np

    if overlap <= 0 or image.width <= overlap * 2:
        raise ValueError(f"图片宽 {image.width}px，无法做 {overlap}px 的无缝过渡")
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    # 预乘 alpha 再混合：直接混合 RGBA 时，透明像素里的底色会在过渡带渗出一圈杂边
    alpha = pixels[..., 3:]
    premultiplied = np.concatenate([pixels[..., :3] * alpha, alpha], axis=-1)
    weight = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[None, :, None]

    def blend(layer):
        seam = layer[:, -overlap:] * (1 - weight) + layer[:, :overlap] * weight
        return np.concatenate([seam, layer[:, overlap:-overlap]], axis=1)

    blended = blend(premultiplied)
    blended_alpha = blended[..., 3:]
    # 全透明处保留原 RGB（去背景时已填成外推的前景色）而不是写 0：
    # 引擎双线性采样会混入透明邻居的 RGB，黑色会在边缘拉出一圈暗边
    rgb = np.divide(blended[..., :3], blended_alpha, out=blend(pixels[..., :3]),
                    where=blended_alpha > 0)
    result = np.concatenate([rgb, blended_alpha], axis=-1)
    return Image.fromarray(np.round(np.clip(result, 0, 1) * 255).astype(np.uint8), "RGBA")


def describe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"size": [image.width, image.height]}


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    ImageGenerateRunner(),
    RemoveBackgroundRunner(),
    UpscaleRunner(),
    TrimRunner(),
    ResizeRunner(),
    ColorMatchRunner(),
    AtmosphereRunner(),
    TileHorizontalRunner(),
    CompositeRunner(),
)
