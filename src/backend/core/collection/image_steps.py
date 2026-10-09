import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.backend.core.collection.generator_runner import GeneratorStepRunner, move_into
from src.backend.core.collection.steps import StepContext, StepRunner


class ImageGenerateRunner(GeneratorStepRunner):
    type_name = "image.generate"

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items()
                  if k not in ("model_name", "prompt", "prompt_suffix", "size")}
        # 画幅写成 "宽x高"，好让模板用一个字段切换（占位符替换出来的都是字符串）
        if "size" in ctx.params:
            params["width"], params["height"] = parse_size(ctx.params["size"])
        suffix = ctx.params.get("prompt_suffix", "")
        if isinstance(suffix, dict):
            # 后缀随画幅变：竖版的人形要 full body 才不会被截到大腿，
            # 方形的球形角色带 full body 反而会长出手脚（实测），一个后缀兼顾不了两头
            suffix = _suffix_for_size(suffix, ctx.params.get("size"))
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


class ImportRunner(StepRunner):
    """取用户建包时导入的图片（纯 CPU）：和"生成"互为二选一的起点，下游把它当成生成出来的图用。

    字段里存的是相对资源包的路径（library._import_images 复制进来的），这里统一转成 PNG：
    下游按 <步骤>.png 找图，也不用关心用户导入的是 jpg 还是 webp。
    """
    type_name = "image.import"

    def run(self, ctx: StepContext) -> list[Path]:
        field = ctx.params.get("field")
        if not field or not ctx.values.get(field):
            raise ValueError(f"{self.type_name} 需要用 field 指向一个已导入图片的字段：{field}")
        source = ctx.out_dir.parent / ctx.values[field]
        if not source.is_file():
            raise FileNotFoundError(f"导入的图片不见了：{source}")
        path = ctx.out_dir / f"{ctx.step_id}.png"
        with Image.open(source) as image:
            image.convert("RGBA").save(path)
        ctx.meta.update(describe_image(path))
        return [path]


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


class MontageRunner(StepRunner):
    """把多张透明图按输入顺序并排拼成一张（纯 CPU）：角色立绘 + 各视角的转面总览图。

    每张图按统一高度等比缩放、底边对齐：转面图要比的是同一个角色在不同角度下的样子，
    各视角裁边后高矮不一，不统一高度就没法并排看。
    """
    type_name = "image.montage"

    def run(self, ctx: StepContext) -> list[Path]:
        height = int(ctx.params.get("height", 768))
        gap = int(ctx.params.get("gap", 24))
        if height <= 0 or gap < 0:
            raise ValueError(f"{self.type_name} 参数不合法：height={height} gap={gap}")
        if len(ctx.inputs) < 2:
            raise ValueError(f"{self.type_name} 至少要两张图，实际 {len(ctx.inputs)} 张")
        views = []
        for source in ctx.inputs:
            with Image.open(source) as image:
                rgba = image.convert("RGBA")
                width = max(round(rgba.width * height / rgba.height), 1)
                views.append(rgba.resize((width, height), Image.Resampling.LANCZOS))
        canvas = Image.new("RGBA", (sum(v.width for v in views) + gap * (len(views) - 1), height),
                           (0, 0, 0, 0))
        x = 0
        for view in views:
            canvas.paste(view, (x, 0))
            x += view.width + gap
        path = ctx.out_dir / f"{ctx.step_id}.png"
        canvas.save(path)
        ctx.meta.update({"views": len(views), **describe_image(path)})
        return [path]


class SplitRunner(StepRunner):
    """把去好背景的组图按透明通道切成一个个元素（纯 CPU）：同一张图里画出的元素风格天然统一。

    切出几个就是几个，不按预设格子数：模型说画 6 个常常画成 5 个或 7 个。
    产物按先行后列的阅读顺序命名为 <步骤>_1.png …，每张只保留自己的像素，邻居被裁进框的部分清成透明。
    """
    type_name = "image.split"

    def run(self, ctx: StepContext) -> list[Path]:
        padding = int(ctx.params.get("padding", 4))
        min_area = float(ctx.params.get("min_area", 0.002))
        with Image.open(single_input(self.type_name, ctx)) as image:
            pieces = split_pieces(image.convert("RGBA"), min_area, padding)
        paths = []
        for index, piece in enumerate(pieces, start=1):
            path = ctx.out_dir / f"{ctx.step_id}_{index}.png"
            piece.save(path)
            paths.append(path)
        ctx.meta.update({**describe_image(paths[0]), "pieces": len(paths)})
        return paths


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


def split_pieces(image: Image.Image, min_area: float = 0.002,
                 padding: int = 4) -> list[Image.Image]:
    """按透明通道的连通区域切出各个元素，丢掉面积不到整图 min_area 的碎点。"""
    alpha = np.asarray(image.getchannel("A"))
    labels = _label_blocks(alpha > _SPLIT_ALPHA)
    count = int(labels.max())
    if count == 0:
        raise ValueError("图片完全透明，去背景可能把主体也去掉了")
    # 块标签放大回像素：每个像素只属于一个块，块又只属于一个元素，相邻元素不会互相混入
    pixel_labels = labels.repeat(_SPLIT_BLOCK, 0).repeat(_SPLIT_BLOCK, 1)
    pixel_labels = pixel_labels[:alpha.shape[0], :alpha.shape[1]]

    rgba = np.asarray(image)
    boxes = []
    for label in range(1, count + 1):
        mask = (pixel_labels == label) & (alpha > 0)
        if mask.sum() < min_area * alpha.size:
            continue
        rows, cols = np.nonzero(mask)
        top, bottom = max(rows.min() - padding, 0), min(rows.max() + 1 + padding, alpha.shape[0])
        left, right = max(cols.min() - padding, 0), min(cols.max() + 1 + padding, alpha.shape[1])
        boxes.append((top, bottom, left, right, mask))
    if not boxes:
        raise ValueError("组图里没有足够大的元素可切，可能元素画得太小或去背景失败")

    pieces = []
    for top, bottom, left, right, mask in _reading_order(boxes):
        crop = rgba[top:bottom, left:right].copy()
        crop[..., 3] = np.where(mask[top:bottom, left:right], crop[..., 3], 0)
        pieces.append(Image.fromarray(crop, "RGBA"))
    return pieces


# 不透明度超过它才算元素：去背景后背景上常残留几乎看不见的薄雾，会把相邻元素连成一片
_SPLIT_ALPHA = 32
# 按块判断连通：块内有任何不透明像素就算占用，元素内部几像素的缝隙（如蘑菇伞和杆之间）不会被切开，
# 也把逐像素的 Python 遍历缩小到 1/16
_SPLIT_BLOCK = 4


def _label_blocks(mask: np.ndarray) -> np.ndarray:
    """把像素掩码按块降采样后标记四连通区域，返回块网格上的标签（0 为背景）。"""
    height, width = mask.shape
    pad_h, pad_w = -height % _SPLIT_BLOCK, -width % _SPLIT_BLOCK
    padded = np.pad(mask, ((0, pad_h), (0, pad_w)))
    grid = padded.reshape(padded.shape[0] // _SPLIT_BLOCK, _SPLIT_BLOCK,
                          padded.shape[1] // _SPLIT_BLOCK, _SPLIT_BLOCK).any(axis=(1, 3))
    labels = np.zeros(grid.shape, dtype=np.int32)
    rows, cols = grid.shape
    current = 0
    for start in zip(*np.nonzero(grid)):
        if labels[start]:
            continue
        current += 1
        labels[start] = current
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] and not labels[nr, nc]:
                    labels[nr, nc] = current
                    stack.append((nr, nc))
    return labels


def _reading_order(boxes: list[tuple]) -> list[tuple]:
    """先行后列：按纵向中心排序，中心落在当前行已有元素的上下范围内就算同一行。

    元素高矮不一（高蘑菇和矮蘑菇并排），单按顶边排序会把同一行拆乱。
    """
    rows: list[list[tuple]] = []
    row_bottom = -1
    for box in sorted(boxes, key=lambda b: (b[0] + b[1]) / 2):
        center = (box[0] + box[1]) / 2
        if rows and center <= row_bottom:
            rows[-1].append(box)
            row_bottom = max(row_bottom, box[1])
        else:
            rows.append([box])
            row_bottom = box[1]
    return [box for row in rows for box in sorted(row, key=lambda b: b[2])]


def fit_square(image: Image.Image, size: int) -> Image.Image:
    scale = size / max(image.width, image.height)
    width, height = max(round(image.width * scale), 1), max(round(image.height * scale), 1)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS),
                 ((size - width) // 2, (size - height) // 2))
    return canvas


def _suffix_for_size(suffixes: dict[str, str], size: Any) -> str:
    """prompt_suffix 写成 {"768x1344": "…", "1024x1024": "…"} 时，按这一步的画幅选一条。"""
    key = "x".join(str(n) for n in parse_size(size)) if size is not None else None
    if key not in suffixes:
        raise ValueError(f"prompt_suffix 按画幅给了 {sorted(suffixes)}，"
                         f"但这一步的画幅是 {size!r}")
    return suffixes[key]


def parse_size(value: Any) -> tuple[int, int]:
    """"768x1344" → (768, 1344)。"""
    match = re.fullmatch(r"\s*(\d+)\s*[xX×]\s*(\d+)\s*", str(value))
    if match is None or int(match[1]) <= 0 or int(match[2]) <= 0:
        raise ValueError(f"画幅要写成 宽x高，如 768x1344：{value!r}")
    return int(match[1]), int(match[2])


def describe_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"size": [image.width, image.height]}


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    ImageGenerateRunner(),
    ImportRunner(),
    RemoveBackgroundRunner(),
    UpscaleRunner(),
    TrimRunner(),
    ResizeRunner(),
    MontageRunner(),
    SplitRunner(),
)
