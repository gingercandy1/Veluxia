"""序列帧类步骤的 runner（ADR 0006）：图生视频拆帧、循环检测、逐帧去背景、拼精灵图。

这些步骤的产物是多个文件（帧序列），统一放在条目目录下以步骤 id 命名的子目录里，
文件名用 0000.png 递增，manifest 里的顺序就是播放顺序。
"""
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.core.collection.generator_runner import (
    GeneratorStepRunner,
    call_blocking,
    move_into,
)
from src.backend.core.collection.image_steps import describe_image, single_input
from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.collection.template import PROMPT_FIELD, SOURCE_DESCRIPTION_FIELD
from src.backend.core.image_frame.sprite_export import export_sprite_sheet
from src.shared.enum_type import FactoryType

# 接缝差异超过"相邻两帧平均差异"的这个倍数时，界面提示需要人工检查
_SEAM_CHECK_RATIO = 1.5


class AnimationGenerateRunner(GeneratorStepRunner):
    """立绘铺白底后图生视频，拆成帧序列。

    提示词是「来源角色描述 + 条目原文（动作描述）+ 后缀」，不拼风格锁：画风已经由参考图决定，
    风格词反而会让视频模型改画面而不是做动作。只写「跳跃」时视频模型不知道主体是个球，
    会按人形去做动作长出手脚，所以先点明主体；AI 起草的动作描述里已经写了主体时不再重复。
    """
    type_name = "animation.generate"
    factory_type = FactoryType.Animation

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items()
                  if k not in ("model_name", "prompt_suffix", "scale")}
        action = ctx.values.get(PROMPT_FIELD, "").strip()
        subject = ctx.values.get(SOURCE_DESCRIPTION_FIELD, "").strip()
        if subject and subject.lower() in action.lower():
            subject = ""
        params["content"] = ", ".join(
            p.strip() for p in (subject, action, ctx.params.get("prompt_suffix", "")) if p.strip())
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        scale = reference_scale(self.type_name, ctx, 0.8)
        reference = ctx.out_dir / f"{ctx.step_id}_reference.png"
        # 角色是竖长的，竖版画布让角色在画面里更大，脸和眼睛才有足够的像素不崩
        size = (params["width"], params["height"]) if "width" in params else None
        with Image.open(single_input(self.type_name, ctx)) as portrait:
            white_canvas(portrait.convert("RGBA"), scale, size).save(reference)
        try:
            params["reference_image_path"] = str(reference)
            frame_paths = self._generate_frames(params)
        finally:
            reference.unlink(missing_ok=True)

        target = reset_frame_dir(ctx)
        outputs = [move_into(Path(path), target / f"{index:04d}.png")
                   for index, path in enumerate(frame_paths)]
        ctx.meta.update(self.describe(params))
        ctx.meta.update({"frames": len(outputs), **describe_image(outputs[0])})
        return outputs

    def _generate_frames(self, params: dict[str, Any]) -> list[Path]:
        generator = self.generator
        generator.check_cancelled()
        generator.parse_params(params)
        frame_paths, preview = call_blocking(generator.generate_animation())
        # 预览 GIF 由精灵图步骤按裁好的帧重新生成，这里的没有用
        if preview:
            Path(preview).unlink(missing_ok=True)
        frame_paths = [Path(p) for p in frame_paths or []]
        if not frame_paths or not all(p.is_file() for p in frame_paths):
            raise RuntimeError(f"{self.type_name} 没有生成帧，请查看后端日志")
        return frame_paths


class FramesLoopRunner(StepRunner):
    """从视频帧里找出最像首尾相接的一段（纯 CPU）。

    图生视频开头几帧是"从立绘姿势起步"，结尾也不会回到开头，直接循环播放会跳。
    跳过开头 skip 帧后枚举 (i, j)，取帧 i 与帧 j 最像的一对，输出 i..j-1：播放到 j-1
    之后回到 i，就等于接上了原本的 j。
    接缝不够好时照样输出，只在 meta 里标 needs_check，是否接受交给用户。
    """
    type_name = "frames.loop"

    def run(self, ctx: StepContext) -> list[Path]:
        mode = ctx.params.get("mode", "loop")
        if mode not in ("loop", "pingpong"):
            raise ValueError(f"{self.type_name} 的 mode 只能是 loop 或 pingpong：{mode}")
        skip = int(ctx.params.get("skip", 8))
        min_len = int(ctx.params.get("min_len", 8))
        stride = int(ctx.params.get("stride", 1))
        if skip < 0 or min_len < 2 or stride < 1:
            raise ValueError(f"{self.type_name} 参数不合法：skip={skip} min_len={min_len} stride={stride}")

        frames = list(ctx.inputs)
        if mode == "pingpong":
            # 待机这类幅度很小的动作找不到明显的周期，正放再倒放一定首尾相接。
            # 视频越往后离参考图越远（特效消散、长出肢体），往返只需要一小段，取紧挨开头的部分
            start = int(ctx.params.get("pingpong_skip", skip))
            length = int(ctx.params.get("pingpong_len", 0))
            if start < 0 or length < 0:
                raise ValueError(f"{self.type_name} 参数不合法：pingpong_skip={start} pingpong_len={length}")
            end = start + length if length else len(frames)
            forward = frames[start:end][::stride]
            if len(forward) < 3:
                raise ValueError(f"只有 {len(frames)} 帧，跳过 {start} 帧后不够做往返循环")
            selected = forward + forward[-2:0:-1]
            ctx.meta.update({"mode": mode, "start": start, "end": min(end, len(frames)),
                             "frames": len(selected)})
        else:
            loop = find_loop(load_signatures(frames), skip, min_len, stride)
            selected = frames[loop.start:loop.end:stride]
            ctx.meta.update({
                "mode": mode, "start": loop.start, "end": loop.end, "frames": len(selected),
                "seam_error": round(loop.error, 4), "seam_ratio": round(loop.ratio, 2),
                "needs_check": loop.ratio > _SEAM_CHECK_RATIO,
            })

        target = reset_frame_dir(ctx)
        outputs = []
        for index, source in enumerate(selected):
            path = target / f"{index:04d}.png"
            shutil.copyfile(source, path)
            outputs.append(path)
        return outputs


class FramesRemoveBgRunner(GeneratorStepRunner):
    """逐帧去背景：放在循环检测之后，只处理最终用得上的帧。"""
    type_name = "frames.remove_bg"

    def run(self, ctx: StepContext) -> list[Path]:
        if not ctx.inputs:
            raise ValueError(f"{self.type_name} 没有输入帧")
        extra = {k: v for k, v in ctx.params.items() if k != "model_name"}
        target = reset_frame_dir(ctx)
        outputs = []
        for index, source in enumerate(ctx.inputs):
            path = self.generate_once({**extra, "input_path": str(source)})
            outputs.append(move_into(path, target / f"{index:04d}.png"))
        ctx.meta.update({"model": self._model_name, "frames": len(outputs)})
        return outputs


class SpriteSheetRunner(StepRunner):
    """按全部帧的并集包围盒统一裁边，拼精灵图 + atlas JSON + GIF 预览（纯 CPU）。

    逐帧单独裁边会让角色在帧间抖动，所以复用 sprite_export 的并集裁边。
    精灵图放在产物第一位：界面按产物里的图片做预览，GIF 不算图片，不会被误当成封面。
    """
    type_name = "frames.sheet"

    def run(self, ctx: StepContext) -> list[Path]:
        if not ctx.inputs:
            raise ValueError(f"{self.type_name} 没有输入帧")
        fps = int(ctx.params.get("fps", 12))
        if fps <= 0:
            raise ValueError(f"{self.type_name} 的 fps 必须大于 0：{fps}")
        # atlas 里的帧名以条目 id 开头（如 walk_0000），导入引擎后按名字就能认出是哪个动作
        result = export_sprite_sheet(
            [str(p) for p in ctx.inputs], ctx.out_dir, ctx.item.id,
            columns=int(ctx.params.get("columns", 0)),
            padding=int(ctx.params.get("padding", 0)), trim=True, fps=fps)
        gif_path = ctx.out_dir / f"{ctx.item.id}_preview.gif"
        with Image.open(result.sheet_path) as sheet:
            cells = sheet_cells(sheet.convert("RGBA"), len(ctx.inputs), result.columns,
                                result.frame_width, result.frame_height,
                                int(ctx.params.get("padding", 0)))
        save_gif(cells, gif_path, fps)
        ctx.meta.update({
            "frames": len(ctx.inputs), "fps": fps,
            "frame_size": [result.frame_width, result.frame_height],
            "grid": [result.columns, result.rows],
        })
        return [result.sheet_path, result.atlas_path, gif_path]


@dataclass(frozen=True)
class LoopMatch:
    start: int
    end: int
    error: float
    ratio: float


def load_signatures(paths: list[Path], size: int = 64):
    """帧缩到 size×size 灰度后比较：够分辨姿势，又能忽略生成时的细小噪点。"""
    import numpy as np

    signatures = []
    for path in paths:
        with Image.open(path) as image:
            small = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
            signatures.append(np.asarray(small, dtype=np.float32) / 255.0)
    return np.stack(signatures)


def find_loop(signatures, skip: int, min_len: int, stride: int = 1) -> LoopMatch:
    """在 skip 之后找首尾最像的一段 [start, end)，长度至少 min_len、且是 stride 的整数倍。

    ratio 是接缝差异除以相邻帧的平均差异：画面里角色只占一小块时绝对差异都很小，
    按相邻帧归一化后，阈值才不随角色大小、动作幅度变化。ratio ≤ 1 说明接缝和正常的一帧一样平滑。
    """
    import numpy as np

    count = len(signatures)
    if count - skip <= min_len:
        raise ValueError(f"只有 {count} 帧，跳过 {skip} 帧后不够找出至少 {min_len} 帧的循环")
    # 对齐到 stride 的整数倍：抽帧后首尾间隔才和其他帧一致
    min_len = -(-min_len // stride) * stride
    best: tuple[float, int, int] | None = None
    for start in range(skip, count - min_len):
        for end in range(start + min_len, count, stride):
            error = float(np.abs(signatures[start] - signatures[end]).mean())
            # 差异相同时取更长的一段：包含的完整周期越多，动作越自然
            if best is None or error < best[0] - 1e-6 or (
                    abs(error - best[0]) <= 1e-6 and end - start > best[2] - best[1]):
                best = (error, start, end)
    if best is None:
        raise ValueError(f"只有 {count} 帧，找不到长度为 {stride} 整数倍的循环段")
    error, start, end = best
    steps = np.abs(np.diff(signatures[start:end + 1], axis=0)).mean(axis=(1, 2))
    typical = float(steps.mean()) if steps.size else 0.0
    ratio = error / typical if typical > 1e-6 else 0.0
    return LoopMatch(start, end, error, ratio)


def reset_frame_dir(ctx: StepContext) -> Path:
    """重跑时先清掉旧帧：帧数可能变少，残留的旧帧会混进下游。"""
    target = ctx.out_dir / ctx.step_id
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def reference_scale(type_name: str, ctx: StepContext, default: float) -> float:
    """white_canvas 的 scale 参数：角色占画面的比例。"""
    scale = float(ctx.params.get("scale", default))
    if not 0 < scale <= 1:
        raise ValueError(f"{type_name} 的 scale 必须在 0 到 1 之间：{scale}")
    return scale


def white_canvas(image: Image.Image, scale: float,
                 size: tuple[int, int] | None = None) -> Image.Image:
    """把透明底立绘等比放进白底画布，size 为空时是正方形。

    视频模型对比例不符的输入会补黑边，大片黑边会压住动作；白底和角色包生成立绘时的背景一致，
    之后逐帧去背景也更干净。scale 是角色占画面的比例，留出走路摆臂的空间。
    给了 size 就直接缩放到这个尺寸，模型拿到的就是它要的分辨率，不会再裁切或补边。
    """
    aspect = size[0] / size[1] if size else 1.0
    height = max(round(max(image.width / aspect, image.height) / scale), 1)
    width = max(round(height * aspect), 1)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
    if size:
        canvas = canvas.resize(size, Image.Resampling.LANCZOS)
    return canvas.convert("RGB")


def sheet_cells(sheet: Image.Image, count: int, columns: int, width: int, height: int,
                padding: int) -> list[Image.Image]:
    cells = []
    for index in range(count):
        x = padding + (index % columns) * (width + padding)
        y = padding + (index // columns) * (height + padding)
        cells.append(sheet.crop((x, y, x + width, y + height)))
    return cells


def save_gif(frames: list[Image.Image], path: Path, fps: int) -> None:
    # GIF 只有 1 位透明，半透明边缘会出锯齿；预览铺白底，和生成时的背景一致
    flattened = []
    for frame in frames:
        canvas = Image.new("RGBA", frame.size, (255, 255, 255, 255))
        canvas.alpha_composite(frame)
        flattened.append(canvas.convert("RGB"))
    flattened[0].save(path, save_all=True, append_images=flattened[1:],
                      duration=max(round(1000 / fps), 20), loop=0)


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    AnimationGenerateRunner(),
    FramesLoopRunner(),
    FramesRemoveBgRunner(),
    SpriteSheetRunner(),
)
