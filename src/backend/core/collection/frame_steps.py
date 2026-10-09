import random
import shutil
import tempfile
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
from src.backend.core.collection.template import PROMPT_FIELD
from src.backend.core.image_frame.sprite_export import export_sprite_sheet
from src.shared.enum_type import FactoryType

# 接缝差异超过"相邻两帧平均差异"的这个倍数时，界面提示需要人工检查
_SEAM_CHECK_RATIO = 1.5

# 画面比例 -> (宽, 高)。总像素都压在约 26 万，保证 8GB 显存下 Wan2.1 14B 的占用和竖屏一致；
# 边长保持 16 的倍数，视频模型要求
VIDEO_SIZES = {
    "portrait": (384, 672),
    "landscape": (672, 384),
    "square": (512, 512),
    # 体型小、占屏幕比例不大的角色（球、精灵）：像素只有竖屏的 40%，生成快得多
    "compact": (320, 320),
}
# 小画布上角色要占得更满，否则脸和眼睛的像素不够；覆盖模板里的 scale
SIZE_SCALES = {"compact": 0.8}

# 播放方式。vanish 是"播完主体消失"（消散、传送）：视频把末帧钉成空画布让模型自己淡出，
# 之后的截取、精灵图都按单次播放处理
PLAYBACK_MODES = ("loop", "pingpong", "once", "vanish")


def playback_mode(type_name: str, mode: str, key: str = "mode") -> str:
    """校验播放方式，并折成下游步骤认的三种之一。"""
    if mode not in PLAYBACK_MODES:
        raise ValueError(f"{type_name} 的 {key} 只能是 {'、'.join(PLAYBACK_MODES)}：{mode}")
    return "once" if mode == "vanish" else mode


class AnimationGenerateRunner(GeneratorStepRunner):
    """立绘铺白底后图生视频，拆成帧序列。

    提示词只用条目原文（动作描述）+ 后缀，不拼风格锁：画风已经由参考图决定，
    风格词反而会让视频模型改画面而不是做动作。
    """
    type_name = "animation.generate"
    factory_type = FactoryType.Animation

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items()
                  if k not in ("model_name", "prompt_suffix", "scale", "background")}
        orientation = params.pop("orientation", "")
        if orientation:
            if orientation not in VIDEO_SIZES:
                raise ValueError(f"{self.type_name} 的 orientation 无效：{orientation}")
            params["width"], params["height"] = VIDEO_SIZES[orientation]
        loop_mode = params.pop("loop_mode", "")
        if loop_mode:
            params["loop"] = playback_mode(self.type_name, loop_mode, "loop_mode") != "once"
            if loop_mode == "vanish":
                params["end_blank"] = True
        action = ctx.values.get(PROMPT_FIELD, "")
        parts = [action, ctx.params.get("prompt_suffix", "")]
        if loop_mode in ("loop", "pingpong"):
            parts.append("seamless repeating motion, end in the same pose as the beginning")
        params["content"] = ", ".join(p.strip() for p in parts if p.strip())
        # 视频生成器默认种子固定，不随机的话重新生成得到一模一样的结果；
        # 动作幅度很依赖种子，用户只能靠重跑换一个。种子会记进 meta，仍可复现
        params.setdefault("seed", random.randrange(2**31))
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        scale = SIZE_SCALES.get(ctx.params.get("orientation", "")) or reference_scale(
            self.type_name, ctx, 0.8)
        reference = ctx.out_dir / f"{ctx.step_id}_reference.png"
        # 角色是竖长的，竖版画布让角色在画面里更大，脸和眼睛才有足够的像素不崩
        size = (params["width"], params["height"]) if "width" in params else None
        with Image.open(single_input(self.type_name, ctx)) as portrait:
            background = canvas_background(portrait, ctx.params.get("background", "white"))
            white_canvas(portrait.convert("RGBA"), scale, size, background).save(reference)
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
    mode 为 once 时不找循环：攻击、死亡这类只播一次的动作，原样输出整段视频。
    它们本来就该从立绘姿势起步，所以不跳过开头（要跳可以用 once_skip）。
    """
    type_name = "frames.loop"

    def run(self, ctx: StepContext) -> list[Path]:
        mode = playback_mode(self.type_name, ctx.params.get("mode", "loop"))
        skip = int(ctx.params.get("skip", 8))
        min_len = int(ctx.params.get("min_len", 8))
        stride = int(ctx.params.get("stride", 1))
        if skip < 0 or min_len < 2 or stride < 1:
            raise ValueError(f"{self.type_name} 参数不合法：skip={skip} min_len={min_len} stride={stride}")

        frames = list(ctx.inputs)
        if mode == "once":
            once_skip = int(ctx.params.get("once_skip", 0))
            if once_skip < 0 or len(frames) - once_skip < 2:
                raise ValueError(f"只有 {len(frames)} 帧，跳过 {once_skip} 帧后不够做单次播放")
            selected = frames[once_skip::stride]
            ctx.meta.update({"mode": mode, "start": once_skip, "frames": len(selected),
                             "needs_check": False})
        elif mode == "pingpong":
            # 待机这类幅度很小的动作找不到明显的周期。视频生成时末帧已经钉回起始姿势，
            # 整段正放、结尾叠化进开头即可首尾相接；不倒放，烟雾、飘动倒着播运动方向会反，很怪
            overlap = int(ctx.params.get("crossfade", 6))
            forward = frames[skip:][::stride]
            outputs = blend_loop_seam(self.type_name, forward, overlap, reset_frame_dir(ctx))
            ctx.meta.update({"mode": mode, "start": skip, "frames": len(outputs),
                             "crossfade": overlap})
            return outputs
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


def blend_loop_seam(type_name: str, paths: list[Path], overlap: int, target: Path) -> list[Path]:
    """把结尾 overlap 帧淡化叠进开头，输出 len(paths) - overlap 帧到 target，首尾无缝。"""
    import numpy as np

    count = len(paths)
    if overlap < 0 or count <= overlap * 2:
        raise ValueError(f"{type_name} 的叠化帧数 {overlap} 不合适：只有 {count} 帧，"
                         "不能为负且必须少于一半")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGBA"), dtype=np.float32))
    outputs = []
    for index in range(count - overlap):
        frame = frames[index]
        if index < overlap:
            weight = (index + 1) / (overlap + 1)
            frame = frames[count - overlap + index] * (1 - weight) + frame * weight
        path = target / f"{index:04d}.png"
        Image.fromarray(frame.round().astype(np.uint8)).save(path)
        outputs.append(path)
    return outputs


class FramesCrossfadeRunner(StepRunner):
    """把结尾 frames 帧淡化叠进开头，做成无缝循环（纯 CPU），思路和音乐的 audio.loop 一样。

    末帧锚定后结尾已经接近第一帧，但闪光、粒子这类细节不会正好回到原样，
    直接跳回第一帧会一闪（实测接缝差异是相邻帧的 4～5 倍）；叠 6 帧后和普通相邻帧一样平滑。
    输出比输入少 frames 帧：被叠进开头的结尾不再单独播放。
    frames 为 0 表示单次播放（释放、爆发这类只播一次的特效）：原样输出，不叠化，
    这样模板用一个字段就能切换播放方式，不必为两种方式各写一套步骤。
    """
    type_name = "frames.crossfade"

    def run(self, ctx: StepContext) -> list[Path]:
        overlap = int(ctx.params.get("frames", 6))
        outputs = blend_loop_seam(self.type_name, list(ctx.inputs), overlap, reset_frame_dir(ctx))
        ctx.meta.update({"frames": len(outputs), "crossfade": overlap})
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


class FramesSwayRunner(StepRunner):
    """程序化网格摇摆：透明底静态图 → 无缝循环序列帧（纯 CPU，ADR 0008）。

    不加载模型、不占显存租约。算法和参数校验在 image_frame/mesh_warp.py，这里只做文件进出和取消检查。
    """
    type_name = "frames.sway"

    def run(self, ctx: StepContext) -> list[Path]:
        frames = int(ctx.params.get("frames", 24))
        amplitude = float(ctx.params.get("amplitude", 0.08))
        power = float(ctx.params.get("power", 1.5))
        lag = float(ctx.params.get("lag", 0.6))
        density = int(ctx.params.get("mesh_density", 12))
        wave = float(ctx.params.get("wave", 1.5))
        flutter = float(ctx.params.get("flutter", 0.3))

        from src.backend.core.image_frame.mesh_warp import SwayParams, iter_sway_frames

        # 参数范围由 SwayParams 校验
        params = SwayParams(frames, amplitude, power, lag, density, wave, flutter)
        with Image.open(single_input(self.type_name, ctx)) as image:
            rendered = iter_sway_frames(image, params)

        target = reset_frame_dir(ctx)
        outputs = []
        # 逐帧渲染，两帧之间检查取消
        for index, frame in enumerate(rendered):
            if ctx.cancel_event.is_set():
                raise RuntimeError(f"{self.type_name} 已取消")
            path = target / f"{index:04d}.png"
            frame.save(path)
            outputs.append(path)
        ctx.meta.update({
            "frames": len(outputs), "amplitude": amplitude, "power": power,
            "lag": lag, "mesh_density": density, "wave": wave, "flutter": flutter,
        })
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
        stride = int(ctx.params.get("stride", 1))
        size = int(ctx.params.get("size", 0))
        if stride < 1 or size < 0:
            raise ValueError(f"{self.type_name} 参数不合法：stride={stride} size={size}")
        frames = list(ctx.inputs)[::stride]
        # atlas 里的帧名以条目 id 开头（如 walk_0000），导入引擎后按名字就能认出是哪个动作
        with tempfile.TemporaryDirectory(dir=ctx.out_dir) as scratch:
            if size:
                frames = shrink_frames(frames, size, Path(scratch))
            result = export_sprite_sheet(
                [str(p) for p in frames], ctx.out_dir, ctx.item.id,
                columns=int(ctx.params.get("columns", 0)),
                padding=int(ctx.params.get("padding", 0)), trim=True, fps=fps,
                loop=_sheet_loop(self.type_name, ctx.params.get("mode")))
        gif_path = ctx.out_dir / f"{ctx.item.id}_preview.gif"
        with Image.open(result.sheet_path) as sheet:
            cells = sheet_cells(sheet.convert("RGBA"), len(frames), result.columns,
                                result.frame_width, result.frame_height,
                                int(ctx.params.get("padding", 0)))
        save_gif(cells, gif_path, fps)
        ctx.meta.update({
            "frames": len(frames), "fps": fps,
            "frame_size": [result.frame_width, result.frame_height],
            "grid": [result.columns, result.rows],
        })
        return [result.sheet_path, result.atlas_path, gif_path]


def _sheet_loop(type_name: str, mode: Any) -> bool | None:
    """精灵图 atlas 里写的"是否循环"：模板把循环方式（frames.loop 的 mode）传过来，
    没传就不写，沿用以前不带这个字段的 atlas。"""
    if mode is None or mode == "":
        return None
    return playback_mode(type_name, mode) != "once"


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


def canvas_background(image: Image.Image, mode: str) -> tuple[int, int, int]:
    """补边用的底色：white 是角色立绘的白底；auto 取左上角像素，
    特效图标是深色底，补白边会让视频模型把白边当成画面的一部分去动。"""
    if mode == "white":
        return (255, 255, 255)
    if mode == "auto":
        return image.convert("RGB").getpixel((0, 0))
    raise ValueError(f"background 只能是 white 或 auto：{mode}")


def white_canvas(image: Image.Image, scale: float, size: tuple[int, int] | None = None,
                 background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """把透明底立绘等比放进白底画布，size 为空时是正方形。

    视频模型对比例不符的输入会补黑边，大片黑边会压住动作；白底和角色包生成立绘时的背景一致，
    之后逐帧去背景也更干净。scale 是角色占画面的比例，留出走路摆臂的空间。
    给了 size 就直接缩放到这个尺寸，模型拿到的就是它要的分辨率，不会再裁切或补边。
    """
    aspect = size[0] / size[1] if size else 1.0
    height = max(round(max(image.width / aspect, image.height) / scale), 1)
    width = max(round(height * aspect), 1)
    canvas = Image.new("RGBA", (width, height), (*background, 255))
    canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
    if size:
        canvas = canvas.resize(size, Image.Resampling.LANCZOS)
    return canvas.convert("RGB")


def shrink_frames(frames: list[Path], size: int, target: Path) -> list[Path]:
    """帧的长边缩到 size 像素：视频模型的原始帧有 700 多像素，几十帧拼成精灵图会大到引擎难以导入。"""
    shrunk = []
    for index, source in enumerate(frames):
        with Image.open(source) as frame:
            scale = size / max(frame.size)
            small = frame.resize(
                (max(round(frame.width * scale), 1), max(round(frame.height * scale), 1)),
                Image.Resampling.LANCZOS)
        path = target / f"{index:04d}.png"
        small.save(path)
        shrunk.append(path)
    return shrunk


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
    FramesCrossfadeRunner(),
    FramesRemoveBgRunner(),
    FramesSwayRunner(),
    SpriteSheetRunner(),
)
