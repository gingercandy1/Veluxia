"""把拼好的环形长图切成一段段，交给 Qwen-Layered 分层。只含纯计算，不加载模型。

长图坐标：按环顺序把各屏 ring.png 从左到右排起来，x 从 0 到总宽；环首尾相接，
切点 x 和 x + 总宽是同一个位置。
"""
from pathlib import Path

import numpy as np
from PIL import Image

# 段宽：Qwen-Layered 按面积缩放输入，宽高比超过约 2:1 就会被压扁；太窄则段数和耗时都翻倍
MIN_SEGMENT = 896
MAX_SEGMENT = 1536
# 手动拖切线时允许更窄的段（比如一小块只有前景的区域），但不能窄到没法分层
MIN_MANUAL_SEGMENT = 256
# 自动切点的候选间隔：8 像素足够精细，DP 的规模也小
CUT_STEP = 8
_COST_SMOOTH = 15


def long_image(images: list[Image.Image]) -> Image.Image:
    heights = {image.height for image in images}
    if len(heights) != 1:
        raise ValueError(f"环里各屏高度不一致，没法拼成长图：{sorted(heights)}")
    canvas = Image.new("RGB", (sum(image.width for image in images), heights.pop()))
    x = 0
    for image in images:
        canvas.paste(image.convert("RGB"), (x, 0))
        x += image.width
    return canvas


def column_cost(depth: np.ndarray) -> np.ndarray:
    """每列的"切了会伤到物体"的程度。

    depth 是归一到 0~1 的相对逆深度（越大越近）。树干、灌木比同一行的背景近，
    减去每行中位数后只剩前景物体；地面整行都近，减完接近 0，不会被当成物体。
    平滑时按环处理，首尾互相补齐。
    """
    nearer = np.clip(depth - np.median(depth, axis=1, keepdims=True), 0, None)
    cost = nearer.mean(axis=0)
    half = _COST_SMOOTH // 2
    padded = np.concatenate([cost[-half:], cost, cost[:half]])
    return np.convolve(padded, np.ones(_COST_SMOOTH) / _COST_SMOOTH, mode="valid")


def plan_cuts(cost: np.ndarray, min_segment: int = MIN_SEGMENT,
              max_segment: int = MAX_SEGMENT, step: int = CUT_STEP) -> list[int]:
    """在环上选切点：切点处代价之和最小，相邻切点间距都在 [min_segment, max_segment]。

    先把一个切点固定在全局代价最低处，把环展开成直线，再做动态规划。
    总宽不超过 max_segment 时整条环就是一段，只需一个切点。
    """
    width = len(cost)
    start = int(np.argmin(cost))
    if width <= max_segment:
        return [start]
    rolled = np.roll(cost, -start)
    positions = np.arange(0, width + 1, step)
    if positions[-1] != width:
        positions = np.append(positions, width)
    best = np.full(len(positions), np.inf)
    previous = np.full(len(positions), -1)
    best[0] = 0.0
    for j in range(1, len(positions)):
        span = positions[j] - positions[:j]
        allowed = (span >= min_segment) & (span <= max_segment) & np.isfinite(best[:j])
        if not allowed.any():
            continue
        candidates = np.where(allowed, best[:j], np.inf)
        i = int(np.argmin(candidates))
        best[j] = candidates[i] + rolled[positions[j] % width]
        previous[j] = i
    if not np.isfinite(best[-1]):
        raise ValueError(
            f"环总宽 {width} 像素，没法切成每段 {min_segment}～{max_segment} 像素的若干段")
    cuts, j = [], len(positions) - 1
    while j > 0:
        cuts.append(int(positions[j]))
        j = int(previous[j])
    return sorted((cut + start) % width for cut in cuts)


def segment_widths(cuts: list[int], width: int) -> list[int]:
    """第 i 段从 cuts[i] 到 cuts[i+1]，最后一段绕回第一个切点。"""
    if len(cuts) == 1:
        return [width]
    return [(cuts[(i + 1) % len(cuts)] - cuts[i]) % width for i in range(len(cuts))]


def validate_cuts(cuts: list[int], width: int) -> list[int]:
    """手动调整后的切点：排序去重、检查范围和段宽，返回规范化后的切点。"""
    if not cuts:
        raise ValueError("至少需要一个切点")
    normalized = sorted({int(cut) for cut in cuts})
    if normalized[0] < 0 or normalized[-1] >= width:
        raise ValueError(f"切点超出长图范围 0～{width - 1}")
    for index, segment in enumerate(segment_widths(normalized, width), start=1):
        if segment < MIN_MANUAL_SEGMENT or segment > MAX_SEGMENT:
            raise ValueError(f"第 {index} 段宽 {segment} 像素，每段要在 "
                             f"{MIN_MANUAL_SEGMENT}～{MAX_SEGMENT} 像素之间")
    return normalized


def export_segments(image: Image.Image, cuts: list[int], directory: Path) -> list[Path]:
    """按切点把长图切成段存成 01.png、02.png …；跨过长图末尾的那段从开头接上。"""
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.png"):
        old.unlink()
    width = image.width
    paths = []
    for index, (start, segment) in enumerate(zip(cuts, segment_widths(cuts, width)), start=1):
        piece = Image.new("RGB", (segment, image.height))
        head = min(segment, width - start)
        piece.paste(image.crop((start, 0, start + head, image.height)), (0, 0))
        if head < segment:
            piece.paste(image.crop((0, 0, segment - head, image.height)), (head, 0))
        path = directory / f"{index:02d}.png"
        piece.save(path, format="PNG")
        paths.append(path)
    return paths
