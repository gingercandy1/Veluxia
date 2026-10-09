"""循环拼接的边缘排序、接缝窗口和补丁处理（纯 CPU）。"""
from pathlib import Path

import numpy as np
from PIL import Image

EDGE_COLUMNS = 8
BAND = 256
FEATHER = 24
HALF_WINDOW = 512
HARMONIZE_MARGIN = 64
PATCH_HALF = (BAND + 2 * FEATHER) // 2


def edge_strips(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """只载入左右边缘，避免排序阶段把整张大图转成浮点数组。"""
    with Image.open(path) as image:
        if image.width < EDGE_COLUMNS:
            raise ValueError(f"图片宽度不足 {EDGE_COLUMNS} 列：{path}")
        right = np.asarray(
            image.crop((image.width - EDGE_COLUMNS, 0, image.width, image.height)).convert("RGB"),
            dtype=np.float32,
        ).mean(axis=1)
        left = np.asarray(
            image.crop((0, 0, EDGE_COLUMNS, image.height)).convert("RGB"),
            dtype=np.float32,
        ).mean(axis=1)
    return right, left


def edge_cost_matrix(strips: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """cost[i][j] 表示 i 右边缘接 j 左边缘的逐行平均绝对差。"""
    count = len(strips)
    if not strips:
        return np.zeros((0, 0), dtype=np.float64)
    heights = {right.shape[0] for right, _ in strips} | {
        left.shape[0] for _, left in strips
    }
    if len(heights) != 1:
        raise ValueError("循环拼接的图片高度必须一致")
    if any(right.shape != left.shape or right.ndim != 2 or right.shape[1] != 3
           for right, left in strips):
        raise ValueError("图片边缘数据必须是相同高度的 (H, 3) RGB 数组")

    cost = np.zeros((count, count), dtype=np.float64)
    for i, (right, _) in enumerate(strips):
        for j, (_, left) in enumerate(strips):
            if i != j:
                cost[i, j] = np.abs(right - left).mean()
    return cost


def _tour_cost(cost: np.ndarray, order: list[int]) -> float:
    return sum(cost[order[k], order[(k + 1) % len(order)]] for k in range(len(order)))


def order_loop(cost: np.ndarray) -> list[int]:
    """每个起点各做一次最近邻，再用 2-opt 反转片段改进，取总代价最低的闭环。"""
    count = len(cost)
    if count < 2:
        raise ValueError(f"循环拼接至少要两张图，实际 {count} 张")
    best, best_cost = None, float("inf")
    for start in range(count):
        order, left = [start], set(range(count)) - {start}
        while left:
            nxt = min(left, key=lambda j: cost[order[-1], j])
            order.append(nxt)
            left.remove(nxt)
        improved = True
        while improved:
            improved = False
            current = _tour_cost(cost, order)
            for a in range(1, count - 1):
                for b in range(a + 1, count):
                    trial = order[:a] + order[a:b + 1][::-1] + order[b + 1:]
                    trial_cost = _tour_cost(cost, trial)
                    if trial_cost < current - 1e-9:
                        order, current, improved = trial, trial_cost, True
        if current < best_cost:
            best, best_cost = order, current
    return best


def build_window(left: Image.Image, right: Image.Image) -> Image.Image:
    """把左图末尾和右图开头各 512 列并排，令接缝位于窗口正中。"""
    if left.height != right.height:
        raise ValueError("接缝两侧图片的高度必须一致")
    if left.width < HALF_WINDOW or right.width < HALF_WINDOW:
        raise ValueError(f"接缝图片宽度必须至少为 {HALF_WINDOW} 列")
    window = Image.new("RGB", (HALF_WINDOW * 2, left.height))
    left_crop = left.crop((left.width - HALF_WINDOW, 0, left.width, left.height)).convert("RGB")
    right_crop = right.crop((0, 0, HALF_WINDOW, right.height)).convert("RGB")
    window.paste(left_crop, (0, 0))
    window.paste(right_crop, (HALF_WINDOW, 0))
    return window


def band_alpha(height: int) -> np.ndarray:
    """返回 256 列不透明重画带及两侧各 24 列的线性羽化 alpha。"""
    fade = np.linspace(0.0, 1.0, FEATHER, dtype=np.float32)
    alpha = np.concatenate((fade, np.ones(BAND, dtype=np.float32), fade[::-1]))
    return np.broadcast_to(alpha[None, :, None], (height, alpha.size, 1)).copy()


def extract_patch(window: Image.Image) -> Image.Image:
    """从窗口中央取 304 列补丁，透明度只在羽化带外逐渐降到零。"""
    rgb = window.convert("RGB")
    if rgb.width != HALF_WINDOW * 2:
        raise ValueError(f"接缝窗口宽度必须为 {HALF_WINDOW * 2} 列")
    patch_width = BAND + 2 * FEATHER
    start = HALF_WINDOW - BAND // 2 - FEATHER
    patch = rgb.crop((start, 0, start + patch_width, rgb.height))
    alpha = Image.fromarray(np.round(band_alpha(rgb.height)[:, :, 0] * 255).astype(np.uint8))
    patch.putalpha(alpha)
    return patch


def narrow_merge(inpainted: Image.Image, harmonized: Image.Image) -> Image.Image:
    """只取融合图中间 256+128 列，并在两端用 64 列线性遮罩过渡。"""
    base, blend = inpainted.convert("RGB"), harmonized.convert("RGB")
    if base.size != blend.size or base.width != HALF_WINDOW * 2:
        raise ValueError("窄带融合的输入必须是尺寸相同的 1024 列窗口")
    width = BAND + 2 * HARMONIZE_MARGIN
    start = HALF_WINDOW - BAND // 2 - HARMONIZE_MARGIN
    fade = np.linspace(0.0, 1.0, HARMONIZE_MARGIN, dtype=np.float32)
    alpha = np.concatenate((fade, np.ones(BAND, dtype=np.float32), fade[::-1]))
    mask = Image.new("L", base.size)
    mask.paste(Image.fromarray(np.round(alpha[None, :] * 255).astype(np.uint8))
               .resize((width, base.height)), (start, 0))
    return Image.composite(blend, base, mask)


def apply_patch(base: Image.Image, patch: Image.Image, side: str) -> Image.Image:
    """按补丁 alpha 只覆盖条目图靠接缝的一半区域，其他像素保持不变。"""
    if side not in ("left", "right"):
        raise ValueError(f"未知补丁方向 {side!r}，可选 'left' 或 'right'")
    rgba = patch.convert("RGBA")
    if rgba.width != BAND + 2 * FEATHER or rgba.height != base.height:
        raise ValueError("接缝补丁尺寸与条目图高度不匹配")
    if base.width < PATCH_HALF:
        raise ValueError(f"条目图宽度不足 {PATCH_HALF} 列，无法贴入接缝补丁")

    result = base.convert("RGB").copy()
    if side == "right":
        overlay = rgba.crop((0, 0, PATCH_HALF, rgba.height))
        x = base.width - PATCH_HALF
    else:
        overlay = rgba.crop((PATCH_HALF, 0, rgba.width, rgba.height))
        x = 0

    region = result.crop((x, 0, x + PATCH_HALF, base.height)).convert("RGBA")
    result.paste(Image.alpha_composite(region, overlay).convert("RGB"), (x, 0))
    return result
