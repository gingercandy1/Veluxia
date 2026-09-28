"""无缝平铺（tileable）：把 UNet / VAE 的卷积改成循环填充，让图像边缘首尾相接。

用实例级 `_conv_forward` 覆盖而不是改 `padding_mode`，因为 `padding_mode` 无法只对单个轴生效，
而横版卷轴背景只需要左右无缝。关闭时删掉实例属性即可回落到类方法，不留残余状态
（生成器是单例，模型会被反复复用）。
"""
from contextlib import contextmanager
from itertools import product

import torch
import torch.nn.functional as F

from src.backend.core.image.panorama import window_weight

TILE_MODES = ("off", "both", "horizontal", "vertical")


def validate_tile_mode(mode: str) -> str:
    if mode not in TILE_MODES:
        raise ValueError(f"未知的无缝模式: {mode}，可选: {', '.join(TILE_MODES)}")
    return mode


def _pad_axis(x: torch.Tensor, pad_h: int, pad_w: int, tile_h: bool, tile_w: bool) -> torch.Tensor:
    # 分轴填充：平铺的轴用循环填充，其余轴保持原本的零填充
    if pad_w:
        x = F.pad(x, (pad_w, pad_w, 0, 0), mode="circular" if tile_w else "constant")
    if pad_h:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="circular" if tile_h else "constant")
    return x


def _make_tiled_conv_forward(conv: torch.nn.Conv2d, tile_h: bool, tile_w: bool):
    pad_h, pad_w = conv.padding

    def tiled_conv_forward(input, weight, bias):
        padded = _pad_axis(input, pad_h, pad_w, tile_h, tile_w)
        return F.conv2d(padded, weight, bias, conv.stride, 0, conv.dilation, conv.groups)

    return tiled_conv_forward


def _tile_axes(mode: str) -> tuple[bool, bool]:
    validate_tile_mode(mode)
    return mode in ("both", "vertical"), mode in ("both", "horizontal")


# 分块解码 VAE 时 latent 两侧各补多少列（×8 为像素）用来首尾衔接，解码后裁掉
CIRCULAR_CONTEXT = 8
# 循环平移窗口两端的渐变带占比
CIRCULAR_OVERLAP = 0.25


@contextmanager
def circular_transformer(transformer: torch.nn.Module, mode: str):
    """DiT（如 Z-Image）没有可改填充的卷积：每次前向改为把 latent 沿平铺轴循环平移后前向、再移回，
    两次平移错开半屏，按两端渐变的权重合并。一次的首尾接缝恰好落在另一次的正中、权重为 1，
    所以每一步（包括最后一步）接缝处都有完整上下文，结果首尾连续。

    不用对侧内容补边：补边会让首尾几列 token 在序列里出现两次，全局注意力下完全相同的 token
    互相强烈关注，分布外，首尾会生成一条碎块乱码带。平移后每个 token 只出现一次。
    两轴都平铺时取两轴平移的全部组合（4 次前向），任一点都有一次在两轴上都远离接缝。
    逐次前向而不是拼成一批，8GB 放不下；和 panorama_transformer 一样临时包实例 forward
    （包的是 cpu offload 已经包过的那个），退出时原样还原，两者可叠用。
    输入是 Z-Image 的格式：latent 张量列表，每个形如 (C, F, H, W)，同一批尺寸相同。
    """
    tile_h, tile_w = _tile_axes(mode)
    dims = [dim for dim, enabled in ((-2, tile_h), (-1, tile_w)) if enabled]
    had_own_forward = "forward" in transformer.__dict__
    original = transformer.forward

    def forward(latents, *args, **kwargs):
        height, width = latents[0].shape[-2:]
        device = latents[0].device
        # 每轴两个平移量，接缝分别落在 1/4 和 3/4 处；取 2 的倍数，保持和 2×2 patch 对齐
        choices = [[(dim, size // 4 // 2 * 2), (dim, size * 3 // 4 // 2 * 2)]
                   for dim, size in ((-2, height), (-1, width)) if dim in dims]
        total = [latent.new_zeros(latent.shape, dtype=torch.float32) for latent in latents]
        weight_sum = torch.zeros(height, width, device=device)
        for shifts in product(*choices):
            weight = torch.ones(height, width, device=device)
            for dim, shift in shifts:
                ramp = torch.roll(window_weight(weight.shape[dim], CIRCULAR_OVERLAP), shift).to(device)
                weight = weight * (ramp[:, None] if dim == -2 else ramp)
            shift_dims = [dim for dim, _ in shifts]
            amounts = [shift for _, shift in shifts]
            outputs = original([torch.roll(latent, [-a for a in amounts], shift_dims)
                                for latent in latents], *args, **kwargs)[0]
            for acc, out in zip(total, outputs, strict=True):
                acc += torch.roll(out, amounts, shift_dims).float() * weight
            weight_sum += weight
        return ([(acc / weight_sum).to(latent.dtype) for acc, latent in zip(total, latents, strict=True)],)

    transformer.forward = forward
    try:
        yield
    finally:
        if had_own_forward:
            transformer.forward = original
        else:
            del transformer.forward


def apply_tile_mode(modules, mode: str) -> None:
    """对 modules（如 UNet、VAE）内所有 Conv2d 应用无缝模式；"off" 时还原。"""
    tile_h, tile_w = _tile_axes(mode)
    for root in modules:
        for module in root.modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            if mode == "off":
                module.__dict__.pop("_conv_forward", None)
            else:
                module._conv_forward = _make_tiled_conv_forward(module, tile_h, tile_w)
