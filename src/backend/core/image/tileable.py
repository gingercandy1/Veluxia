"""无缝平铺（tileable）：把 UNet / VAE 的卷积改成循环填充，让图像边缘首尾相接。

用实例级 `_conv_forward` 覆盖而不是改 `padding_mode`，因为 `padding_mode` 无法只对单个轴生效，
而横版卷轴背景只需要左右无缝。关闭时删掉实例属性即可回落到类方法，不留残余状态
（生成器是单例，模型会被反复复用）。
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F

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


# DiT 两侧各补多少 latent 列（×8 为像素）：越宽接缝两边互相看到的内容越多，代价是 token 变多
CIRCULAR_CONTEXT = 8


@contextmanager
def circular_transformer(transformer: torch.nn.Module, mode: str, context: int = CIRCULAR_CONTEXT):
    """DiT（如 Z-Image）没有可改填充的卷积：每次前向前把 latent 沿平铺轴用对侧内容循环补宽，
    前向后裁回原尺寸。这样每一步（包括最后一步）模型都看得到首尾相接处两侧，结果首尾连续。

    用 forward hook 而不是替换 forward：cpu offload 已经包过一层 forward，替换会绕开它。
    输入是 Z-Image 的格式：latent 张量列表，每个形如 (C, F, H, W)。
    """
    tile_h, tile_w = _tile_axes(mode)
    dims = [dim for dim, enabled in ((-2, tile_h), (-1, tile_w)) if enabled]

    def pad(latent: torch.Tensor) -> torch.Tensor:
        for dim in dims:
            size = latent.shape[dim]
            # 取 2 的倍数，保持和 2×2 patch 对齐；画幅太窄时不能比自身还宽
            width = min(context, size // 2 // 2 * 2)
            latent = torch.cat([latent.narrow(dim, size - width, width), latent,
                                latent.narrow(dim, 0, width)], dim=dim)
        return latent

    def crop(output: torch.Tensor, original: torch.Size) -> torch.Tensor:
        for dim in dims:
            width = (output.shape[dim] - original[dim]) // 2
            output = output.narrow(dim, width, original[dim])
        return output

    shapes: list[torch.Size] = []

    def pre_hook(_module, args):
        latents, *rest = args
        shapes[:] = [latent.shape for latent in latents]
        return ([pad(latent) for latent in latents], *rest)

    def post_hook(_module, _args, output):
        outputs, *rest = output
        return ([crop(out, shape) for out, shape in zip(outputs, shapes, strict=True)], *rest)

    handles = [transformer.register_forward_pre_hook(pre_hook),
               transformer.register_forward_hook(post_hook)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


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
