"""无缝平铺（tileable）：把 UNet / VAE 的卷积改成循环填充，让图像边缘首尾相接。

用实例级 `_conv_forward` 覆盖而不是改 `padding_mode`，因为 `padding_mode` 无法只对单个轴生效，
而横版卷轴背景只需要左右无缝。关闭时删掉实例属性即可回落到类方法，不留残余状态
（生成器是单例，模型会被反复复用）。
"""
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


def apply_tile_mode(modules, mode: str) -> None:
    """对 modules（如 UNet、VAE）内所有 Conv2d 应用无缝模式；"off" 时还原。"""
    validate_tile_mode(mode)
    tile_h = mode in ("both", "vertical")
    tile_w = mode in ("both", "horizontal")
    for root in modules:
        for module in root.modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue
            if mode == "off":
                module.__dict__.pop("_conv_forward", None)
            else:
                module._conv_forward = _make_tiled_conv_forward(module, tile_h, tile_w)
