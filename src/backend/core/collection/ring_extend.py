"""循环画布的高度补齐与外扩：纯图像计算，不加载模型。

环上各屏高度不一致时（部分屏做了外扩），长图和接缝窗口都要求高度相同，
所以较矮的屏用边缘行拉伸补齐；外扩则是在上下各补一条空白交给模型续画，
续画完再把中间的原图像素换回去，保证模型只改了新补的部分。
"""
import numpy as np
from PIL import Image

# 外扩每边补的像素；plan.md 约定的默认值
EXTEND_PAD = 128
# Qwen-Image-2.1 对宽高的要求：32 的倍数且在 256～2048
MODEL_SIDE_STEP = 32
MODEL_MIN_SIDE = 256
MODEL_MAX_SIDE = 2048


def _pad_with_edges(image: Image.Image, top: int, bottom: int) -> Image.Image:
    """上下各补 top / bottom 行，补的内容是原图最上 / 最下一行拉伸出来的，颜色衔接自然。"""
    rgb = image.convert("RGB")
    if top == 0 and bottom == 0:
        return rgb
    canvas = Image.new("RGB", (rgb.width, rgb.height + top + bottom))
    canvas.paste(rgb, (0, top))
    if top:
        canvas.paste(rgb.crop((0, 0, rgb.width, 1)).resize((rgb.width, top)), (0, 0))
    if bottom:
        canvas.paste(rgb.crop((0, rgb.height - 1, rgb.width, rgb.height))
                     .resize((rgb.width, bottom)), (0, top + rgb.height))
    return canvas


def fit_height(image: Image.Image, height: int) -> Image.Image:
    """宽不变，居中放置，上下用边缘行拉伸补到 height；已经够高就原样返回（RGB）。"""
    if image.height >= height:
        return image.convert("RGB")
    missing = height - image.height
    top = missing // 2
    return _pad_with_edges(image, top, missing - top)


def extend_canvas(image: Image.Image, pad: int) -> Image.Image:
    """上下各补 pad 像素（边缘行拉伸），作为模型续画的起点。"""
    if pad < 0:
        raise ValueError(f"外扩像素不能为负：{pad}")
    return _pad_with_edges(image, pad, pad)


def restore_middle(result: Image.Image, original: Image.Image, pad: int,
                   feather: int = 8) -> Image.Image:
    """把 result 中间的原图区域换回原图像素，只留下模型新补的上下两条。

    交界处 feather 像素做线性过渡：直接硬贴会在原图边缘留下一条可见的线。
    """
    rgb = result.convert("RGB")
    base = original.convert("RGB")
    if rgb.size != (base.width, base.height + 2 * pad):
        raise ValueError(
            f"续画结果尺寸 {rgb.size} 与外扩画布 {(base.width, base.height + 2 * pad)} 不符")
    rows = np.arange(base.height, dtype=np.float32)
    # 原图第一行 / 最后一行只占一点点权重，往里 feather 行后才完全是原图
    ramp = np.minimum(np.minimum((rows + 1) / (feather + 1),
                                 (base.height - rows) / (feather + 1)), 1.0)
    if pad == 0:
        ramp = np.ones_like(ramp)
    alpha = ramp[:, None, None]
    middle = np.asarray(rgb.crop((0, pad, rgb.width, pad + base.height)), dtype=np.float32)
    blended = np.asarray(base, dtype=np.float32) * alpha + middle * (1.0 - alpha)
    out = rgb.copy()
    out.paste(Image.fromarray(np.round(blended).astype(np.uint8), "RGB"), (0, pad))
    return out


def model_dimensions(width: int, height: int, target_area: int) -> tuple[int, int]:
    """给模型的合法宽高：保持比例、面积接近 target_area，取 32 的倍数并夹在 256～2048。

    Qwen-Image-2.1 要求宽高是 32 的倍数且在 256～2048；面积取编辑分辨率的平方，
    和管线自己默认的推导一致（参考图也是缩到这个面积），否则大图会让 8GB 显存吃紧。
    输出会再缩放回需要的尺寸，所以比例上的几个像素误差无所谓。
    """
    scale = (target_area / (width * height)) ** 0.5

    def snap(side: float) -> int:
        stepped = round(side * scale / MODEL_SIDE_STEP) * MODEL_SIDE_STEP
        return max(MODEL_MIN_SIDE, min(MODEL_MAX_SIDE, stepped))

    return snap(width), snap(height)
