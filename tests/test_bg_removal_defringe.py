import numpy as np
from PIL import Image

from src.backend.core.image.bg_removal import defringe


def _disk_with_white_halo() -> Image.Image:
    """红色实心圆，外面套一圈半透明白（模拟白底图上遮罩偏宽留下的白边）。"""
    size = 64
    y, x = np.mgrid[:size, :size]
    dist = np.hypot(x - size / 2, y - size / 2)
    data = np.zeros((size, size, 4), dtype=np.uint8)
    halo = (dist >= 20) & (dist < 21)
    data[halo] = (255, 255, 255, 80)
    disk = dist < 20
    data[disk] = (200, 30, 30, 255)
    return Image.fromarray(data, "RGBA")


def test_defringe_removes_translucent_white_halo():
    out = np.asarray(defringe(_disk_with_white_halo()))
    y, x = np.mgrid[:64, :64]
    dist = np.hypot(x - 32, y - 32)
    assert out[(dist >= 20) & (dist < 22)][:, 3].max() == 0
    assert out[dist < 15][:, 3].min() == 255


def test_defringe_edge_colour_comes_from_inside_not_white():
    out = np.asarray(defringe(_disk_with_white_halo()))
    y, x = np.mgrid[:64, :64]
    dist = np.hypot(x - 32, y - 32)
    edge = out[(dist >= 18) & (dist < 20)]
    visible = edge[edge[:, 3] > 0]
    assert visible[:, 1].max() < 100


def test_defringe_keeps_fully_transparent_image_transparent():
    out = np.asarray(defringe(Image.new("RGBA", (8, 8), (0, 0, 0, 0))))
    assert out[..., 3].max() == 0
