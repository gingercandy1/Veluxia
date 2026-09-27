"""超分 / 抠图升级的契约测试（无权重、无 GPU 也可跑）。"""
import json

import torch

from src.app.param import GenerationRequest
from src.backend.core.generator_registry import register_all
from PIL import Image

from src.backend.core.image.upscale import RealEsrganAnimeGenerator, tiled_upscale
from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT


def test_tiled_upscale_matches_whole_image():
    # 最近邻放大是逐像素的，分块结果必须与整图完全一致，任何裁剪偏移都会暴露出来
    model = torch.nn.Upsample(scale_factor=2, mode="nearest")
    tensor = torch.rand(1, 3, 37, 53)
    whole = model(tensor)
    tiled = tiled_upscale(model, tensor, scale=2, tile=16, pad=4)
    assert torch.equal(whole, tiled)


def test_tiled_upscale_calls_on_tile_for_every_tile():
    calls = []
    tensor = torch.rand(1, 3, 32, 32)
    tiled_upscale(torch.nn.Upsample(scale_factor=2), tensor, 2, tile=16, pad=0, on_tile=lambda: calls.append(1))
    assert len(calls) == 4


class _NearestUpscaler(torch.nn.Module):
    """模拟 spandrel 描述符：4 倍最近邻放大，带 scale 属性和一个参数（取 dtype 用）。"""
    scale = 4

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")


def test_outscale_keeps_rgb_of_transparent_pixels():
    # 输出倍率不等于模型倍率时要再缩放；全透明像素的 RGB 不能被清成黑色，否则引擎里边缘发暗
    generator = object.__new__(RealEsrganAnimeGenerator)
    generator.pipe, generator.device = _NearestUpscaler(), "cpu"
    generator.tile_mode, generator.outscale = "off", 2
    generator.check_cancelled = lambda: None
    image = Image.new("RGBA", (16, 16), (0, 200, 0, 0))
    image.paste((255, 0, 0, 255), (0, 0, 8, 16))
    result = generator._upscale(image)
    assert result.size == (32, 32)
    assert result.getpixel((30, 16)) == (0, 200, 0, 0)
    assert result.getpixel((2, 16)) == (255, 0, 0, 255)


def test_new_image_models_registered_and_resolvable():
    register_all()
    with open(f"{PROJECT_ROOT}/models.json", "r", encoding="utf-8") as f:
        image_models = json.load(f)["image"]
    names = GeneratorFactory.get_generator_names(FactoryType.Image)
    for name in ("rembg-birefnet", "RealESRGAN-anime-6B", "RealESRGAN-x4plus"):
        assert name in image_models and name in names


def test_first_image_attachment_becomes_reference_image(tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    note = tmp_path / "n.txt"
    note.write_text("x")
    request = GenerationRequest.build(
        FactoryType.Image, "RealESRGAN-anime-6B", "p", attachments=[str(note), str(image)])
    assert request.to_api_payload()["extra"]["reference_image"] == str(image)


def test_no_image_attachment_leaves_reference_image_unset():
    request = GenerationRequest.build(FactoryType.Image, "SDXL", "p")
    assert "reference_image" not in request.to_api_payload()["extra"]
