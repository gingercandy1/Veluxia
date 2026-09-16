"""图片生成器注册契约测试（无模型权重、无 GPU 也可跑）。

契约：每个生成器类的 names 键必须与 models.json["image"] 的键一一对应，
否则 server.py 的 import 注册后，UI 下拉与 Factory 会对不上。
"""
import json

from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT

import src.backend.core.image.flux_schnell  # noqa: F401
import src.backend.core.image.sdxl  # noqa: F401
import src.backend.core.image.sd35_medium  # noqa: F401
import src.backend.core.image.z_image  # noqa: F401
import src.backend.core.image.qwen_image  # noqa: F401
import src.backend.core.image.bg_removal  # noqa: F401


def _image_config():
    with open(f"{PROJECT_ROOT}/models.json", "r", encoding="utf-8") as f:
        return json.load(f)["image"]


def test_all_image_generators_registered():
    names = GeneratorFactory.get_generator_names(FactoryType.Image)
    for expected in ("Flux.1-schnell", "SDXL", "SD3.5-Medium", "Z-Image-Turbo",
                       "Qwen-Image-Lightning", "rembg-u2net"):
        assert expected in names, f"{expected} 未注册到 Factory"


def test_names_match_models_json():
    """names 键集合 == models.json image 键集合（防漂移）。"""
    from src.backend.core.image.flux_schnell import FluxSchnellGenerator
    from src.backend.core.image.sdxl import SDXLGenerator
    from src.backend.core.image.sd35_medium import SD35MediumGenerator
    from src.backend.core.image.z_image import ZImageGenerator
    from src.backend.core.image.qwen_image import QwenImageLightningGenerator
    from src.backend.core.image.bg_removal import BgRemovalGenerator

    declared = {}
    for cls in (FluxSchnellGenerator, SDXLGenerator, SD35MediumGenerator,
                ZImageGenerator, QwenImageLightningGenerator,
                BgRemovalGenerator):
        declared.update(cls.names)
    assert set(declared.keys()) == set(_image_config().keys())


def test_bg_removal_is_image_generator():
    from src.backend.core.image.bg_removal import BgRemovalGenerator
    assert BgRemovalGenerator.type == FactoryType.Image
    assert callable(getattr(BgRemovalGenerator, "generate", None))
