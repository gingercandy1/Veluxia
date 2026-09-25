"""生成器注册契约测试（无模型权重、无 GPU 也可跑）。

契约：models.json 是模型清单的唯一来源，register_all() 按它注册；
每个条目的 generator 路径都必须能导入到对应类型基类的子类，
否则 UI 下拉里能选到、真正生成时才报"找不到生成器"。
"""
import json

import pytest

from src.backend.core.generator_registry import register_all
from src.backend.core.model_base import (
    BaseAnimationGenerator,
    BaseImageFrameGenerator,
    BaseImageGenerator,
    BaseSpeechGenerator,
    BaseTextGenerator,
    BaseTranscriptionGenerator,
    GeneratorFactory,
    GeneratorSpec,
)
from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT

_BASES = {
    FactoryType.Image: BaseImageGenerator,
    FactoryType.Text: BaseTextGenerator,
    FactoryType.Animation: BaseAnimationGenerator,
    FactoryType.Speech: BaseSpeechGenerator,
    FactoryType.ImageFrame: BaseImageFrameGenerator,
    FactoryType.Transcription: BaseTranscriptionGenerator,
}


def _models(ty: FactoryType) -> dict:
    with open(f"{PROJECT_ROOT}/models.json", encoding="utf-8") as f:
        group = json.load(f).get(FactoryType.convert_to_text(ty), {})
    # 以下划线开头的是类别级配置（如 _generator），不是模型
    return {name: info for name, info in group.items() if not name.startswith("_")}


def _registered_models():
    register_all()
    for ty, base in _BASES.items():
        for name in _models(ty):
            yield pytest.param(ty, base, name, id=f"{ty.name}/{name}")


def test_image_models_registered():
    register_all()
    names = GeneratorFactory.get_generator_names(FactoryType.Image)
    for expected in ("Flux.1-schnell", "SDXL", "SD3.5-Medium", "Z-Image-Turbo",
                     "Qwen-Image-Lightning", "rembg-u2net"):
        assert expected in names, f"{expected} 未注册到 Factory"


@pytest.mark.parametrize("ty, base, name", list(_registered_models()))
def test_every_model_resolves_to_its_base_class(ty, base, name):
    entry = GeneratorFactory._generators[ty].get(name)
    assert entry is not None, f"{name} 未注册到 Factory"
    cls = entry.load() if isinstance(entry, GeneratorSpec) else entry
    assert issubclass(cls, base), f"{name} -> {cls.__name__} 不是 {base.__name__} 的子类"


def test_bg_removal_is_image_generator():
    from src.backend.core.image.bg_removal import BgRemovalGenerator
    assert BgRemovalGenerator.type == FactoryType.Image
    assert callable(getattr(BgRemovalGenerator, "generate", None))
