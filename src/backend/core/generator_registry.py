from typing import Dict
from src.backend.core.model_base import GeneratorFactory, GeneratorSpec, load_models_config
from src.shared.enum_type import FactoryType

_DEFAULT_KEY = "_generator"


def _iter_models():
    """产出 (类型, 模型名, tag, GeneratorSpec)；缺少 generator 的条目会被跳过并提示。"""
    config = load_models_config()
    for ty in FactoryType:
        group = config.get(FactoryType.convert_to_text(ty))
        if not isinstance(group, dict):
            continue
        default = group.get(_DEFAULT_KEY)
        for name, info in group.items():
            if not isinstance(info, dict):
                continue
            path = info.get("generator", default)
            if not path:
                print(f"⚠️ models.json: {ty.name}/{name} 未指定 generator，已跳过")
                continue
            yield ty, name, str(info.get("tag", "")), GeneratorSpec.parse(path)


def register_all() -> None:
    """只读 models.json，不 import 任何生成器模块，毫秒级完成。"""
    for ty, name, tag, spec in _iter_models():
        GeneratorFactory.register_generator(ty, name, spec)
        GeneratorFactory.register_model_info(ty, tag, name)


def warmup() -> None:
    """后台预热：提前 import 重型依赖与全部生成器，缩短首次生成等待。不影响名称列表可用性。"""
    import torch  # noqa: F401
    specs: Dict[GeneratorSpec, None] = {spec: None for *_, spec in _iter_models()}
    for spec in specs:
        try:
            spec.load()
        except Exception as e:
            print(f"⚠️ 预加载失败 {spec.module}: {e}")
