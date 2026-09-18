"""
生成器注册：models.json 是模型清单的唯一来源。

新增 / 删除模型只改 models.json：
- 每个模型项写 "generator": "包.模块:类名"（模块路径相对 src.backend.core），tag 决定 UI 分组；
- 同一类别下多数模型共用一个生成器时，可在类别里写 "_generator" 作为默认值，模型项可用 "generator" 覆盖；
- 同一个生成器类可对应多个模型名（如 LTX-2.3 / LTX-2.5）。
只有新增“生成器类”（新的 .py 实现）时才需要新写代码，且仍不用改这里。

启动时只读 json，不 import 生成器；类在首次生成时才加载，warmup() 可在后台提前加载。
"""
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
