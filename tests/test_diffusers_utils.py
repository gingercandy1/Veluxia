"""diffusers 公共加载步骤的契约测试（无模型权重、无 GPU 也可跑）。

契约：models.json 里 GGUF 模型统一为 repo_id=官方骨架、gguf_repo_id=量化仓库、
filename=.gguf 文件名；LoRA 用 "loras" 列表声明。字段写错要在加载前就暴露。
"""
import json
from unittest.mock import MagicMock

import pytest

from src.backend.core.diffusers_utils import (
    LORA_ROOT,
    LoraSpec,
    apply_loras,
    apply_offload,
    gguf_filename,
    parse_loras,
)
from src.shared.settings import PROJECT_ROOT


def _all_models() -> dict:
    with open(f"{PROJECT_ROOT}/models.json", encoding="utf-8") as f:
        config = json.load(f)
    return {name: info
            for group in config.values()
            for name, info in group.items()
            if not name.startswith("_") and isinstance(info, dict)}


def test_gguf_filename_only_accepts_gguf():
    assert gguf_filename("model-Q4_K_M.gguf") == "model-Q4_K_M.gguf"
    assert gguf_filename("model.safetensors") is None
    assert gguf_filename(None) is None


@pytest.mark.parametrize("name", [
    name for name, info in _all_models().items()
    if info.get("generator", "").startswith(("image.", "animation."))
    and gguf_filename(info.get("filename"))
])
def test_diffusers_gguf_models_declare_both_repos(name):
    # 文本模型（llama.cpp）的 GGUF 只有一个仓库，只约束 diffusers 这一侧
    info = _all_models()[name]
    assert info.get("repo_id"), "repo_id 应指向官方骨架（结构配置 / VAE / 文本编码器）"
    assert info.get("gguf_repo_id"), "gguf_repo_id 应指向量化权重仓库"
    assert info["repo_id"] != info["gguf_repo_id"]
    assert "base_id" not in info, "base_id 已废弃，骨架统一写在 repo_id"


def test_models_json_loras_parse():
    for name, info in _all_models().items():
        for lora in parse_loras(info):
            assert lora.weight_name.endswith(".safetensors"), name


def test_parse_loras_defaults_and_names():
    specs = parse_loras({"loras": [
        {"repo_id": "a/b", "weight_name": "x.safetensors", "name": "fast", "scale": 0.8},
        {"repo_id": "c/d", "weight_name": "sub/y.safetensors"},
    ]})
    assert specs[0] == LoraSpec("a/b", "x.safetensors", "fast", 0.8)
    # 没写名字要自动编号，否则 set_adapters 会把两个 LoRA 当成同一个
    assert specs[1].name == "lora_1"
    assert specs[1].scale == 1.0
    assert specs[1].path == LORA_ROOT / "c--d" / "sub" / "y.safetensors"


def test_parse_loras_rejects_missing_fields():
    with pytest.raises(ValueError, match="weight_name"):
        parse_loras({"loras": [{"repo_id": "a/b"}]})
    assert parse_loras(None) == []


def test_apply_loras_sets_every_adapter_with_scale():
    pipe = MagicMock()
    apply_loras(pipe, [LoraSpec("a/b", "sub/x.safetensors", "one", 1.0),
                       LoraSpec("c/d", "y.safetensors", "two", 0.5)])
    first = pipe.load_lora_weights.call_args_list[0]
    assert first.args == (str(LORA_ROOT / "a--b" / "sub"),)
    assert first.kwargs == {"weight_name": "x.safetensors", "adapter_name": "one"}
    pipe.set_adapters.assert_called_once_with(["one", "two"], adapter_weights=[1.0, 0.5])


def test_apply_loras_noop_without_loras():
    pipe = MagicMock()
    apply_loras(pipe, [])
    pipe.load_lora_weights.assert_not_called()
    pipe.set_adapters.assert_not_called()


def test_apply_offload_modes():
    pipe = MagicMock()
    apply_offload(pipe, "model")
    pipe.enable_model_cpu_offload.assert_called_once()
    pipe.vae.enable_slicing.assert_called_once()
    pipe.vae.enable_tiling.assert_called_once()

    pipe = MagicMock()
    apply_offload(pipe, "sequential")
    pipe.enable_sequential_cpu_offload.assert_called_once()

    with pytest.raises(ValueError):
        apply_offload(MagicMock(), "unknown")
