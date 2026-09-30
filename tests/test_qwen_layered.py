"""Qwen-Image-Layered 分层生成器的契约测试（无权重、无 GPU、不需要 diffusers）。

真实管线的调用方式另在本机用极小随机权重端到端跑过；这里只守住配置、下载范围、
参数校验，以及 GGUF Embedding 解压这类纯逻辑。
"""
import pytest
import torch

from src.backend.core import diffusers_utils
from src.backend.core.generator_registry import register_all
from src.backend.core.image import qwen_layered
from src.backend.core.image.qwen_layered import QwenImageLayeredGenerator
from src.backend.core.model_base import GeneratorFactory, GeneratorSpec, SingletonMeta
from src.shared.enum_type import FactoryType

MODEL_NAME = "Qwen-Image-Layered"


@pytest.fixture
def generator(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen_layered, "IMAGE_MODEL_ROOT", tmp_path)
    SingletonMeta._instances.pop(QwenImageLayeredGenerator, None)
    yield QwenImageLayeredGenerator(model_name=MODEL_NAME, device="cpu")
    SingletonMeta._instances.pop(QwenImageLayeredGenerator, None)


def test_registered_for_library_but_hidden_from_chat():
    register_all()
    entry = GeneratorFactory._generators[FactoryType.Image][MODEL_NAME]
    cls = entry.load() if isinstance(entry, GeneratorSpec) else entry
    assert cls is QwenImageLayeredGenerator
    # 一次产出多张图层，聊天页的图片接口只收一张，所以不进模型下拉
    listed = [name for names in GeneratorFactory.get_model_info(FactoryType.Image).values()
              for name in names]
    assert MODEL_NAME not in listed and "Qwen-Image-Edit-2509" in listed


def test_reads_shared_encoder_and_tuning_from_models_json(generator, tmp_path):
    assert generator.gguf_filename.endswith(".gguf")
    assert generator.gguf_repo_id and generator.model_id == "Qwen/Qwen-Image-Layered"
    # 文本编码器与 Edit-2509 共用，不写死在代码里
    assert generator.text_encoder_local == tmp_path / "qwen-image-edit-2509"
    assert generator.offload.get("type") in ("leaf_level", "block_level")
    assert generator.defaults["resolution"] in qwen_layered.RESOLUTIONS
    generator._load_model()  # 分段加载：通用入口不预先载入任何一段
    assert generator.pipe is None and generator.encoder_pipe is None


def test_downloads_skip_gguf_transformer_and_shared_encoder(generator, monkeypatch):
    calls = []
    monkeypatch.setattr(qwen_layered, "ensure_snapshot",
                        lambda repo, local, ignore_patterns=None: calls.append(
                            ("snapshot", repo, local.name, tuple(ignore_patterns or ()))))
    monkeypatch.setattr(qwen_layered, "ensure_file",
                        lambda repo, filename, local: calls.append(("file", repo, filename)))
    (generator.text_encoder_local / "text_encoder").mkdir(parents=True)
    generator._check_model_file()
    assert ("snapshot", "Qwen/Qwen-Image-Layered", "qwen-image-layered",
            ("transformer/*", "text_encoder/*")) in calls
    # 共用目录按编辑模型自己的方式整仓下载，否则它以后会把半个目录当成已下载
    assert ("snapshot", "Qwen/Qwen-Image-Edit-2509", "qwen-image-edit-2509",
            ("transformer/*",)) in calls
    assert ("file", generator.gguf_repo_id, generator.gguf_filename) in calls


def test_missing_shared_encoder_is_reported(generator, monkeypatch):
    monkeypatch.setattr(qwen_layered, "ensure_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(qwen_layered, "ensure_file", lambda *args, **kwargs: None)
    with pytest.raises(FileNotFoundError, match="文本编码器"):
        generator._check_model_file()


def test_missing_encoder_config_is_reported(generator):
    generator.text_encoder_local = None
    with pytest.raises(ValueError, match="text_encoder"):
        generator._check_model_file()


def test_parse_params_applies_defaults_and_validates(generator):
    generator.parse_params({"content": "forest", "layers": "4"})
    assert generator.layers == 4 and generator.prompt == "forest"
    assert generator.num_inference_steps == generator.defaults["num_inference_steps"]
    assert generator.true_cfg_scale == generator.defaults["true_cfg_scale"]
    assert generator.resolution == 640
    with pytest.raises(ValueError, match="图层数"):
        generator.parse_params({"layers": 1})
    with pytest.raises(ValueError, match="分辨率"):
        generator.parse_params({"layers": 3, "resolution": 800})


def test_generate_requires_image_and_embeds(generator):
    import asyncio

    generator.parse_params({"layers": 3})
    with pytest.raises(ValueError, match="分层的图"):
        asyncio.run(generator.generate())
    generator.parse_params({"layers": 3, "reference_image": "scene.png"})
    with pytest.raises(ValueError, match="编码文件"):
        asyncio.run(generator.generate())


# ---- diffusers_utils ----
class _QuantizedWeight(torch.nn.Parameter):
    """仿 GGUFParameter：带 quant_type、形状是压缩后的字节。"""

    def __new__(cls, data, quant_type):
        instance = super().__new__(cls, data, requires_grad=False)
        instance.quant_type = quant_type
        return instance


def test_dequantize_gguf_embeddings_restores_only_quantized_embeddings():
    model = torch.nn.Module()
    model.quantized = torch.nn.Embedding(2, 8)
    model.quantized.weight = _QuantizedWeight(torch.zeros(2, 10, dtype=torch.uint8), "Q8_0")
    model.plain = torch.nn.Embedding(2, 8)
    model.linear = torch.nn.Linear(4, 4)
    plain_weight = model.plain.weight

    fixed = diffusers_utils.dequantize_gguf_embeddings(
        model, dequantize=lambda weight: torch.ones(2, 8))
    assert fixed == ["quantized"]
    assert model.quantized.weight.shape == (2, 8)
    assert model.quantized.weight.dtype == torch.bfloat16
    assert not model.quantized.weight.requires_grad
    assert model.plain.weight is plain_weight


def test_group_offload_options_are_passed_through():
    calls = []

    class _Pipe:
        vae = None

        def enable_group_offload(self, **kwargs):
            calls.append(kwargs)

    diffusers_utils.apply_offload(_Pipe(), "group", "cpu")
    diffusers_utils.apply_offload(_Pipe(), "group", "cpu",
                                  group={"type": "block_level", "blocks_per_group": 2,
                                         "use_stream": True})
    assert calls[0]["offload_type"] == "leaf_level" and calls[0]["use_stream"] is False
    assert "num_blocks_per_group" not in calls[0]
    assert calls[1]["offload_type"] == "block_level"
    assert calls[1]["num_blocks_per_group"] == 2 and calls[1]["use_stream"] is True
