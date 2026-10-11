"""Qwen-Image-2.1（Turbo）生成器的契约测试（无权重、无 GPU）。

真实管线的加载和出图要在装好权重的机器上实测；这里守住配置读取、下载、参数校验、
两段加载的互斥、提示词编码缓存和 RGBA 保存这些纯逻辑。
"""
import asyncio
from types import SimpleNamespace

import diffusers
import pytest
from PIL import Image

from src.backend.core.generator_registry import register_all
from src.backend.core.image import qwen_image_21
from src.backend.core.image.qwen_image_21 import QwenImage21Generator
from src.backend.core.model_base import GeneratorFactory, GeneratorSpec, SingletonMeta
from src.shared.enum_type import FactoryType
from src.shared.settings import MODELS_DIR_ENV

MODEL_NAME = "Qwen-Image-2.1-Turbo"
GGUF_MODEL_NAME = "Qwen-Image-2.1-Turbo-Q4"


@pytest.fixture
def generator(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(qwen_image_21, "get_temp_dir", lambda name: str(tmp_path))
    SingletonMeta._instances.pop(QwenImage21Generator, None)
    yield QwenImage21Generator(model_name=MODEL_NAME, device="cpu")
    SingletonMeta._instances.pop(QwenImage21Generator, None)


class _FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode_prompt(self, prompt, device):
        import torch
        self.calls.append(prompt)
        return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool), None


class _FakePipe:
    def __init__(self, mode="RGBA"):
        self.mode, self.kwargs = mode, None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(images=[Image.new(self.mode, (64, 32), (10, 20, 30, 0))])


def _use_fakes(generator, monkeypatch, mode="RGBA"):
    """把两段加载换成预先放好的替身：只测 generate 本身的流程。"""
    generator.encoder_pipe, generator.pipe = _FakeEncoder(), _FakePipe(mode)
    monkeypatch.setattr(generator, "_use_encoder", lambda: None)
    monkeypatch.setattr(generator, "_use_denoiser", lambda: None)


def test_registered_and_listed_for_chat():
    register_all()
    entry = GeneratorFactory._generators[FactoryType.Image][MODEL_NAME]
    cls = entry.load() if isinstance(entry, GeneratorSpec) else entry
    assert cls is QwenImage21Generator
    listed = [name for names in GeneratorFactory.get_model_info(FactoryType.Image).values()
              for name in names]
    assert MODEL_NAME in listed


def test_reads_config_and_loads_nothing_up_front(generator, tmp_path):
    assert generator.model_id == "Qwen/Qwen-Image-2.1-Turbo"
    # 本地目录按仓库名取，路径走 models_root()，不是模块级常量
    assert generator.base_local == tmp_path / "image" / "qwen-image-2.1-turbo"
    assert generator.offload["type"] == "block_level"
    assert generator.offload["leaf_components"] == ["vae"]
    generator._load_model()
    assert generator.pipe is None and generator.encoder_pipe is None


def test_download_uses_the_configured_repo_and_requires_one(generator, monkeypatch):
    calls = []
    monkeypatch.setattr(qwen_image_21, "ensure_snapshot",
                        lambda repo, local: calls.append((repo, local.name)))
    generator._check_model_file()
    assert calls == [("Qwen/Qwen-Image-2.1-Turbo", "qwen-image-2.1-turbo")]
    generator.model_id = None
    with pytest.raises(ValueError, match="repo_id"):
        generator._check_model_file()


@pytest.fixture
def gguf_generator(tmp_path, monkeypatch):
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path))
    SingletonMeta._instances.pop(QwenImage21Generator, None)
    yield QwenImage21Generator(model_name=GGUF_MODEL_NAME, device="cpu")
    SingletonMeta._instances.pop(QwenImage21Generator, None)


def test_gguf_entry_uses_its_own_dir_and_the_quantized_file(gguf_generator, tmp_path):
    assert gguf_generator.model_id == "Qwen/Qwen-Image-2.1-Turbo"
    assert gguf_generator.gguf_filename.endswith(".gguf") and gguf_generator.gguf_repo_id
    # 量化版的骨架不含 transformer 权重，不能和整仓下载的目录混用
    assert gguf_generator.base_local == tmp_path / "image" / "qwen-image-2.1-turbo-gguf"
    assert gguf_generator.gguf_local == gguf_generator.base_local / "gguf"


def test_gguf_download_skips_the_full_transformer(gguf_generator, monkeypatch):
    calls = []
    monkeypatch.setattr(qwen_image_21, "ensure_snapshot",
                        lambda repo, local, ignore_patterns=None, marker=None: calls.append(
                            ("snapshot", repo, tuple(ignore_patterns or ()), marker)))
    monkeypatch.setattr(qwen_image_21, "ensure_file",
                        lambda repo, filename, local: calls.append(("file", repo, filename)))
    gguf_generator._check_model_file()
    assert calls == [
        # marker：骨架以 model_index.json 判定，目录里只有 config / GGUF 时仍要补下载
        ("snapshot", "Qwen/Qwen-Image-2.1-Turbo", ("transformer/*",), "model_index.json"),
        ("file", "Qwen/Qwen-Image-2.1-Turbo", "transformer/config.json"),
        ("file", gguf_generator.gguf_repo_id, gguf_generator.gguf_filename),
    ]
    gguf_generator.gguf_repo_id = ""
    with pytest.raises(ValueError, match="gguf_repo_id"):
        gguf_generator._check_model_file()


def test_gguf_denoiser_passes_the_quantized_transformer_to_the_pipeline(
        gguf_generator, monkeypatch):
    loaded, built = [], []
    marker = object()

    def fake_loader(model_cls, path, base):
        loaded.append((model_cls.__name__, path.name, base))
        return marker

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            built.append(kwargs)
            return _FakePipe()

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", FakePipeline, raising=False)
    monkeypatch.setattr(qwen_image_21, "load_gguf_transformer", fake_loader)
    monkeypatch.setattr(qwen_image_21, "dequantize_gguf_embeddings", lambda module: ["emb"])
    monkeypatch.setattr(qwen_image_21, "apply_offload", lambda *args, **kwargs: None)

    gguf_generator._use_denoiser()

    assert loaded == [("QwenImage21Transformer2DModel", gguf_generator.gguf_filename,
                       gguf_generator.base_local)]
    assert built[0]["transformer"] is marker and built[0]["text_encoder"] is None


def test_full_weights_entry_never_touches_the_gguf_path(generator, monkeypatch):
    assert generator.gguf_filename is None
    built = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            built.append(kwargs)
            return _FakePipe()

    def boom(*args, **kwargs):
        raise AssertionError("全量权重不该走 GGUF 加载")

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", FakePipeline, raising=False)
    monkeypatch.setattr(qwen_image_21, "load_gguf_transformer", boom)
    monkeypatch.setattr(qwen_image_21, "apply_offload", lambda *args, **kwargs: None)
    generator._use_denoiser()
    assert "transformer" not in built[0]


@pytest.mark.parametrize("raw, message", [
    ({"content": "  "}, "需要提示词"),
    ({"content": "a", "width": 500}, "32 的倍数"),
    ({"content": "a", "height": 128}, "之间"),
    ({"content": "a", "width": 4096}, "之间"),
])
def test_parse_params_rejects_bad_input(generator, raw, message):
    with pytest.raises(ValueError, match=message):
        generator.parse_params(raw)


def test_parse_params_applies_defaults(generator):
    generator.parse_params({"content": " a forest "})
    assert (generator.prompt, generator.width, generator.height) == ("a forest", 1024, 1024)
    assert generator.save_path.suffix == ".png"


def test_generate_keeps_alpha_and_does_not_pass_steps_or_cfg(generator, monkeypatch):
    _use_fakes(generator, monkeypatch)
    generator.parse_params({"content": "ground strip", "width": 512, "height": 256})

    path = asyncio.run(generator.generate())

    with Image.open(path) as saved:
        assert saved.mode == "RGBA" and saved.getpixel((0, 0))[3] == 0
    kwargs = generator.pipe.kwargs
    assert (kwargs["width"], kwargs["height"]) == (512, 256)
    # Turbo 的采样日程存在检查点里：传步数 / CFG 不会生效，也不该传
    assert "num_inference_steps" not in kwargs and "true_cfg_scale" not in kwargs
    assert "prompt" not in kwargs and kwargs["prompt_embeds"].shape == (1, 3, 4)


def test_same_prompt_skips_the_encoder_and_unload_clears_the_cache(generator, monkeypatch):
    _use_fakes(generator, monkeypatch)
    for prompt in ("a", "a", "b"):
        generator.parse_params({"content": prompt})
        asyncio.run(generator.generate())
    # 连续同一句提示词只编码一次；换了提示词才重新编码
    assert generator.encoder_pipe.calls == ["a", "b"]

    generator.unload()
    assert generator._embeds_cache is None and generator.pipe is None
    assert generator.encoder_pipe is None


def test_two_stages_are_mutually_exclusive_and_load_only_their_half(generator, monkeypatch):
    loads, offloads = [], []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loads.append(kwargs)
            return _FakePipe()

    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", FakePipeline, raising=False)
    monkeypatch.setattr(qwen_image_21, "apply_offload",
                        lambda pipe, mode, device, group=None: offloads.append((mode, group)))

    generator.pipe = object()
    generator._use_encoder()
    assert generator.pipe is None and generator.encoder_pipe is not None
    assert loads[0]["transformer"] is None and loads[0]["vae"] is None
    assert "text_encoder" not in loads[0]

    generator._use_denoiser()
    assert generator.encoder_pipe is None and generator.pipe is not None
    assert loads[1]["text_encoder"] is None and "transformer" not in loads[1]
    # 去噪阶段用 models.json 里配的卸载方式；编码阶段用默认的按层卸载
    assert offloads == [("group", None), ("group", generator.offload)]
