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

    def encode_prompt(self, prompt, device, image=None):
        import torch
        self.calls.append((prompt, image))
        # 和真管线一样：第三项标出参考图 token 的位置，纯文字时全为 False
        pad_mask = torch.zeros(1, 3, dtype=torch.bool)
        if image is not None:
            pad_mask[0, 1] = True
        return torch.ones(1, 3, 4), torch.ones(1, 3, dtype=torch.bool), pad_mask


class _FakePipe:
    """照 diffusers 0.41.0 的 QwenImage21Pipeline：__call__ 会拿着 prompt_embeds 再走一遍
    encode_prompt，给了参考图却没有 image_pad_mask 时报同样的错。"""

    def __init__(self, mode="RGBA"):
        self.mode, self.kwargs, self.pad_mask = mode, None, None

    def encode_prompt(self, prompt=None, image=None, device=None, num_images_per_prompt=1,
                      prompt_embeds=None, prompt_embeds_mask=None, image_pad_mask=None):
        if image_pad_mask is None:
            if image is not None:
                raise ValueError("Pass `image_pad_mask` alongside `prompt_embeds` when the "
                                 "embeddings cover condition images")
            image_pad_mask = prompt_embeds.new_zeros(prompt_embeds.shape[:2], dtype=bool)
        return prompt_embeds, prompt_embeds_mask, image_pad_mask

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        _, _, self.pad_mask = self.encode_prompt(
            image=kwargs.get("image"), prompt_embeds=kwargs["prompt_embeds"],
            prompt_embeds_mask=kwargs.get("prompt_embeds_mask"))
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
    assert [prompt for prompt, _ in generator.encoder_pipe.calls] == ["a", "b"]

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


def test_reference_image_conditions_both_encoder_and_denoiser(generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    # 透明底的参考图要先铺白底，不能直接转 RGB（会变黑底）
    ref = tmp_path / "ref.png"
    Image.new("RGBA", (512, 256), (255, 0, 0, 0)).save(ref)
    generator.parse_params({"content": "steeper ground", "reference_image": str(ref),
                            "width": 512, "height": 256})
    asyncio.run(generator.generate())

    encoded_prompt, encoded_image = generator.encoder_pipe.calls[0]
    assert encoded_prompt == "steeper ground"
    assert encoded_image[0].mode == "RGB" and encoded_image[0].getpixel((0, 0)) == (255, 255, 255)
    assert generator.pipe.kwargs["image"][0] is encoded_image[0]
    assert generator.pipe.kwargs["width"] == 512 and generator.pipe.kwargs["height"] == 256
    # 编码时得到的参考图 token 位置原样交给去噪管线；用完就撤掉，不留给下一次调用
    assert generator.pipe.pad_mask.tolist() == [[False, True, False]]
    assert "encode_prompt" not in vars(generator.pipe)


@pytest.mark.parametrize("size", [(1920, 1080), (512, 256), (1200, 640), (640, 1200), (64, 32)])
@pytest.mark.parametrize("resolution", [512, 768, 1024])
def test_edit_size_matches_the_diffusers_pipeline(size, resolution):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
    expected = calculate_dimensions(resolution * resolution, size[0] / size[1])[:2]
    assert qwen_image_21._edit_size(size, resolution) == expected


@pytest.mark.parametrize("size", [(1760, 320), (1672, 320), (200, 480), (240, 576), (304, 768),
                                  (1208, 768), (3600, 768), (64, 32)])
def test_reference_size_is_a_fixed_point_of_the_pipeline_resize(size):
    # 管线会把参考图再缩一次：缩两次的结果必须等于缩一次，否则编码器和 latent 的 token 数对不上
    stable = qwen_image_21._stable_edit_size(size, 768)
    assert qwen_image_21._edit_size(stable, 768) == stable


def test_a_size_the_pipeline_would_change_is_resized_until_it_stops_changing(
        generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    ref = tmp_path / "ref.png"
    # 1760x320 缩一次是 1760x320，管线再缩会变成 1792x320（实机上就是这样报的 shape mismatch）
    Image.new("RGB", (1672, 320)).save(ref)
    generator.parse_params({"content": "edit", "reference_image": str(ref)})
    asyncio.run(generator.generate())
    _, encoded_image = generator.encoder_pipe.calls[0]
    size = encoded_image[0].size
    assert qwen_image_21._edit_size(size, 768) == size


def test_reference_is_resized_once_so_encoder_and_denoiser_see_the_same_image(
        generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    # 原图远大于 768²：不缩的话编码器会产出远多于去噪阶段 latent 的图像 token（shape mismatch）
    ref = tmp_path / "ref.png"
    Image.new("RGB", (3840, 2160), (10, 20, 30)).save(ref)
    generator.parse_params({"content": "edit", "reference_image": str(ref)})
    asyncio.run(generator.generate())
    _, encoded_image = generator.encoder_pipe.calls[0]
    assert encoded_image[0].size == qwen_image_21._edit_size((3840, 2160), 768)
    assert generator.pipe.kwargs["image"][0] is encoded_image[0]


def test_cached_embeddings_keep_the_image_pad_mask(generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 256)).save(ref)
    for _ in range(2):
        generator.parse_params({"content": "edit", "reference_image": str(ref)})
        asyncio.run(generator.generate())
    # 第二次走缓存，不再编码，但掩码仍要带上，否则管线报错
    assert len(generator.encoder_pipe.calls) == 1
    assert generator.pipe.pad_mask.tolist() == [[False, True, False]]


def test_pad_mask_override_is_removed_even_when_denoising_fails(generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)

    def boom(**kwargs):
        raise RuntimeError("oom")

    generator.pipe.__class__ = type("_Failing", (_FakePipe,), {"__call__": lambda self, **k: boom()})
    ref = tmp_path / "ref.png"
    Image.new("RGB", (512, 256)).save(ref)
    generator.parse_params({"content": "edit", "reference_image": str(ref)})
    with pytest.raises(RuntimeError, match="oom"):
        asyncio.run(generator.generate())
    assert "encode_prompt" not in vars(generator.pipe)


def test_text_only_call_does_not_pass_image(generator, monkeypatch):
    _use_fakes(generator, monkeypatch)
    generator.parse_params({"content": "a"})
    asyncio.run(generator.generate())
    assert generator.encoder_pipe.calls == [("a", None)]  # 文生图：编码器不带参考图
    assert "image" not in generator.pipe.kwargs


def test_edit_defaults_output_resolution_to_768_but_text_to_image_passes_none(
        generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 32)).save(ref)
    # 有参考图且没指定：默认 768（1024 下编辑会忽略指令，diffusers #14824）
    generator.parse_params({"content": "edit", "reference_image": str(ref)})
    asyncio.run(generator.generate())
    assert generator.pipe.kwargs["output_resolution"] == 768
    # 调用方指定的值原样传下去
    generator.parse_params({"content": "edit", "reference_image": str(ref),
                            "output_resolution": 512})
    asyncio.run(generator.generate())
    assert generator.pipe.kwargs["output_resolution"] == 512
    # 文生图保持原样：不传 output_resolution，管线用自己的默认
    generator.parse_params({"content": "plain"})
    asyncio.run(generator.generate())
    assert "output_resolution" not in generator.pipe.kwargs


def test_output_resolution_must_be_in_range(generator):
    with pytest.raises(ValueError, match="output_resolution"):
        generator.parse_params({"content": "a", "output_resolution": 100})


def test_cache_is_keyed_by_reference_content(generator, monkeypatch, tmp_path):
    _use_fakes(generator, monkeypatch)
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(red)
    Image.new("RGB", (8, 8), (0, 0, 255)).save(blue)
    for ref in (red, red, blue):
        generator.parse_params({"content": "same prompt", "reference_image": str(ref)})
        asyncio.run(generator.generate())
    # 同一句提示词，但参考图换了，编码结果不能复用
    assert len(generator.encoder_pipe.calls) == 2


def test_cancelled_stage_loads_raise_without_loading(generator):
    """加载阶段点了停止：权重不读盘，直接抛 GenerationCancelled。"""
    import threading

    from src.backend.core.exceptions import GenerationCancelled
    generator.cancel_event = threading.Event()
    generator.cancel_event.set()
    with pytest.raises(GenerationCancelled):
        generator._use_encoder()
    assert generator.encoder_pipe is None
    with pytest.raises(GenerationCancelled):
        generator._use_denoiser()
    assert generator.pipe is None


def test_generate_aborts_before_encoding_when_cancelled(generator, monkeypatch):
    import threading

    from src.backend.core.exceptions import GenerationCancelled
    _use_fakes(generator, monkeypatch)
    generator.cancel_event = threading.Event()
    generator.cancel_event.set()
    generator.parse_params({"content": "hello"})
    with pytest.raises(GenerationCancelled):
        asyncio.run(generator.generate())
    assert generator.encoder_pipe.calls == []
