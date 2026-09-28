"""转面图契约（ADR 0006）：编码 / 去噪两步 runner 的参数与产物、模板、生成器的分段加载与编码缓存。

用假的生成器和假的编码器管线，无权重、无 GPU 也可跑。
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.backend.core.collection.edit_steps import EditEncodeRunner, EditRunner
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template
from src.backend.core.image.qwen_edit import QwenImageEditPlusGenerator
from src.backend.core.model_base import (
    BaseImageGenerator,
    GeneratorFactory,
    GeneratorSpec,
    SingletonMeta,
    load_models_config,
)
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem

MODEL_NAME = "Qwen-Image-Edit-2509"


def _ctx(tmp_path, params, inputs=(), step_id="encode", prompt="将镜头向左旋转45度"):
    out_dir = tmp_path / "pack" / "left45"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="left45", prompt=prompt), prompt=f"{prompt}, pixel art",
        negative_prompt="", inputs=list(inputs), params=params, out_dir=out_dir,
        step_id=step_id, values={"prompt": prompt}, style_prompt="pixel art",
    )


class _FakeEdit(BaseImageGenerator):
    output_dir = None

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        self.raw = raw
        with Image.open(raw["reference_image"]) as reference:
            self.reference_size = reference.size

    def save_prompt_embeds(self):
        path = type(self).output_dir / "embeds.pt"
        path.write_bytes(b"embeds")
        return path

    async def generate(self):
        path = type(self).output_dir / "edited.png"
        Image.new("RGB", (40, 40), "white").save(path)
        return path


@pytest.fixture
def fake_edit(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _FakeEdit.output_dir = raw
    GeneratorFactory.register_generator(FactoryType.Image, "fake-edit", _FakeEdit)
    yield
    GeneratorFactory._generators[FactoryType.Image].pop("fake-edit", None)
    SingletonMeta._instances.pop(_FakeEdit, None)
    GeneratorFactory._holder = None


def test_encode_runner_writes_reference_and_embeds(tmp_path, fake_edit):
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (34, 68), (0, 0, 255, 255)).save(portrait)
    params = {"model_name": "fake-edit", "scale": 0.5, "prompt_suffix": "纯白背景"}
    runner = EditEncodeRunner()
    ctx = _ctx(tmp_path, params, [portrait])
    with runner.open(params, threading.Event()):
        outputs = runner.run(ctx)
        generator = SingletonMeta._instances[_FakeEdit]
    # 只用条目原文 + 后缀，不拼风格锁
    assert generator.raw["content"] == "将镜头向左旋转45度, 纯白背景"
    assert "scale" not in generator.raw and "prompt_suffix" not in generator.raw
    assert generator.reference_size == (136, 136)
    # 参考图留作产物：去噪步骤必须用编码时的同一张图
    assert [p.name for p in outputs] == ["encode.png", "encode.pt"]
    assert all(p.parent == ctx.out_dir and p.is_file() for p in outputs)
    assert "reference_image" not in ctx.meta.get("params", {})
    assert ctx.meta["prompt"] == "将镜头向左旋转45度, 纯白背景"


def test_encode_runner_requires_instruction(tmp_path, fake_edit):
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(portrait)
    params = {"model_name": "fake-edit"}
    runner = EditEncodeRunner()
    with runner.open(params, threading.Event()), pytest.raises(ValueError, match="指令"):
        runner.run(_ctx(tmp_path, params, [portrait], prompt=""))


def test_edit_runner_denoises_from_encode_outputs(tmp_path, fake_edit):
    reference = tmp_path / "encode.png"
    Image.new("RGB", (20, 20), "white").save(reference)
    embeds = tmp_path / "encode.pt"
    embeds.write_bytes(b"embeds")
    params = {"model_name": "fake-edit", "num_inference_steps": 4}
    runner = EditRunner()
    ctx = _ctx(tmp_path, params, [reference, embeds], step_id="turn")
    with runner.open(params, threading.Event()):
        outputs = runner.run(ctx)
        generator = SingletonMeta._instances[_FakeEdit]
    assert generator.raw["reference_image"] == str(reference)
    assert generator.raw["prompt_embeds_path"] == str(embeds)
    assert [p.name for p in outputs] == ["turn.png"]
    assert ctx.meta["size"] == [40, 40] and ctx.meta["params"] == {"num_inference_steps": 4}


def test_edit_runner_rejects_inputs_without_embeds(tmp_path, fake_edit):
    reference = tmp_path / "encode.png"
    Image.new("RGB", (20, 20), "white").save(reference)
    params = {"model_name": "fake-edit"}
    runner = EditRunner()
    with runner.open(params, threading.Event()), pytest.raises(ValueError, match="编码文件"):
        runner.run(_ctx(tmp_path, params, [reference], step_id="turn"))


def test_character_turnaround_template_encodes_then_denoises():
    template = load_template("character_turnaround")
    assert template.source == "character" and template.cover == "trim"
    assert template.step("encode").inputs == ("@source",)
    assert template.step("turn").inputs == ("encode",)
    runners = registered_runners()
    assert all(step.type in runners for step in template.steps)
    # 两步都要用 models.json 里登记的编辑模型
    assert {template.step(s).params["model_name"] for s in ("encode", "turn")} == {MODEL_NAME}


# ---- 生成器 ----
class _FakeEncoderPipe:
    def __init__(self):
        self.calls = []
        self.image_processor = SimpleNamespace(
            resize=lambda image, height, width: image.resize((width, height)))

    def encode_prompt(self, prompt, image, device):
        self.calls.append((prompt, image[0].size))
        return torch.full((1, 3, 4), float(len(self.calls))), None


@pytest.fixture
def edit_generator(monkeypatch):
    SingletonMeta._instances.pop(QwenImageEditPlusGenerator, None)
    generator = QwenImageEditPlusGenerator(model_name=MODEL_NAME, device="cpu")
    encoder = _FakeEncoderPipe()

    def use_encoder():
        generator.encoder_pipe = encoder
    monkeypatch.setattr(generator, "_use_encoder", use_encoder)
    yield generator, encoder
    SingletonMeta._instances.pop(QwenImageEditPlusGenerator, None)


def test_edit_model_is_registered_with_gguf_and_local_loras():
    entry = load_models_config()["image"][MODEL_NAME]
    assert GeneratorSpec.parse(entry["generator"]).load() is QwenImageEditPlusGenerator
    SingletonMeta._instances.pop(QwenImageEditPlusGenerator, None)
    generator = QwenImageEditPlusGenerator(model_name=MODEL_NAME, device="cpu")
    try:
        assert generator.gguf_filename.endswith(".gguf")
        assert [lora.name for lora in generator.loras] == ["lightning", "camera"]
        # LoRA 和底座放在一起，不进公共 LoRA 目录
        assert all(generator.base_local / "lora" in lora.path.parents for lora in generator.loras)
        # 分段加载：通用加载入口不预先载入任何一段
        generator._load_model()
        assert generator.pipe is None and generator.encoder_pipe is None
    finally:
        SingletonMeta._instances.pop(QwenImageEditPlusGenerator, None)


def test_prompt_embeds_are_cached_per_image_and_instruction(edit_generator):
    generator, encoder = edit_generator
    image = Image.new("RGB", (300, 600), "white")
    first = generator._prompt_embeds(image, "将镜头向左旋转45度")
    again = generator._prompt_embeds(image.copy(), "将镜头向左旋转45度")
    assert again is first and len(encoder.calls) == 1
    generator._prompt_embeds(image, "将镜头向左旋转90度")
    generator._prompt_embeds(Image.new("RGB", (300, 600), "black"), "将镜头向左旋转45度")
    assert len(encoder.calls) == 3
    # 编码器看的是按面积缩小的参考图，比例大致不变（宽高各取 32 的倍数）
    width, height = encoder.calls[0][1]
    assert width * height <= 384 * 384 * 1.1 and abs(height / width - 2) < 0.2


def test_save_prompt_embeds_round_trips(edit_generator, tmp_path):
    generator, _ = edit_generator
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(portrait)
    generator.parse_params({"reference_image": str(portrait), "content": "将镜头向左旋转45度"})
    path = generator.save_prompt_embeds()
    try:
        assert path.suffix == ".pt"
        generator.prompt_embeds_path = str(path)
        embeds, mask = generator._load_embeds()
        assert torch.equal(embeds, torch.ones(1, 3, 4)) and mask is None
    finally:
        path.unlink(missing_ok=True)


def test_reference_flattens_transparency_onto_white(edit_generator, tmp_path):
    generator, _ = edit_generator
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(portrait)
    generator.parse_params({"reference_image": str(portrait)})
    assert generator._reference().getpixel((0, 0)) == (255, 255, 255)


def test_generate_without_reference_is_an_error(edit_generator):
    generator, _ = edit_generator
    generator.parse_params({"content": "将镜头向左旋转45度"})
    with pytest.raises(ValueError, match="参考图"):
        asyncio.run(generator.generate())
