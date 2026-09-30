"""分层步骤契约：编码 / 去噪两步 runner 的参数与产物，以及合并后的"场景"模板。

用假的分层生成器，无权重、无 GPU 也可跑。
"""
import threading

import pytest
from PIL import Image

from src.backend.core.collection.executor import CollectionExecutor
from src.backend.core.collection.layer_steps import LayerEncodeRunner, LayerRunner
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template
from src.backend.core.model_base import BaseImageGenerator, GeneratorFactory, SingletonMeta
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem

MODEL = "fake-layered"


class _FakeLayered(BaseImageGenerator):
    output_dir = None
    layer_count = 3

    def _check_model_file(self):
        pass

    def _load_model(self):
        """分段加载的生成器：通用入口什么都不加载。"""

    def parse_params(self, raw):
        self.raw = raw

    def save_prompt_embeds(self):
        path = type(self).output_dir / "embeds.pt"
        path.write_bytes(b"embeds")
        return path

    async def generate(self):
        paths = []
        for index in range(type(self).layer_count):
            path = type(self).output_dir / f"raw_layer_{index}.png"
            Image.new("RGBA", (32, 16), (index, 0, 0, 255)).save(path)
            paths.append(path)
        return paths


@pytest.fixture
def fake_layered(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _FakeLayered.output_dir = raw
    _FakeLayered.layer_count = 3
    GeneratorFactory.register_generator(FactoryType.Image, MODEL, _FakeLayered)
    yield
    GeneratorFactory._generators[FactoryType.Image].pop(MODEL, None)
    SingletonMeta._instances.pop(_FakeLayered, None)
    GeneratorFactory._holder = None


def _ctx(tmp_path, params, inputs=(), step_id="layer_encode"):
    out_dir = tmp_path / "pack" / "01"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="01", prompt="forest"), prompt="forest, hand-painted",
        negative_prompt="", inputs=list(inputs), params=params, out_dir=out_dir,
        step_id=step_id, values={"prompt": "forest"}, style_prompt="hand-painted",
    )


def test_encode_uses_the_generation_prompt(tmp_path, fake_layered):
    params = {"model_name": MODEL, "prompt_suffix": "side view, no text"}
    runner = LayerEncodeRunner()
    ctx = _ctx(tmp_path, params)
    with runner.open(params, threading.Event()):
        [embeds] = runner.run(ctx)
        generator = SingletonMeta._instances[_FakeLayered]
    # 和生成整图时同一句：条目描述 + 风格锁 + 后缀，不让模型看图重写
    assert generator.raw == {"content": "forest, hand-painted, side view, no text"}
    assert embeds.name == "layer_encode.pt" and embeds.parent == ctx.out_dir
    assert ctx.meta["prompt"] == "forest, hand-painted, side view, no text"


def test_layer_denoises_into_numbered_layers(tmp_path, fake_layered):
    scene = tmp_path / "generate.png"
    Image.new("RGB", (64, 32)).save(scene)
    embeds = tmp_path / "layer_encode.pt"
    embeds.write_bytes(b"embeds")
    params = {"model_name": MODEL, "layers": "3"}
    runner = LayerRunner()
    ctx = _ctx(tmp_path, params, [scene, embeds], step_id="layer")
    with runner.open(params, threading.Event()):
        outputs = runner.run(ctx)
        generator = SingletonMeta._instances[_FakeLayered]
    assert generator.raw == {"layers": "3", "reference_image": str(scene),
                             "prompt_embeds_path": str(embeds)}
    # 从底到顶编号，交付和导出时一眼能看出叠放顺序
    assert [p.name for p in outputs] == ["layer_1.png", "layer_2.png", "layer_3.png"]
    with Image.open(outputs[0]) as bottom:
        assert bottom.getpixel((0, 0))[0] == 0
    assert ctx.meta["layers"] == 3 and ctx.meta["size"] == [32, 16]
    assert "reference_image" not in ctx.meta["params"]


def test_layer_needs_scene_and_embeds(tmp_path, fake_layered):
    scene = tmp_path / "generate.png"
    Image.new("RGB", (8, 8)).save(scene)
    params = {"model_name": MODEL, "layers": "3"}
    runner = LayerRunner()
    with runner.open(params, threading.Event()), pytest.raises(ValueError, match="编码文件"):
        runner.run(_ctx(tmp_path, params, [scene], step_id="layer"))


def test_layer_without_output_is_an_error(tmp_path, fake_layered):
    _FakeLayered.layer_count = 0
    scene, embeds = tmp_path / "generate.png", tmp_path / "e.pt"
    Image.new("RGB", (8, 8)).save(scene)
    embeds.write_bytes(b"x")
    params = {"model_name": MODEL, "layers": "3"}
    runner = LayerRunner()
    with runner.open(params, threading.Event()), pytest.raises(RuntimeError, match="没有生成图层"):
        runner.run(_ctx(tmp_path, params, [scene, embeds], step_id="layer"))


# ---- 合并后的"场景"模板 ----
def test_scene_template_merges_layering_into_one_template(tmp_path):
    template = load_template("scene_background")
    assert template.name == "场景" and template.cover == "upscale"
    assert {f.id for f in template.fields} >= {"view", "tile", "length", "segment_desc", "layers"}
    layers = next(f for f in template.fields if f.id == "layers")
    assert layers.default == "0" and [v for v, _ in layers.options] == ["0", "3", "4", "6"]
    # 分层一张十几分钟，不让 AI 起草随手选上
    assert layers.draft is False
    assert [s.id for s in template.steps if s.deliverable] == ["upscale", "layers"]
    conditional = {s.id for s in template.steps if s.when == "layers"}
    assert conditional == {"layer_encode", "layer", "layers"}
    # 分层用生成时的 1344×768 原图，不用放大后的
    assert template.step("layer").inputs == ("generate", "layer_encode")
    assert template.step("layer").params["layers"] == "{layers}"
    assert all(step.type in registered_runners() for step in template.steps)
    CollectionExecutor(tmp_path, template)


def test_old_scene_templates_are_gone():
    for template_id in ("scene_ground", "scene_parallax"):
        with pytest.raises(ValueError, match="模板不存在"):
            load_template(template_id)


@pytest.mark.parametrize("fields, active", [
    ({}, False),
    ({"layers": "0"}, False),
    ({"layers": "4"}, True),
])
def test_layer_steps_follow_the_layers_field(fields, active):
    template = load_template("scene_background")
    assert template.step_active("upscale", fields)
    for step_id in ("layer_encode", "layer", "layers"):
        assert template.step_active(step_id, fields) is active


def test_layers_only_on_single_screen():
    template = load_template("scene_background")
    template.check_item_fields({"layers": "4", "length": "1"})
    template.check_item_fields({"layers": "0", "length": "3"})
    with pytest.raises(ValueError, match="分层只支持单屏.*2 屏"):
        template.check_item_fields({"layers": "4", "length": "2"})
