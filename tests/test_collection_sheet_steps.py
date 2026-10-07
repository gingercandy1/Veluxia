"""组图步骤 runner 契约：LLM 写变体清单、按清单拼出图提示词。

用假的生成器，无权重、无 GPU 也可跑。
"""
import json
import threading
from typing import ClassVar

import pytest
from PIL import Image

from src.backend.core.collection.sheet_steps import (
    VARIANTS_NAME,
    SheetGenerateRunner,
    VariantsRunner,
    parse_variants,
)
from src.backend.core.collection.steps import StepContext
from src.backend.core.collection.template import load_template
from src.backend.core.model_base import (
    BaseImageGenerator,
    BaseTextGenerator,
    GeneratorFactory,
    SingletonMeta,
)
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem


class _FakeLLM(BaseTextGenerator):
    reply = ""
    calls: ClassVar[list] = []

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass

    async def generate(self):
        raise NotImplementedError

    def complete_chat(self, messages, max_tokens=1024, temperature=0.7, json_output=False,
                      seed=None):
        type(self).calls.append((messages, json_output, seed))
        return type(self).reply


class _FakeImage(BaseImageGenerator):
    output_dir = None

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        self.raw = raw

    async def generate(self):
        path = type(self).output_dir / "raw_output.png"
        Image.new("RGBA", (8, 8)).save(path)
        return path


@pytest.fixture
def fake_llm():
    _FakeLLM.reply, _FakeLLM.calls = "", []
    GeneratorFactory.register_generator(FactoryType.Text, "fake-llm", _FakeLLM)
    yield "fake-llm"
    GeneratorFactory._generators[FactoryType.Text].pop("fake-llm", None)
    SingletonMeta._instances.pop(_FakeLLM, None)
    GeneratorFactory._holder = None


@pytest.fixture
def fake_image(tmp_path):
    _FakeImage.output_dir = tmp_path
    GeneratorFactory.register_generator(FactoryType.Image, "fake-img", _FakeImage)
    yield "fake-img"
    GeneratorFactory._generators[FactoryType.Image].pop("fake-img", None)
    SingletonMeta._instances.pop(_FakeImage, None)
    GeneratorFactory._holder = None


def _ctx(tmp_path, params, inputs=(), step_id="sheet_variants"):
    out_dir = tmp_path / "pack" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="x", prompt="mushrooms"), prompt="mushrooms, pixel art",
        negative_prompt="blurry", inputs=list(inputs), params=params, out_dir=out_dir,
        step_id=step_id, values={"prompt": "mushrooms"},
    )


def test_variants_runner_writes_the_list_from_the_item_text(tmp_path, fake_llm):
    _FakeLLM.reply = json.dumps({"variants": ["tiny red cap", "huge white stalk", "small cluster",
                                              "extra"]})
    runner = VariantsRunner()
    params = {"model_name": fake_llm, "count": "3"}
    ctx = _ctx(tmp_path, params)
    with runner.open(params, threading.Event()):
        [path] = runner.run(ctx)
    assert path.name == VARIANTS_NAME
    # 多写的截掉，数量和出图方式一致
    assert parse_variants(path.read_text(encoding="utf-8")) == [
        "tiny red cap", "huge white stalk", "small cluster"]
    messages, json_output, seed = _FakeLLM.calls[0]
    # 给 LLM 的是条目原文，不带风格锁
    assert "mushrooms" in messages[1]["content"] and "pixel art" not in messages[1]["content"]
    assert json_output and seed is not None
    assert ctx.meta["variants"] == ["tiny red cap", "huge white stalk", "small cluster"]


def test_variants_runner_rejects_a_short_list(tmp_path, fake_llm):
    _FakeLLM.reply = json.dumps({"variants": ["only one", "two"]})
    runner = VariantsRunner()
    params = {"model_name": fake_llm, "count": "6"}
    with runner.open(params, threading.Event()), pytest.raises(ValueError, match="只写了 2 个"):
        runner.run(_ctx(tmp_path, params))


@pytest.mark.parametrize(("text", "message"), [
    ("not json", "不是合法的 JSON"),
    ('{"items": []}', "没有 variants 数组"),
    ('{"variants": ["", " "]}', "是空的"),
])
def test_parse_variants_explains_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_variants(text)


def test_sheet_generate_numbers_the_variants_in_the_prompt(tmp_path, fake_image):
    variants = tmp_path / VARIANTS_NAME
    variants.write_text(json.dumps({"variants": ["tiny red cap", "huge white stalk"]}),
                        encoding="utf-8")
    runner = SheetGenerateRunner()
    params = {"model_name": fake_image, "prompt_suffix": "white background"}
    with runner.open(params, threading.Event()):
        [path] = runner.run(_ctx(tmp_path, params, inputs=[variants], step_id="sheet_generate"))
        raw = SingletonMeta._instances[_FakeImage].raw
    assert raw == {"content": "mushrooms, pixel art, 2 separate objects: (1) tiny red cap; "
                              "(2) huge white stalk, white background",
                   "negative_prompt": "blurry"}
    assert path.name == "sheet_generate.png"


def test_sheet_generate_needs_the_variants_file(tmp_path, fake_image):
    runner = SheetGenerateRunner()
    params = {"model_name": fake_image}
    with runner.open(params, threading.Event()), pytest.raises(ValueError, match="缺少上游产物"):
        runner.run(_ctx(tmp_path, params, step_id="sheet_generate"))


def test_scene_props_sheet_chain_writes_variants_first():
    tpl = load_template("scene_props")
    assert tpl.step("sheet_generate").inputs == ("sheet_variants",)
    assert tpl.step("sheet_variants").params["count"] == "{sheet}"
