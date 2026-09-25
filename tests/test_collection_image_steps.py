"""图片类步骤 runner 契约（ADR 0004）：租约范围、参数拼装、产物落点、失败转异常。

用假的生成器，无权重、无 GPU 也可跑。
"""
import threading

import pytest
from PIL import Image

from src.backend.core.collection.executor import CollectionExecutor
from src.backend.core.collection.image_steps import (
    ImageGenerateRunner,
    RemoveBackgroundRunner,
    TrimRunner,
    trim_transparent,
)
from src.backend.core.collection.steps import StepContext
from src.backend.core.collection.template import load_template
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import BaseImageGenerator, GeneratorFactory, SingletonMeta
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem


class _FakeImage(BaseImageGenerator):
    output_dir = None
    fail = False

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        self.raw = raw

    async def generate(self):
        if type(self).fail:
            return None
        path = type(self).output_dir / "raw_output.png"
        Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(path)
        return path


@pytest.fixture
def fake_model(tmp_path):
    _FakeImage.output_dir = tmp_path
    _FakeImage.fail = False
    GeneratorFactory.register_generator(FactoryType.Image, "fake-img", _FakeImage)
    yield "fake-img"
    GeneratorFactory._generators[FactoryType.Image].pop("fake-img", None)
    SingletonMeta._instances.pop(_FakeImage, None)
    GeneratorFactory._holder = None


def _ctx(tmp_path, params, inputs=(), step_id="generate"):
    out_dir = tmp_path / "pack" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="x", prompt="mushroom"), prompt="mushroom, dark",
        negative_prompt="blurry", inputs=list(inputs), params=params,
        out_dir=out_dir, step_id=step_id,
    )


def test_generate_builds_prompt_and_moves_output_into_pack(tmp_path, fake_model):
    runner = ImageGenerateRunner()
    params = {"model_name": fake_model, "prompt_suffix": "white background", "width": 512}
    with runner.open(params, threading.Event()):
        [path] = runner.run(_ctx(tmp_path, params))
        raw = SingletonMeta._instances[_FakeImage].raw
    assert raw == {"content": "mushroom, dark, white background", "width": 512,
                   "negative_prompt": "blurry"}
    assert path == tmp_path / "pack" / "x" / "generate.png" and path.is_file()
    assert not (tmp_path / "raw_output.png").exists()


def test_lease_is_held_for_the_whole_group(tmp_path, fake_model):
    runner = ImageGenerateRunner()
    with (
        runner.open({"model_name": fake_model}, threading.Event()),
        pytest.raises(GeneratorBusyError),
        GeneratorFactory.acquire(FactoryType.Image, fake_model),
    ):
        pass
    with GeneratorFactory.acquire(FactoryType.Image, fake_model):
        pass


def test_generator_returning_none_becomes_an_error(tmp_path, fake_model):
    _FakeImage.fail = True
    runner = ImageGenerateRunner()
    params = {"model_name": fake_model}
    with (
        runner.open(params, threading.Event()),
        pytest.raises(RuntimeError, match="没有生成文件"),
    ):
        runner.run(_ctx(tmp_path, params))


def test_input_step_passes_upstream_output(tmp_path, fake_model):
    runner = RemoveBackgroundRunner()
    params = {"model_name": fake_model}
    source = tmp_path / "source.png"
    Image.new("RGBA", (4, 4)).save(source)
    with runner.open(params, threading.Event()):
        [path] = runner.run(_ctx(tmp_path, params, inputs=[source], step_id="remove_bg"))
        assert SingletonMeta._instances[_FakeImage].raw == {"input_path": str(source)}
    assert path.name == "remove_bg.png"


def test_trim_crops_to_alpha_with_padding(tmp_path):
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    image.paste((255, 255, 255, 255), (40, 30, 60, 50))
    source = tmp_path / "in.png"
    image.save(source)
    [path] = TrimRunner().run(_ctx(tmp_path, {"padding": 2}, inputs=[source], step_id="trim"))
    with Image.open(path) as result:
        assert result.size == (24, 24)


def test_trim_rejects_fully_transparent_image():
    with pytest.raises(ValueError, match="完全透明"):
        trim_transparent(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))


def test_builtin_runners_cover_the_scene_props_template(tmp_path):
    # 构造时会校验模板里每种步骤类型都有 runner
    CollectionExecutor(tmp_path, load_template("scene_props"))
