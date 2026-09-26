"""图片类步骤 runner 契约（ADR 0004）：租约范围、参数拼装、产物落点、失败转异常。

用假的生成器，无权重、无 GPU 也可跑。
"""
import threading

import pytest
from PIL import Image

from src.backend.core.collection.executor import CollectionExecutor
from src.backend.core.collection.image_steps import (
    ColorMatchRunner,
    CompositeRunner,
    ImageGenerateRunner,
    RemoveBackgroundRunner,
    TileHorizontalRunner,
    TrimRunner,
    tile_horizontal,
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


def _gradient(width: int, height: int = 4) -> Image.Image:
    image = Image.new("RGBA", (width, height))
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (x * 255 // (width - 1), 0, 0, 255))
    return image


def test_tile_x_wraps_right_edge_into_left_edge(tmp_path):
    source = tmp_path / "in.png"
    _gradient(80).save(source)
    [path] = TileHorizontalRunner().run(
        _ctx(tmp_path, {"overlap": 0.25}, inputs=[source], step_id="far"))
    with Image.open(path) as result:
        red = [result.getpixel((x, 0))[0] for x in range(result.width)]
    assert result.size == (60, 4)
    # 平铺时最右一列接着最左一列：左端从原图右端开始过渡，接缝两侧颜色相近
    assert abs(red[0] - red[-1]) <= 5
    # 过渡带结束后回到原图 overlap 处
    assert abs(red[20] - 20 * 255 // 79) <= 1


def test_tile_x_does_not_bleed_color_from_transparent_pixels():
    image = Image.new("RGBA", (40, 2), (0, 255, 0, 0))
    image.paste((255, 0, 0, 255), (30, 0, 40, 2))
    tiled = tile_horizontal(image, 10)
    red, green, _, alpha = tiled.getpixel((5, 0))
    assert 0 < alpha < 255 and green == 0 and red == 255


def test_tile_x_rejects_bad_overlap(tmp_path):
    source = tmp_path / "in.png"
    _gradient(10).save(source)
    with pytest.raises(ValueError, match="overlap"):
        TileHorizontalRunner().run(_ctx(tmp_path, {"overlap": 0.6}, inputs=[source]))


def test_composite_stacks_layers_in_input_order(tmp_path):
    far, near = tmp_path / "far.png", tmp_path / "near.png"
    Image.new("RGBA", (8, 8), (0, 0, 255, 255)).save(far)
    top = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    top.paste((255, 0, 0, 255), (0, 4, 8, 8))
    top.save(near)
    [path] = CompositeRunner().run(_ctx(tmp_path, {}, inputs=[far, near], step_id="preview"))
    with Image.open(path) as result:
        assert result.size == (8, 8)
        assert result.getpixel((0, 0)) == (0, 0, 255, 255)
        assert result.getpixel((0, 7)) == (255, 0, 0, 255)


def _two_tone(first, second, alpha=255):
    image = Image.new("RGBA", (8, 8), (*first, alpha))
    image.paste((*second, alpha), (0, 4, 8, 8))
    return image


def test_color_match_shifts_hue_toward_reference_and_keeps_alpha(tmp_path):
    target, reference = tmp_path / "mid.png", tmp_path / "far.png"
    layer = _two_tone((200, 40, 40), (120, 20, 20))
    layer.putpixel((0, 0), (0, 255, 0, 0))
    layer.save(target)
    _two_tone((40, 60, 200), (80, 100, 230)).save(reference)
    [path] = ColorMatchRunner().run(
        _ctx(tmp_path, {"strength": 1.0}, inputs=[target, reference], step_id="mid_color"))
    with Image.open(path) as result:
        red, _, blue, alpha = result.getpixel((0, 7))
        assert blue > red
        assert result.getpixel((0, 0))[3] == 0 and alpha == 255


def test_color_match_with_zero_strength_is_identity(tmp_path):
    target, reference = tmp_path / "mid.png", tmp_path / "far.png"
    _two_tone((200, 40, 40), (120, 20, 20)).save(target)
    _two_tone((40, 60, 200), (80, 100, 230)).save(reference)
    [path] = ColorMatchRunner().run(
        _ctx(tmp_path, {"strength": 0.0}, inputs=[target, reference], step_id="mid_color"))
    with Image.open(path) as result:
        # 只经过一次 LAB 往返，允许少量量化误差
        assert all(abs(a - b) <= 3 for a, b in zip(result.getpixel((0, 7)), (120, 20, 20, 255)))


def test_color_match_requires_target_and_reference(tmp_path):
    source = tmp_path / "in.png"
    _two_tone((0, 0, 0), (255, 255, 255)).save(source)
    with pytest.raises(ValueError, match="两张输入"):
        ColorMatchRunner().run(_ctx(tmp_path, {}, inputs=[source]))


def test_builtin_runners_cover_the_scene_parallax_template(tmp_path):
    template = load_template("scene_parallax")
    CollectionExecutor(tmp_path, template)
    # 同模型的步骤相邻：执行器按步骤分组，相邻才不会来回换模型
    models = [step.params.get("model_name") for step in template.steps]
    runs = [name for index, name in enumerate(models)
            if name and (index == 0 or models[index - 1] != name)]
    assert len(runs) == len(set(runs))
    assert [s.id for s in template.steps if s.deliverable] == ["far", "mid", "near", "preview"]
    # 中近景都要以远景为参考对齐色调，否则各层色温不一，叠起来穿帮
    assert template.step("mid_color").inputs == ("mid_remove_bg", "far_generate")
    assert template.step("near_color").inputs == ("near_remove_bg", "far_generate")
