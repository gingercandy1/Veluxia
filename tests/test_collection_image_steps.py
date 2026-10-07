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
    SplitRunner,
    TrimRunner,
    UpscaleRunner,
    parse_size,
    split_pieces,
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


@pytest.mark.parametrize("value, expected", [
    ("768x1344", (768, 1344)), ("1024X1024", (1024, 1024)), (" 512 x 640 ", (512, 640)),
    ("512×640", (512, 640)),
])
def test_parse_size_reads_width_by_height(value, expected):
    assert parse_size(value) == expected


@pytest.mark.parametrize("value", ["", "768", "768x", "x1344", "0x100", "100x0", "a x b", "1x2x3"])
def test_parse_size_rejects_malformed_sizes(value):
    with pytest.raises(ValueError, match="画幅"):
        parse_size(value)


def test_generate_runner_turns_size_into_width_and_height(tmp_path):
    ctx = _ctx(tmp_path, {"model_name": "m", "size": "1024x1024", "prompt_suffix": "white"})
    params = ImageGenerateRunner().build_params(ctx)
    assert (params["width"], params["height"]) == (1024, 1024)
    assert "size" not in params  # 生成器只认 width / height


def test_generate_runner_picks_the_suffix_for_the_canvas(tmp_path):
    suffixes = {"768x1344": "full body character, white", "1024x1024": "white"}
    for size, expected in (("768x1344", "full body character, white"), ("1024x1024", "white"),
                           (" 1024 X 1024 ", "white")):
        ctx = _ctx(tmp_path, {"model_name": "m", "size": size, "prompt_suffix": suffixes})
        # 条目提示词 + 风格锁之后，接的正是这个画幅对应的后缀
        assert ImageGenerateRunner().build_params(ctx)["content"] == f"mushroom, dark, {expected}"
    # 竖版的后缀不能漏进方形：这正是球形角色长出手脚的原因
    square = ImageGenerateRunner().build_params(_ctx(
        tmp_path, {"model_name": "m", "size": "1024x1024", "prompt_suffix": suffixes}))
    assert "full body" not in square["content"]


def test_generate_runner_rejects_a_canvas_the_suffix_table_does_not_cover(tmp_path):
    ctx = _ctx(tmp_path, {"model_name": "m", "size": "512x512",
                          "prompt_suffix": {"768x1344": "a"}})
    with pytest.raises(ValueError, match="prompt_suffix"):
        ImageGenerateRunner().build_params(ctx)
    ctx = _ctx(tmp_path, {"model_name": "m", "prompt_suffix": {"768x1344": "a"}})
    with pytest.raises(ValueError, match="prompt_suffix"):
        ImageGenerateRunner().build_params(ctx)  # 没给画幅也不能默默选一个


def test_generate_runner_without_size_keeps_explicit_width_and_height(tmp_path):
    ctx = _ctx(tmp_path, {"model_name": "m", "width": 512, "height": 640})
    params = ImageGenerateRunner().build_params(ctx)
    assert (params["width"], params["height"]) == (512, 640)


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


def test_generate_prompt_param_replaces_item_prompt_but_keeps_style(tmp_path, fake_model):
    runner = ImageGenerateRunner()
    params = {"model_name": fake_model, "prompt": "mossy stone ground", "prompt_suffix": "side view"}
    ctx = _ctx(tmp_path, params)
    ctx.style_prompt = "dark"
    with runner.open(params, threading.Event()):
        runner.run(ctx)
        raw = SingletonMeta._instances[_FakeImage].raw
    assert raw["content"] == "mossy stone ground, dark, side view"
    assert "prompt" not in raw


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


def test_input_step_processes_each_upstream_image_in_order(tmp_path, fake_model):
    # 分层产出 N 个图层，放大步骤逐张处理，产物按图层顺序编号，不能互相覆盖
    runner = UpscaleRunner()
    params = {"model_name": fake_model, "outscale": 2}
    sources = []
    for index in range(1, 4):
        source = tmp_path / f"layer_{index}.png"
        Image.new("RGBA", (4, 4)).save(source)
        sources.append(source)
    seen = []
    original = _FakeImage.parse_params
    _FakeImage.parse_params = lambda self, raw: (seen.append(raw["input_path"]),
                                                 original(self, raw))
    try:
        with runner.open(params, threading.Event()):
            ctx = _ctx(tmp_path, params, inputs=sources, step_id="layers")
            paths = runner.run(ctx)
    finally:
        _FakeImage.parse_params = original
    assert seen == [str(p) for p in sources]
    assert [p.name for p in paths] == ["layers_1.png", "layers_2.png", "layers_3.png"]
    assert all(p.is_file() for p in paths)
    assert ctx.meta["params"] == {"outscale": 2} and ctx.meta["size"] == [8, 8]


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


def _sheet(*boxes, size=(200, 200)):
    image = Image.new("RGBA", size, (255, 255, 255, 0))
    for box in boxes:
        image.paste((200, 50, 50, 255), box)
    return image


def test_split_cuts_pieces_in_reading_order(tmp_path):
    # 同一行里高矮不一：高的顶边更靠上，但仍要排在左边那个矮的后面
    sheet = _sheet((10, 30, 40, 60), (80, 10, 110, 60), (20, 120, 60, 160))
    source = tmp_path / "sheet.png"
    sheet.save(source)
    ctx = _ctx(tmp_path, {"padding": 2, "min_area": 0.002}, inputs=[source], step_id="split")
    paths = SplitRunner().run(ctx)
    assert [p.name for p in paths] == ["split_1.png", "split_2.png", "split_3.png"]
    sizes = []
    for path in paths:
        with Image.open(path) as piece:
            sizes.append(piece.size)
    assert sizes == [(34, 34), (34, 54), (44, 44)]
    assert ctx.meta["pieces"] == 3


def test_split_clears_neighbour_pixels_inside_the_box():
    # L 形元素的包围框会罩住旁边的小方块，小方块的像素不能出现在 L 的切片里
    sheet = _sheet((10, 10, 20, 100), (10, 90, 100, 100), (40, 40, 70, 70))
    pieces = split_pieces(sheet, min_area=0.002, padding=0)
    assert len(pieces) == 2
    l_shape = pieces[0]
    assert l_shape.size == (90, 90)
    assert l_shape.getpixel((45, 45))[3] == 0


def test_split_drops_specks_and_keeps_small_gaps_together():
    # 2 像素的缝不切开（块内连通），零星噪点丢掉
    sheet = _sheet((10, 10, 50, 30), (10, 32, 50, 60), (150, 150, 152, 152))
    pieces = split_pieces(sheet, min_area=0.002, padding=0)
    assert [p.size for p in pieces] == [(40, 50)]


def test_split_ignores_faint_background_haze():
    sheet = Image.new("RGBA", (100, 100), (255, 255, 255, 10))
    sheet.paste((200, 50, 50, 255), (10, 10, 30, 30))
    sheet.paste((200, 50, 50, 255), (60, 60, 90, 90))
    assert len(split_pieces(sheet, min_area=0.002, padding=0)) == 2


def test_split_rejects_sheet_without_pieces():
    with pytest.raises(ValueError, match="完全透明"):
        split_pieces(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
    with pytest.raises(ValueError, match="没有足够大的元素"):
        split_pieces(_sheet((0, 0, 2, 2)), min_area=0.01)


def test_scene_props_picks_single_or_sheet_chain():
    tpl = load_template("scene_props")
    single = {s.id for s in tpl.steps if tpl.step_active(s.id, {})}
    sheet = {s.id for s in tpl.steps if tpl.step_active(s.id, {"sheet": "6"})}
    assert single == {"generate", "remove_bg", "upscale", "trim"}
    assert sheet == {"sheet_variants", "sheet_generate", "sheet_remove_bg", "sheet_split",
                     "sheet_upscale"}


def test_builtin_runners_cover_the_scene_props_template(tmp_path):
    # 构造时会校验模板里每种步骤类型都有 runner
    CollectionExecutor(tmp_path, load_template("scene_props"))
