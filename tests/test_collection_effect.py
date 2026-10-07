"""特效包（技能）契约：生成 / 导入图片二选一的两条链、导入图片的校验与复制、
动画精灵图的缩放抽帧、补边底色。纯 CPU，无权重、无 GPU 也可跑。
"""
import pytest
from PIL import Image

from src.backend.core.collection import library
from src.backend.core.collection.executor import CollectionExecutor
from src.backend.core.collection.frame_steps import (
    FramesCrossfadeRunner,
    SpriteSheetRunner,
    canvas_background,
    white_canvas,
)
from src.backend.core.collection.image_steps import ImportRunner
from src.backend.core.collection.steps import StepContext
from src.backend.core.collection.template import load_template, parse_template
from src.shared.schemas import CollectionItem, NewCollectionItem

IMPORTED = "01/reference.png"


def _active(fields):
    tpl = load_template("effect_skill")
    return {s.id for s in tpl.steps if tpl.step_active(s.id, fields)}


def _png(path, size=(64, 48), color=(10, 20, 60)):
    Image.new("RGB", size, color).save(path)
    return path


def _ctx(tmp_path, params, inputs=(), step_id="sheet", values=None):
    out_dir = tmp_path / "pack" / "01"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="01", prompt="fire"), prompt="fire", negative_prompt="",
        inputs=list(inputs), params=params, out_dir=out_dir, step_id=step_id,
        values=values or {"prompt": "fire"},
    )


def test_effect_skill_generates_icon_unless_an_image_is_imported():
    generated = _active({"sfx": "whoosh"})
    imported = _active({"sfx": "whoosh", "reference": IMPORTED})
    assert generated == {"icon", "upscale", "resize", "vfx", "video", "loop", "sheet", "sfx"}
    assert imported == {"import", "import_video", "import_loop", "import_sheet", "sfx"}


def test_effect_skill_animates_a_separate_vfx_image_freely():
    tpl = load_template("effect_skill")
    # 动画从黑底特效图出发，而不是从带边框的图标：图标一动就溶解变形
    assert tpl.step("video").inputs == ("vfx",)
    assert "black background" in tpl.step("vfx").params["prompt_suffix"]
    for video, loop in (("video", "loop"), ("import_video", "import_loop")):
        params = tpl.step(video).params
        # 末帧锚定会把火焰压成几乎静止（实测帧差只有不锚定的 1/4），特效不锚定
        assert not params.get("loop")
        # "keep / stay" 这类措辞同样会压住动作
        assert "keep" not in params["prompt_suffix"] and "stay" not in params["prompt_suffix"]
        assert tpl.step(loop).type == "frames.crossfade" and tpl.step(loop).inputs == (video,)


def test_pack_video_and_audio_steps_pick_a_fresh_seed():
    from src.backend.core.collection.audio_steps import SpeechGenerateRunner
    from src.backend.core.collection.frame_steps import AnimationGenerateRunner
    seeds = set()
    for _ in range(3):
        ctx = StepContext(item=CollectionItem(id="01"), prompt="", negative_prompt="", inputs=[],
                          params={"model_name": "m", "content": "boom"}, out_dir=None,
                          step_id="x", values={"prompt": "fire"})
        seeds.add(SpeechGenerateRunner().build_params(ctx)["seed"])
        seeds.add(AnimationGenerateRunner().build_params(ctx)["seed"])
    # 默认种子固定的话重新生成得到同一段，用户没法换一个更好的结果
    assert len(seeds) > 1
    ctx.params["seed"] = 5
    assert SpeechGenerateRunner().build_params(ctx)["seed"] == 5


def test_effect_skill_builds_an_executor_with_every_step_registered(tmp_path):
    # 构造时会校验模板里每种步骤类型都有 runner
    CollectionExecutor(tmp_path, load_template("effect_skill"))


def test_effect_skill_audio_uses_enough_steps():
    sfx = load_template("effect_skill").step("sfx")
    assert sfx.params["num_inference_steps"] >= 50


def test_template_rejects_unknown_field_kind():
    with pytest.raises(ValueError, match="类型不支持"):
        parse_template({"id": "t", "type": "effect",
                        "fields": [{"id": "pic", "kind": "video"}],
                        "steps": [{"id": "a", "type": "image.generate"}]})


def test_import_runner_converts_the_imported_file_to_png(tmp_path):
    ctx = _ctx(tmp_path, {"field": "reference"}, step_id="import",
               values={"prompt": "fire", "reference": "01/reference.jpg"})
    _png(ctx.out_dir / "reference.jpg")
    (path,) = ImportRunner().run(ctx)
    assert path == ctx.out_dir / "import.png"
    with Image.open(path) as image:
        assert image.mode == "RGBA" and image.size == (64, 48)
    assert ctx.meta["size"] == [64, 48]


def test_import_runner_needs_a_field_with_an_image(tmp_path):
    with pytest.raises(ValueError, match="field"):
        ImportRunner().run(_ctx(tmp_path, {"field": "reference"}, step_id="import"))
    ctx = _ctx(tmp_path, {"field": "reference"}, step_id="import",
               values={"prompt": "fire", "reference": IMPORTED})
    with pytest.raises(FileNotFoundError):
        ImportRunner().run(ctx)


def test_new_items_rejects_missing_or_broken_image(tmp_path):
    template = load_template("effect_skill")
    missing = NewCollectionItem(prompt="fire", fields={"sfx": "x", "reference": str(tmp_path / "no.png")})
    with pytest.raises(ValueError, match="找不到文件"):
        library._new_items(template, [missing], [])
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    bad = NewCollectionItem(prompt="fire", fields={"sfx": "x", "reference": str(broken)})
    with pytest.raises(ValueError, match="不是可用的图片"):
        library._new_items(template, [bad], [])


def test_import_images_copies_into_the_item_dir_and_stores_a_relative_path(tmp_path):
    template = load_template("effect_skill")
    source = _png(tmp_path / "mine.PNG")
    new_item = NewCollectionItem(prompt="fire", fields={"sfx": "x", "reference": str(source)})
    items = library._new_items(template, [new_item], [])
    pack = tmp_path / "pack"
    pack.mkdir()
    library._import_images(template, items, pack)
    assert items[0].fields["reference"] == "01/reference.png"
    assert (pack / "01" / "reference.png").is_file()
    # 条目不再依赖原文件：删掉原图后资源包照样完整
    source.unlink()
    assert (pack / "01" / "reference.png").is_file()


def test_sprite_sheet_strides_and_shrinks_frames(tmp_path):
    frames = []
    for index in range(8):
        path = tmp_path / f"f{index}.png"
        # 每帧颜色不同：相同的相邻帧存 GIF 时会被合并，帧数就对不上了
        Image.new("RGBA", (200, 100), (200, 20 * index, 50, 255)).save(path)
        frames.append(path)
    ctx = _ctx(tmp_path, {"fps": 16, "size": 50, "stride": 2}, frames)
    sheet, atlas, gif = SpriteSheetRunner().run(ctx)
    assert ctx.meta["frames"] == 4 and ctx.meta["frame_size"] == [50, 25]
    with Image.open(gif) as preview:
        assert preview.n_frames == 4
    # 缩放用的临时目录用完即清，条目目录里只留产物
    assert {p.name for p in ctx.out_dir.iterdir()} == {sheet.name, atlas.name, gif.name}


@pytest.mark.parametrize("params", [{"stride": 0}, {"size": -1}])
def test_sprite_sheet_rejects_bad_stride_or_size(tmp_path, params):
    frame = _png(tmp_path / "f.png")
    with pytest.raises(ValueError, match="参数不合法"):
        SpriteSheetRunner().run(_ctx(tmp_path, params, [frame]))


def _shade_frames(tmp_path, shades):
    paths = []
    for index, shade in enumerate(shades):
        path = tmp_path / f"c{index}.png"
        Image.new("RGBA", (8, 8), (shade, shade, shade, 255)).save(path)
        paths.append(path)
    return paths


def test_crossfade_blends_the_tail_into_the_head(tmp_path):
    # 8 帧、叠 2 帧：输出 6 帧，开头两帧分别是 1/3、2/3 的"开头"权重
    frames = _shade_frames(tmp_path, [0, 30, 60, 90, 120, 150, 180, 210])
    ctx = _ctx(tmp_path, {"frames": 2}, frames, step_id="loop")
    outputs = FramesCrossfadeRunner().run(ctx)
    shades = [Image.open(p).getpixel((0, 0))[0] for p in outputs]
    assert shades == [round(180 * 2 / 3 + 0 / 3), round(210 / 3 + 30 * 2 / 3), 60, 90, 120, 150]
    assert ctx.meta == {"frames": 6, "crossfade": 2}
    # 循环时最后一帧（150）接回开头（主要是结尾 180 的成分），比直接跳回 0 平滑得多
    assert abs(shades[0] - shades[-1]) < abs(0 - 150)


def test_crossfade_zero_frames_plays_once_unchanged(tmp_path):
    frames = _shade_frames(tmp_path, [0, 30, 60, 90])
    ctx = _ctx(tmp_path, {"frames": "0"}, frames, step_id="loop")
    outputs = FramesCrossfadeRunner().run(ctx)
    assert [Image.open(p).getpixel((0, 0))[0] for p in outputs] == [0, 30, 60, 90]


def test_effect_skill_playback_field_drives_the_crossfade():
    tpl = load_template("effect_skill")
    values = {"0", "6"}
    assert {value for value, _ in next(f for f in tpl.fields if f.id == "playback").options} == values
    # 默认单次播放：释放类特效只播一次，不叠化也就没有重影
    assert next(f for f in tpl.fields if f.id == "playback").default == "0"
    for step in ("loop", "import_loop"):
        assert tpl.step(step).params["frames"] == "{playback}"


def test_crossfade_rejects_overlap_longer_than_half(tmp_path):
    frames = _shade_frames(tmp_path, [0, 10, 20, 30])
    with pytest.raises(ValueError, match="叠化帧数"):
        FramesCrossfadeRunner().run(_ctx(tmp_path, {"frames": 2}, frames, step_id="loop"))


class _FakeWanPipe:
    """只模拟 TI2V-5B 的 prepare_latents：返回 (latents, condition, 第一帧为 0 的 mask)。"""

    def __init__(self, expand_timesteps=True):
        self.config = type("Config", (), {"expand_timesteps": expand_timesteps})()

    def prepare_latents(self):
        import torch
        mask = torch.ones(1, 1, 5, 2, 2)
        mask[:, :, 0] = 0
        return "latents", "condition", mask


def _wan(pipe):
    pytest.importorskip("torch", reason="锚定逻辑操作 torch 张量")
    from src.backend.core.animation.wan2_2 import Wan2VideoGenerator
    generator = Wan2VideoGenerator.__new__(Wan2VideoGenerator)
    generator.pipe, generator.model_name = pipe, "Wan2.2-TI2V-Turbo"
    return generator


def test_wan_anchor_pins_the_last_latent_frame_and_restores():
    pipe = _FakeWanPipe()
    generator = _wan(pipe)
    generator._anchor_last_frame()
    _latents, _condition, mask = pipe.prepare_latents()
    assert mask[0, 0, :, 0, 0].tolist() == [0, 1, 1, 1, 0]
    # 生成器是单例：还原后不循环的生成恢复原样
    del pipe.prepare_latents
    assert pipe.prepare_latents()[2][0, 0, :, 0, 0].tolist() == [0, 1, 1, 1, 1]


def test_wan_anchor_refuses_models_without_expand_timesteps():
    with pytest.raises(ValueError, match="不支持循环模式"):
        _wan(_FakeWanPipe(expand_timesteps=False))._anchor_last_frame()


def test_canvas_background_auto_follows_the_image_corner():
    dark = Image.new("RGB", (20, 20), (10, 20, 60))
    assert canvas_background(dark, "auto") == (10, 20, 60)
    assert canvas_background(dark, "white") == (255, 255, 255)
    with pytest.raises(ValueError, match="background"):
        canvas_background(dark, "black")


def test_white_canvas_pads_with_the_given_background():
    wide = Image.new("RGBA", (80, 40), (200, 0, 0, 255))
    canvas = white_canvas(wide, 1.0, (40, 40), (10, 20, 60))
    assert canvas.size == (40, 40)
    assert canvas.getpixel((20, 2)) == (10, 20, 60)
    assert canvas.getpixel((20, 20)) == (200, 0, 0)
