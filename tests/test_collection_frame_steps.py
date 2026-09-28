"""序列帧步骤 runner 契约（ADR 0006）：循环检测、帧目录、精灵图产物、图生视频的参数与白底参考图。

用假的生成器和合成帧，无权重、无 GPU 也可跑。
"""
import json
import threading

import numpy as np
import pytest
from PIL import Image

from src.backend.core.collection.frame_steps import (
    AnimationGenerateRunner,
    FramesLoopRunner,
    FramesRemoveBgRunner,
    SpriteSheetRunner,
    find_loop,
    white_canvas,
)
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template, parse_template
from src.backend.core.model_base import (
    BaseAnimationGenerator,
    BaseImageGenerator,
    GeneratorFactory,
    SingletonMeta,
)
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem


def _ctx(tmp_path, params, inputs=(), step_id="loop", prompt="walk to the right"):
    out_dir = tmp_path / "pack" / "walk"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="walk", prompt=prompt), prompt=f"{prompt}, pixel art",
        negative_prompt="", inputs=list(inputs), params=params, out_dir=out_dir,
        step_id=step_id, values={"prompt": prompt},
    )


def _cyclic_frames(tmp_path, startup=5, period=6, cycles=4):
    """开头 startup 帧是各不相同的"起步"，之后按 period 帧严格循环。"""
    paths = []
    shades = [(i * 37) % 200 + 20 for i in range(period)]
    for index in range(startup + period * cycles):
        image = Image.new("RGB", (32, 32), (255, 255, 255))
        if index < startup:
            shade, x = 240 - index * 3, 2
        else:
            shade, x = shades[(index - startup) % period], 4 + (index - startup) % period * 3
        image.paste((shade, shade, shade), (x, 8, x + 8, 24))
        path = tmp_path / f"f{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


def _signatures(values):
    return np.stack([np.full((4, 4), v, dtype=np.float32) for v in values])


def test_find_loop_skips_startup_and_finds_exact_period():
    # 0..3 起步，之后周期 5
    values = [0.9, 0.8, 0.7, 0.6] + [0.1, 0.2, 0.3, 0.4, 0.5] * 4
    loop = find_loop(_signatures(values), skip=4, min_len=4)
    assert loop.start >= 4 and (loop.end - loop.start) % 5 == 0
    assert loop.error == pytest.approx(0) and loop.ratio == pytest.approx(0)


def test_find_loop_aligns_length_to_stride():
    values = [0.1, 0.2, 0.3] * 8
    loop = find_loop(_signatures(values), skip=0, min_len=3, stride=2)
    assert (loop.end - loop.start) % 2 == 0 and (loop.end - loop.start) % 3 == 0


def test_find_loop_rejects_too_few_frames():
    with pytest.raises(ValueError, match="不够"):
        find_loop(_signatures([0.1] * 10), skip=8, min_len=8)


def test_loop_runner_outputs_numbered_frames_and_seam_meta(tmp_path):
    frames = _cyclic_frames(tmp_path)
    ctx = _ctx(tmp_path, {"skip": 5, "min_len": 4}, frames)
    outputs = FramesLoopRunner().run(ctx)
    assert [p.name for p in outputs[:2]] == ["0000.png", "0001.png"]
    assert all(p.parent == ctx.out_dir / "loop" for p in outputs)
    assert len(outputs) % 6 == 0
    assert ctx.meta["start"] >= 5 and ctx.meta["needs_check"] is False


def test_loop_runner_clears_stale_frames_on_rerun(tmp_path):
    frames = _cyclic_frames(tmp_path)
    stale = tmp_path / "pack" / "walk" / "loop" / "9999.png"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    FramesLoopRunner().run(_ctx(tmp_path, {"skip": 5, "min_len": 4}, frames))
    assert not stale.exists()


def test_pingpong_plays_forward_then_back_without_repeating_ends(tmp_path):
    frames = _cyclic_frames(tmp_path, startup=2, period=3, cycles=1)  # 5 帧
    ctx = _ctx(tmp_path, {"mode": "pingpong", "skip": 2}, frames)
    outputs = FramesLoopRunner().run(ctx)
    # 正放 2,3,4 + 倒放 3（两端不重复）
    assert len(outputs) == 4
    assert outputs[3].read_bytes() == frames[3].read_bytes()


def test_loop_runner_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        FramesLoopRunner().run(_ctx(tmp_path, {"mode": "bounce"}, _cyclic_frames(tmp_path)))


def test_sprite_sheet_puts_png_first_and_names_frames_by_item(tmp_path):
    frames = []
    for index in range(4):
        image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        image.paste((200, 50, 50, 255), (10 + index, 10, 20 + index, 30))
        path = tmp_path / f"n{index}.png"
        image.save(path)
        frames.append(path)
    ctx = _ctx(tmp_path, {"fps": 10}, frames, step_id="sheet")
    sheet, atlas, gif = SpriteSheetRunner().run(ctx)
    assert sheet.suffix == ".png" and atlas.suffix == ".json" and gif.suffix == ".gif"
    data = json.loads(atlas.read_text(encoding="utf-8"))
    assert data["fps"] == 10 and data["frames"][0]["name"] == "walk_0000"
    # 按并集包围盒裁边：宽 = 10 + 3 帧位移
    assert data["frame_size"] == {"w": 13, "h": 20}
    with Image.open(gif) as preview:
        assert preview.n_frames == 4


def test_white_canvas_centers_portrait_with_margin():
    portrait = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    square = white_canvas(portrait, 0.8)
    assert square.size == (100, 100) and square.mode == "RGB"
    assert square.getpixel((0, 0)) == (255, 255, 255)
    assert square.getpixel((50, 50)) == (255, 0, 0)


def test_white_canvas_fits_portrait_into_requested_size():
    portrait = Image.new("RGBA", (40, 80), (255, 0, 0, 255))
    canvas = white_canvas(portrait, 0.8, (60, 100))
    assert canvas.size == (60, 100)
    # 竖长的角色按高度撑满 80%，上下各留 10%
    assert canvas.getpixel((30, 8)) == (255, 255, 255)
    assert canvas.getpixel((30, 12)) == (255, 0, 0)
    assert canvas.getpixel((30, 50)) == (255, 0, 0)


class _FakeAnimation(BaseAnimationGenerator):
    output_dir = None

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        self.raw = raw
        with Image.open(raw["reference_image_path"]) as reference:
            self.reference_size = reference.size

    async def generate(self):
        return None

    async def generate_animation(self):
        paths = []
        for index in range(3):
            path = type(self).output_dir / f"raw_{index}.png"
            Image.new("RGB", (16, 16), (index * 50, 0, 0)).save(path)
            paths.append(path)
        gif = type(self).output_dir / "raw.gif"
        gif.write_bytes(b"gif")
        return paths, gif


class _FakeBgRemoval(BaseImageGenerator):
    output_dir = None

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        self.input_path = raw["input_path"]

    async def generate(self):
        path = type(self).output_dir / "nobg.png"
        with Image.open(self.input_path) as image:
            image.convert("RGBA").save(path)
        return path


@pytest.fixture
def fake_models(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _FakeAnimation.output_dir = raw
    _FakeBgRemoval.output_dir = raw
    GeneratorFactory.register_generator(FactoryType.Animation, "fake-video", _FakeAnimation)
    GeneratorFactory.register_generator(FactoryType.Image, "fake-nobg", _FakeBgRemoval)
    yield
    GeneratorFactory._generators[FactoryType.Animation].pop("fake-video", None)
    GeneratorFactory._generators[FactoryType.Image].pop("fake-nobg", None)
    SingletonMeta._instances.pop(_FakeAnimation, None)
    SingletonMeta._instances.pop(_FakeBgRemoval, None)
    GeneratorFactory._holder = None


def test_animation_runner_uses_action_prompt_and_white_canvas_reference(tmp_path, fake_models):
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (30, 60), (0, 0, 255, 255)).save(portrait)
    params = {"model_name": "fake-video", "num_frames": 49, "scale": 0.5,
              "prompt_suffix": "as if on a treadmill"}
    runner = AnimationGenerateRunner()
    ctx = _ctx(tmp_path, params, [portrait], step_id="video")
    with runner.open(params, threading.Event()):
        outputs = runner.run(ctx)
        generator = SingletonMeta._instances[_FakeAnimation]
    # 只用条目原文 + 后缀，不拼风格锁
    assert generator.raw["content"] == "walk to the right, as if on a treadmill"
    assert generator.raw["num_frames"] == 49 and "scale" not in generator.raw
    assert generator.reference_size == (120, 120)
    assert [p.relative_to(ctx.out_dir).as_posix() for p in outputs] == [
        "video/0000.png", "video/0001.png", "video/0002.png"]
    # 临时参考图和生成器自带的 GIF 都不留下
    assert not (ctx.out_dir / "video_reference.png").exists()
    assert not (tmp_path / "raw" / "raw.gif").exists()
    assert ctx.meta["frames"] == 3


def test_animation_runner_builds_reference_at_requested_size(tmp_path, fake_models):
    portrait = tmp_path / "portrait.png"
    Image.new("RGBA", (30, 60), (0, 0, 255, 255)).save(portrait)
    params = {"model_name": "fake-video", "width": 544, "height": 960, "scale": 0.9}
    runner = AnimationGenerateRunner()
    with runner.open(params, threading.Event()):
        runner.run(_ctx(tmp_path, params, [portrait], step_id="video"))
        generator = SingletonMeta._instances[_FakeAnimation]
    # 宽高照传给视频模型，参考图就是这个尺寸，模型不会再补黑边
    assert (generator.raw["width"], generator.raw["height"]) == (544, 960)
    assert generator.reference_size == (544, 960)


def test_remove_bg_runner_processes_every_frame_in_order(tmp_path, fake_models):
    frames = _cyclic_frames(tmp_path, startup=0, period=3, cycles=1)
    runner = FramesRemoveBgRunner()
    params = {"model_name": "fake-nobg"}
    ctx = _ctx(tmp_path, params, frames, step_id="remove_bg")
    with runner.open(params, threading.Event()):
        outputs = runner.run(ctx)
    assert [p.name for p in outputs] == ["0000.png", "0001.png", "0002.png"]
    for source, output in zip(frames, outputs, strict=True):
        with Image.open(source) as a, Image.open(output) as b:
            assert b.mode == "RGBA" and a.convert("RGBA").tobytes() == b.tobytes()


def test_character_motion_template_binds_source_and_uses_frame_runners():
    motion = load_template("character_motion")
    assert motion.source == "character" and motion.cover == "sheet"
    assert motion.step("video").inputs == ("@source",)
    runners = registered_runners()
    assert all(step.type in runners for step in motion.steps)


def test_source_input_requires_template_source():
    with pytest.raises(ValueError, match="source"):
        parse_template({"id": "bad", "steps": [
            {"id": "a", "type": "x", "inputs": ["@source"]}]})
    with pytest.raises(ValueError, match="来源类型"):
        parse_template({"id": "bad", "source": "monster", "steps": [{"id": "a", "type": "x"}]})
