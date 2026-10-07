"""模板预设条目、按字段命名条目、单次播放、转面总览（角色立绘 / 角色动作）。

无权重、无 GPU 也可跑。
"""
import json

import pytest
from PIL import Image

from src.backend.core.collection import library
from src.backend.core.collection.frame_steps import FramesLoopRunner, SpriteSheetRunner
from src.backend.core.collection.image_steps import MontageRunner
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template, parse_template
from src.backend.router.library import LibraryRouter
from src.shared.schemas import CollectionItem, NewCollectionItem


def _ctx(tmp_path, params, inputs=(), step_id="step", item_id="walk"):
    out_dir = tmp_path / "pack" / item_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id=item_id, prompt="p"), prompt="p", negative_prompt="",
        inputs=list(inputs), params=params, out_dir=out_dir, step_id=step_id,
        values={"prompt": "p"},
    )


def _frames(tmp_path, count, size=(8, 8)):
    folder = tmp_path / "frames"
    folder.mkdir(exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"{index:04d}.png"
        Image.new("RGBA", size, (index * 10 % 255, 0, 0, 255)).save(path)
        paths.append(path)
    return paths


def _spec(**extra):
    return {"id": "t", "fields": [
        {"id": "action"},
        {"id": "mode", "default": "a", "options": [{"value": "a"}, {"value": "b"}]},
    ], "steps": [{"id": "s", "type": "x"}], **extra}


# ---- 模板：预设与 id_field ----
def test_parse_template_reads_presets_and_id_field():
    parsed = parse_template(_spec(
        id_field="action",
        presets=[{"name": "待机", "prompt": "stands", "fields": {"action": "idle", "mode": "b"}}]))
    assert parsed.id_field == "action"
    [preset] = parsed.presets
    assert (preset.name, preset.prompt, preset.fields) == (
        "待机", "stands", {"action": "idle", "mode": "b"})


@pytest.mark.parametrize("presets, message", [
    ([{"name": "x"}], "缺少 name 或 prompt"),
    ([{"prompt": "x"}], "缺少 name 或 prompt"),
    ([{"name": "x", "prompt": "p", "fields": {"nope": "1"}}], "未声明的字段"),
    ([{"name": "x", "prompt": "p", "fields": {"mode": "zzz"}}], "不在可选范围内"),
])
def test_parse_template_rejects_bad_presets(presets, message):
    with pytest.raises(ValueError, match=message):
        parse_template(_spec(presets=presets))


def test_parse_template_rejects_unknown_id_field():
    with pytest.raises(ValueError, match="id_field"):
        parse_template(_spec(id_field="nope"))


def test_templates_without_presets_are_unchanged():
    parsed = parse_template(_spec())
    assert parsed.presets == () and parsed.id_field == ""


def test_router_exposes_presets_to_the_frontend():
    info = LibraryRouter._template_info(load_template("character_motion"))
    assert len(info.presets) >= 10
    first = info.presets[0]
    assert first.name and first.prompt and first.fields["action"]
    assert LibraryRouter._template_info(load_template("character_basic")).presets == []


# ---- 角色动作模板 ----
def test_motion_presets_are_valid_items_covering_the_common_actions():
    motion = load_template("character_motion")
    assert len(motion.presets) >= 10
    actions = [p.fields["action"] for p in motion.presets]
    for needed in ("idle", "walk", "run", "jump", "attack", "hurt", "death"):
        assert needed in actions
    assert len(set(actions)) == len(actions)
    assert all(library._ID_PATTERN.fullmatch(a) for a in actions)
    for preset in motion.presets:
        motion.check_item_fields(dict(preset.fields))  # 每个预设本身都是能建的条目
        # 14B 按字面理解：写了 treadmill 就真的画出一台跑步机
        assert "treadmill" not in preset.prompt.lower()
        assert not any(part in preset.prompt.lower() for part in (
            "arms", "legs", "hands", "feet", "knees", "shoulders",
        ))


def test_motion_presets_support_round_characters_without_inventing_limbs():
    motion = load_template("character_motion")
    by_action = {p.fields["action"]: p.prompt.lower() for p in motion.presets}
    for action in ("walk", "run", "dash"):
        assert "roll" in by_action[action]
    assert "hop" in by_action["walk"]
    suffix = motion.step("video").params["prompt_suffix"].lower()
    assert "preserve the exact character design" in suffix
    assert "never invent arms, legs, hands, or feet" in suffix


def test_motion_presets_loop_only_where_the_action_repeats():
    by_action = {p.fields["action"]: p.fields["loop_mode"] for p in load_template(
        "character_motion").presets}
    for repeats in ("walk", "run", "jump"):
        assert by_action[repeats] == "loop"
    assert by_action["idle"] == "pingpong"
    for once in ("attack", "hurt", "death"):
        assert by_action[once] == "once"


def test_motion_loop_mode_is_a_per_item_column_not_a_pack_default():
    from src.app.ui.library.form.defaults_bar import pack_level
    motion = LibraryRouter._template_info(load_template("character_motion"))
    loop_mode = next(f for f in motion.fields if f.id == "loop_mode")
    assert loop_mode.required and not pack_level(loop_mode)
    assert [o.value for o in loop_mode.options] == ["loop", "pingpong", "once"]


def test_motion_video_prompt_does_not_force_a_cycle():
    step = load_template("character_motion").step("video")
    suffix = step.params["prompt_suffix"]
    assert "fixed locked camera" in suffix and "safety margins" in suffix
    assert step.params["width"] == 384 and step.params["height"] == 672
    assert step.params["scale"] == 0.65
    assert step.params["loop_mode"] == "{loop_mode}"
    assert load_template("character_motion").step("sheet").params["mode"] == "{loop_mode}"


# ---- 条目 id 取自字段 ----
@pytest.fixture
def motion(tmp_path, monkeypatch):
    return load_template("character_motion")


def _new(motion, *fields_list, ids=()):
    items = [NewCollectionItem(id=i, prompt=f"p{n}", fields=f)
             for n, (f, i) in enumerate(zip(fields_list, list(ids) + [""] * len(fields_list)))]
    return library._new_items(motion, items, [])


def test_items_are_named_after_their_action(motion):
    items = _new(motion, {"action": "idle", "loop_mode": "pingpong"},
                 {"action": "walk", "loop_mode": "loop"}, {"loop_mode": "loop"})
    assert [i.id for i in items] == ["idle", "walk", "03"]


def test_action_name_falls_back_to_a_number_when_unusable(motion):
    items = _new(motion, {"action": "待机", "loop_mode": "loop"},       # 不是合法 id
                 {"action": "walk", "loop_mode": "loop"},
                 {"action": "walk", "loop_mode": "loop"},                # 重名
                 {"action": "../x", "loop_mode": "loop"})
    assert [i.id for i in items] == ["01", "walk", "03", "04"]


def test_explicit_id_wins_and_blocks_the_same_action_name(motion):
    items = _new(motion, {"action": "run", "loop_mode": "loop"},
                 {"action": "run", "loop_mode": "loop"}, ids=("run", ""))
    assert items[0].id == "run" and items[1].id == "02"


def test_action_name_does_not_clash_with_existing_items(motion):
    existing = [CollectionItem(id="walk", prompt="old")]
    [added] = library._new_items(
        motion, [NewCollectionItem(prompt="again", fields={"action": "walk", "loop_mode": "loop"})],
        existing)
    assert added.id == "02"


def test_templates_without_id_field_keep_numbering(tmp_path, monkeypatch):
    scene = parse_template(_spec())
    items = library._new_items(scene, [NewCollectionItem(prompt="a", fields={"action": "walk"})], [])
    assert items[0].id == "01"


# ---- frames.loop 单次播放 / atlas loop ----
def test_loop_once_outputs_the_whole_clip_from_the_first_frame(tmp_path):
    frames = _frames(tmp_path, 20)
    ctx = _ctx(tmp_path, {"mode": "once", "skip": 8}, frames, "loop")
    outputs = FramesLoopRunner().run(ctx)
    assert len(outputs) == 20  # 不像循环那样跳过开头：单次动作本来就从立绘姿势起步
    assert ctx.meta["mode"] == "once" and ctx.meta["needs_check"] is False


def test_loop_once_honors_stride_and_once_skip(tmp_path):
    ctx = _ctx(tmp_path, {"mode": "once", "stride": 2, "once_skip": 4}, _frames(tmp_path, 20))
    assert len(FramesLoopRunner().run(ctx)) == 8


def test_loop_once_rejects_too_short_clips(tmp_path):
    with pytest.raises(ValueError, match="单次播放"):
        FramesLoopRunner().run(_ctx(tmp_path, {"mode": "once", "once_skip": 5}, _frames(tmp_path, 6)))


@pytest.mark.parametrize("mode, expected", [("loop", True), ("pingpong", True), ("once", False)])
def test_sheet_atlas_records_whether_the_clip_loops(tmp_path, mode, expected):
    ctx = _ctx(tmp_path, {"fps": 10, "mode": mode}, _frames(tmp_path, 4), "sheet", item_id="death")
    [_sheet, atlas, _gif] = SpriteSheetRunner().run(ctx)
    assert json.loads(atlas.read_text(encoding="utf-8"))["loop"] is expected


def test_sheet_atlas_without_mode_keeps_the_old_shape(tmp_path):
    ctx = _ctx(tmp_path, {"fps": 10}, _frames(tmp_path, 4), "sheet", item_id="old")
    [_sheet, atlas, _gif] = SpriteSheetRunner().run(ctx)
    assert "loop" not in json.loads(atlas.read_text(encoding="utf-8"))


def test_sheet_rejects_unknown_mode(tmp_path):
    ctx = _ctx(tmp_path, {"mode": "bounce"}, _frames(tmp_path, 4), "sheet")
    with pytest.raises(ValueError, match="mode"):
        SpriteSheetRunner().run(ctx)


# ---- image.montage ----
def _view(tmp_path, name, size):
    path = tmp_path / f"{name}.png"
    Image.new("RGBA", size, (255, 0, 0, 255)).save(path)
    return path


def test_montage_aligns_views_at_one_height_in_input_order(tmp_path):
    views = [_view(tmp_path, "front", (100, 200)), _view(tmp_path, "side", (50, 100)),
             _view(tmp_path, "back", (200, 100))]
    ctx = _ctx(tmp_path, {"height": 100, "gap": 10}, views, "views")
    [path] = MontageRunner().run(ctx)
    with Image.open(path) as sheet:
        assert sheet.height == 100
        # 100x200 -> 50x100，50x100 -> 50x100，200x100 -> 200x100，再加两个间隙
        assert sheet.width == 50 + 50 + 200 + 2 * 10
        assert sheet.getpixel((10, 50))[3] == 255          # 第一张
        assert sheet.getpixel((50 + 5, 50))[3] == 0        # 间隙是透明的
    assert ctx.meta["views"] == 3 and path.name == "views.png"


def test_montage_needs_at_least_two_images(tmp_path):
    with pytest.raises(ValueError, match="至少要两张"):
        MontageRunner().run(_ctx(tmp_path, {}, [_view(tmp_path, "a", (10, 10))]))
    with pytest.raises(ValueError, match="参数不合法"):
        MontageRunner().run(_ctx(tmp_path, {"height": 0},
                                 [_view(tmp_path, "a", (10, 10)), _view(tmp_path, "b", (10, 10))]))


def test_montage_runner_is_registered():
    assert "image.montage" in registered_runners()


# ---- 角色立绘 + 转面 ----
def test_character_turnaround_is_off_by_default_and_costs_nothing():
    character = load_template("character_basic")
    off = {"name": "x", "gender": "男性"}
    active = [s for s in character.step_ids() if character.step_active(s, off)]
    assert active == ["generate", "remove_bg", "upscale", "trim", "voice"]


def test_character_turnaround_adds_four_views_and_one_merged_sheet():
    character = load_template("character_basic")
    on = {"name": "x", "gender": "男性", "turnaround": "1"}
    character.check_item_fields(on)
    assert all(character.step_active(s, on) for s in character.step_ids())
    views = character.step("views")
    assert views.type == "image.montage" and views.deliverable
    # 正面（原立绘）在最前，后面是四个视角
    assert views.inputs == ("trim", "view_1", "view_2", "view_3", "view_4")
    instructions = [character.step(f"encode_{n}").params["instruction"] for n in range(1, 5)]
    assert len(set(instructions)) == 4 and all("旋转" in text for text in instructions)
    # 四个视角都从同一张立绘转出
    assert all(character.step(f"encode_{n}").inputs == ("trim",) for n in range(1, 5))
    assert all(step.type in registered_runners() for step in character.steps)


def test_character_turnaround_steps_are_grouped_to_load_the_edit_model_once():
    """执行器按步骤顺序、每步跑完所有条目：先全部编码、再全部去噪，Qwen-Edit 只加载一轮。"""
    ids = load_template("character_basic").step_ids()
    encodes = [ids.index(f"encode_{n}") for n in range(1, 5)]
    turns = [ids.index(f"turn_{n}") for n in range(1, 5)]
    assert max(encodes) < min(turns)
    assert ids.index("voice") < min(encodes)  # 默认流程（立绘、声线）排在转面之前，不受影响


def test_character_turnaround_is_never_drafted_by_ai():
    turnaround = next(s for s in load_template("character_basic").fields if s.id == "turnaround")
    assert turnaround.draft is False and turnaround.default == "0"


def test_other_gender_is_a_valid_choice():
    character = load_template("character_basic")
    character.check_item_fields({"name": "球球", "gender": "中性"})
    with pytest.raises(ValueError, match="性别"):
        character.check_item_fields({"name": "球球"})
