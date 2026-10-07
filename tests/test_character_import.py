"""角色导入模板：用已有的角色图当主角，动作包 / 对话包把它当普通角色用。

无权重、无 GPU 也可跑：抠图步骤换成直接拿导入图，只验证导入、裁边和对其它包的接口。
"""
import pytest
from PIL import Image

from src.backend.core import model_utils
from src.backend.core.collection import library
from src.backend.core.collection.image_steps import ImportRunner, TrimRunner
from src.backend.core.collection.steps import StepContext, registered_runners
from src.backend.core.collection.template import load_template
from src.shared.schemas import (
    CastMember,
    CreatePackRequest,
    NewCollectionItem,
    StepState,
)


@pytest.fixture
def media(tmp_path, monkeypatch):
    root = tmp_path / "media"
    monkeypatch.setattr(model_utils, "get_media_root", lambda: root)
    monkeypatch.setattr(library, "get_media_root", lambda: root)
    return root


@pytest.fixture
def hero_png(tmp_path):
    """透明底里有一个实心圆，四周留白：裁边后应只剩圆。"""
    path = tmp_path / "hero.png"
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    image.paste(Image.new("RGBA", (60, 60), (40, 40, 40, 255)), (70, 80))
    image.save(path)
    return path


def _create_import(hero_png, **fields):
    values = {"portrait": str(hero_png), "name": "烟雾球", "gender": "中性", **fields}
    return library.create_pack(CreatePackRequest(
        name="主角", template="character_import",
        items=[NewCollectionItem(id="smoke", prompt="", fields=values)]))


def _ctx(pack, item, step_id, inputs=(), **params):
    out_dir = library.pack_dir(pack.id) / item.id
    return StepContext(item=item, prompt=item.prompt, negative_prompt="", inputs=list(inputs),
                       params=params, out_dir=out_dir, step_id=step_id,
                       values=load_template("character_import").field_values(item.prompt, item.fields))


def test_template_is_a_character_pack_that_other_packs_can_use():
    template = load_template("character_import")
    assert template.type == "character" and not template.source
    # 动作包、对话包按这两个步骤 id 找立绘和声线，导入的角色必须叫一样的名字
    assert template.step(library.CHARACTER_PORTRAIT_STEP).deliverable
    assert template.step(library.CHARACTER_VOICE_STEP).type == "speech.generate"
    assert template.cover == library.CHARACTER_PORTRAIT_STEP
    assert all(step.type in registered_runners() for step in template.steps)


def test_portrait_comes_from_the_import_not_from_generation():
    template = load_template("character_import")
    types = [step.type for step in template.steps]
    assert "image.generate" not in types  # 导入的图不能被重新生成、改动
    assert template.step("import").params["field"] == "portrait"
    assert template.step("remove_bg").inputs == ("import",)
    assert template.step("trim").inputs == ("remove_bg",)


def test_the_image_name_and_gender_are_required():
    template = load_template("character_import")
    spec = next(s for s in template.fields if s.id == "portrait")
    assert spec.required and spec.kind == "image" and spec.draft is False
    with pytest.raises(ValueError, match="角色图片"):
        template.check_item_fields({"name": "球", "gender": "中性"})
    with pytest.raises(ValueError, match="性别"):
        template.check_item_fields({"portrait": "x.png", "name": "球"})
    template.check_item_fields({"portrait": "x.png", "name": "球", "gender": "中性"})


def test_default_flow_is_import_cutout_trim_and_voice_only():
    template = load_template("character_import")
    fields = {"portrait": "x.png", "name": "球", "gender": "男性"}
    assert [s for s in template.step_ids() if template.step_active(s, fields)] == [
        "import", "remove_bg", "trim", "voice"]
    fields["turnaround"] = "1"
    assert all(template.step_active(s, fields) for s in template.step_ids())


def test_creating_the_pack_copies_the_image_into_the_pack(media, hero_png):
    pack = _create_import(hero_png)
    copied = library.pack_dir(pack.id) / pack.items[0].fields["portrait"]
    assert pack.items[0].fields["portrait"] == "smoke/portrait.png"
    assert copied.is_file() and copied.read_bytes() == hero_png.read_bytes()
    hero_png.unlink()  # 原文件删了也不影响：资源包要能整个拷走
    assert copied.is_file()


def test_creating_the_pack_rejects_a_missing_image(media, tmp_path):
    with pytest.raises(ValueError, match="找不到文件"):
        _create_import(tmp_path / "nope.png")


def test_import_then_trim_produces_the_portrait(media, hero_png):
    pack = _create_import(hero_png)
    item = pack.items[0]
    [imported] = ImportRunner().run(_ctx(pack, item, "import", field="portrait"))
    assert imported.name == "import.png"
    [trimmed] = TrimRunner().run(_ctx(pack, item, "trim", [imported], padding=0))
    assert trimmed.name == "trim.png"
    with Image.open(trimmed) as image:
        assert image.size == (60, 60)  # 四周的留白被裁掉了


def test_motion_pack_can_bind_to_an_imported_character(media, hero_png):
    pack = _create_import(hero_png)
    reference = f"{pack.id}/smoke"
    motion = library.create_pack(CreatePackRequest(
        name="动作", template="character_motion", source=reference,
        items=[NewCollectionItem(prompt="rolls forward",
                                 fields={"action": "roll", "loop_mode": "loop"})]))
    assert motion.source == reference and motion.items[0].id == "roll"


def test_dialogue_pack_can_cast_an_imported_character(media, hero_png):
    pack = _create_import(hero_png)
    talk = library.create_pack(CreatePackRequest(
        name="对白", template="dialogue_scene", cast=[
            CastMember(name="烟雾球", character=f"{pack.id}/smoke")],
        items=[NewCollectionItem(prompt="烟雾球自我介绍")]))
    assert talk.cast[0].character == f"{pack.id}/smoke"


def test_source_portrait_is_the_trimmed_import_once_it_is_done(media, hero_png):
    pack = _create_import(hero_png)
    reference = f"{pack.id}/smoke"
    with pytest.raises(ValueError, match="立绘还没有生成"):
        library._source_portrait(reference)
    item = pack.items[0]
    [imported] = ImportRunner().run(_ctx(pack, item, "import", field="portrait"))
    [trimmed] = TrimRunner().run(_ctx(pack, item, "trim", [imported]))
    stored = library.load_manifest(library.pack_dir(pack.id))
    stored.items[0].steps["trim"] = StepState(status="done", outputs=["smoke/trim.png"])
    library.save_manifest(library.pack_dir(pack.id), stored)
    assert library._source_portrait(reference) == trimmed
