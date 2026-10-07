"""AI 起草条目（资源包新建表单）：提示词按模板字段拼装、输出校验、租约。用假 LLM，无权重也可跑。"""
import json

import pytest

from src.backend.core.collection.drafts import build_draft_messages, draft_items, parse_drafts
from src.backend.core.collection.template import load_template, parse_template
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import BaseTextGenerator, GeneratorFactory, SingletonMeta
from src.shared.enum_type import FactoryType
from src.shared.schemas import CastMember, DraftItemsRequest

TEMPLATE = parse_template({
    "id": "chars", "name": "角色立绘", "description": "全身立绘。", "prompt_label": "外观描述",
    "fields": [
        {"id": "name", "label": "角色名", "required": True},
        {"id": "pose", "label": "姿势", "default": "standing",
         "options": [{"value": "standing", "label": "站立"}, {"value": "sitting", "label": "坐姿"}]},
    ],
    "steps": [{"id": "generate", "type": "image.generate"}],
})


def _request(**kwargs) -> DraftItemsRequest:
    return DraftItemsRequest(**{"template": "scene_props", "theme": "发光森林的地面细节",
                                "count": 3, **kwargs})


def test_messages_describe_every_field_and_the_context():
    request = _request(style="hand-painted", exclude=["mossy rock"],
                       cast=[CastMember(name="亚瑟", description="骑士")])
    system, user = (m["content"] for m in build_draft_messages(TEMPLATE, request))
    assert '"prompt"（外观描述）' in system and '"name"（角色名）' in system
    # 下拉字段要把可选值告诉模型，否则它会自己编
    assert "standing（站立）" in system and "sitting（坐姿）" in system
    assert "正好 3 个" in system
    assert "发光森林" in user and "hand-painted" in user
    assert "亚瑟：骑士" in user and "- mossy rock" in user


def test_parse_keeps_declared_fields_and_maps_option_labels():
    reply = json.dumps({"items": [
        {"prompt": "a knight", "name": "亚瑟", "pose": "坐姿", "extra": "x"},
        {"prompt": "a mage", "name": "梅林", "pose": "flying"},
    ]}, ensure_ascii=False)
    items = parse_drafts(reply, TEMPLATE)
    assert [(i.prompt, i.fields) for i in items] == [
        ("a knight", {"name": "亚瑟", "pose": "sitting"}),
        # 不合法的下拉值留空，建包时走模板默认值
        ("a mage", {"name": "梅林"}),
    ]


def test_character_gender_must_be_chosen_and_leads_the_voice_prompt():
    character = load_template("character_basic")
    system, _ = (m["content"] for m in build_draft_messages(character, _request()))
    # "请选择"占位不是可选答案，不能出现在给模型的可选值里
    assert "男性（男）、女性（女）" in system and "请选择" not in system
    items = parse_drafts(json.dumps({"items": [
        {"prompt": "a knight", "name": "亚瑟", "gender": "男"},
        {"prompt": "a witch", "name": "莉莉", "gender": "请选择"},
    ]}, ensure_ascii=False), character)
    assert items[0].fields["gender"] == "男性" and "gender" not in items[1].fields
    with pytest.raises(ValueError, match="性别"):
        character.check_item_fields(items[1].fields)
    voice = character.step("voice").params["voice_prompt"]
    assert voice.format_map(character.field_values("a knight", items[0].fields)).startswith("男性，")


def test_parse_aligns_length_with_segment_count():
    scene = load_template("scene_background")
    items = parse_drafts(json.dumps({"items": [
        {"prompt": "forest", "length": "2", "segment_desc": "森林 | 村庄 | 城堡"},
        {"prompt": "cave", "segment_desc": "a|b|c|d|e"},
        {"prompt": "town", "length": "3"},
    ]}, ensure_ascii=False), scene)
    # 以分段描述为准改长度；没有 5 屏的选项就丢掉分段描述；没写分段描述的不动
    assert [i.fields.get("length") for i in items] == ["3", None, "3"]
    assert "segment_desc" not in items[1].fields
    for item in items:
        scene.check_item_fields(item.fields)


def test_parse_drops_empty_duplicate_and_excluded_items():
    reply = json.dumps({"items": [
        {"prompt": "Mossy Rock"}, {"prompt": ""}, "junk", {"prompt": "a fern"}, {"prompt": "A Fern"},
    ]})
    items = parse_drafts(reply, TEMPLATE, exclude=["mossy rock"])
    assert [i.prompt for i in items] == ["a fern"]


@pytest.mark.parametrize("reply", [
    '```json\n{"items": [{"prompt": "a fern"}]}\n```',
    '<think>先想想 {不是 JSON}</think>\n{"items": [{"prompt": "a fern"}]}',
    '好的，以下是条目：\n[{"prompt": "a fern"}]\n希望有帮助',
])
def test_parse_tolerates_wrappers_around_json(reply):
    assert [i.prompt for i in parse_drafts(reply, TEMPLATE)] == ["a fern"]


def test_parse_salvages_complete_items_from_truncated_output():
    reply = '{"items": [{"prompt": "a fern"}, {"prompt": "a rock"}, {"prompt": "a lo'
    assert [i.prompt for i in parse_drafts(reply, TEMPLATE)] == ["a fern", "a rock"]


@pytest.mark.parametrize("reply, message", [
    ('{"items": [{"prompt": "a', "不完整"),
    ('{"lines": []}', "items"),
    ('{"items": [{"prompt": ""}]}', "没有给出"),
    ("抱歉，我无法完成", "没有输出 JSON"),
])
def test_parse_rejects_unusable_output(reply, message):
    with pytest.raises(ValueError, match=message):
        parse_drafts(reply, TEMPLATE)


class _FakeLLM(BaseTextGenerator):
    reply = ""
    calls: list = []

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass

    async def generate(self):
        return None

    def complete_chat(self, messages, max_tokens=1024, temperature=0.7, json_output=False,
                      seed=None):
        type(self).calls.append((messages, max_tokens, seed))
        return type(self).reply


@pytest.fixture
def fake_llm():
    _FakeLLM.calls = []
    GeneratorFactory.register_generator(FactoryType.Text, "fake-llm", _FakeLLM)
    yield _FakeLLM
    GeneratorFactory._generators[FactoryType.Text].pop("fake-llm", None)
    SingletonMeta._instances.pop(_FakeLLM, None)
    GeneratorFactory._holder = None


def test_draft_items_caps_count(fake_llm):
    fake_llm.reply = json.dumps({"items": [{"prompt": f"prop {i}"} for i in range(5)]})
    items = draft_items(_request(model_name="fake-llm"))
    assert [i.prompt for i in items] == ["prop 0", "prop 1", "prop 2"]
    [(messages, max_tokens, seed)] = fake_llm.calls
    # 固定种子会让每次重新加载模型后的"刷新"结果都一样
    assert max_tokens <= 3000 and seed is not None
    assert load_template("scene_props").name in messages[0]["content"]


@pytest.mark.parametrize("kwargs, message", [
    ({"theme": "  "}, "主题"),
    ({"count": 0}, "数量"),
    ({"count": 21}, "数量"),
    ({"template": "../models"}, "模板不存在"),
])
def test_draft_items_validates_request(fake_llm, kwargs, message):
    with pytest.raises(ValueError, match=message):
        draft_items(_request(model_name="fake-llm", **kwargs))
    assert not fake_llm.calls


def test_draft_items_is_rejected_while_a_generator_is_busy(fake_llm):
    with (
        GeneratorFactory.exclusive("生成任务"),
        pytest.raises(GeneratorBusyError),
    ):
        draft_items(_request(model_name="fake-llm"))


def test_expensive_fields_are_left_to_the_user():
    # 分层一张十几分钟：模型既不会被要求填它，填了也会被丢掉，保持默认的"不分层"
    template = load_template("scene_background")
    request = DraftItemsRequest(template="scene_background", theme="森林", count=1)
    system = build_draft_messages(template, request)[0]["content"]
    assert '"layers"' not in system and '"length"' in system
    reply = json.dumps({"items": [{"prompt": "a misty forest", "layers": "4", "length": "1"}]})
    [item] = parse_drafts(reply, template)
    assert "layers" not in item.fields and item.fields["length"] == "1"


# ---- 动作包：按绑定的物体写动作 ----
def test_motion_messages_carry_the_subject_and_forbid_extra_limbs():
    motion = load_template("character_motion")
    request = DraftItemsRequest(template="character_motion", theme="常用动作", count=2,
                                source="c1/ball")
    system, user = (m["content"] for m in build_draft_messages(motion, request, "一个红色的球体"))
    assert "绑定的物体：一个红色的球体" in user
    # 没有四肢的主体要改用弹跳、滚动表现，不能写手脚
    assert "没有四肢" in system and "弹跳" in system
    assert "loop（" in system and "pingpong（" in system
    # 烟雾、火焰这类表面特效要一直在，待机只做流动
    assert "消散" in system and "待机" in system
    # 其他模板的提示词不受影响
    plain_system, plain_user = (m["content"] for m in build_draft_messages(TEMPLATE, _request()))
    assert "绑定的物体" not in plain_user and "弹跳" not in plain_system


def test_motion_draft_looks_up_the_bound_character(fake_llm, monkeypatch):
    from src.backend.core.collection import drafts
    looked_up = []
    monkeypatch.setattr(drafts, "character_description",
                        lambda ref: looked_up.append(ref) or "一个红色的球体")
    fake_llm.reply = json.dumps({"items": [
        {"prompt": "红色球体原地弹跳，落地时压扁、弹起时拉长", "loop_mode": "loop"}]},
        ensure_ascii=False)
    [item] = draft_items(DraftItemsRequest(template="character_motion", theme="跳跃", count=1,
                                           source="c1/ball", model_name="fake-llm"))
    assert looked_up == ["c1/ball"] and item.fields == {"loop_mode": "loop"}
    [(messages, _, _)] = fake_llm.calls
    assert "一个红色的球体" in messages[1]["content"]


def test_motion_draft_requires_a_character(fake_llm):
    with pytest.raises(ValueError, match="角色"):
        draft_items(DraftItemsRequest(template="character_motion", theme="跳跃", count=1,
                                      model_name="fake-llm"))
    assert not fake_llm.calls


def test_character_description_reports_missing_pack(tmp_path, monkeypatch):
    from src.backend.core.collection import library
    monkeypatch.setattr(library, "get_media_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="找不到角色包"):
        library.character_description("missing/ball")
