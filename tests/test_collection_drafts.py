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
