"""AI 起草条目：按模板的字段让文本模型一次写出一批条目，填进新建表单，用户改完再建包。

提示词从模板的名称、说明和字段声明拼出来，所以场景、角色、对话等各类模板都能直接用，
新增模板不需要改这里。模型自由输出 JSON，解析时容忍代码块、思考段和截断，再按模板逐项校验。
"""
import json
import random

from src.backend.core.collection.library import character_description
from src.backend.core.collection.template import (
    PROMPT_FIELD,
    FieldSpec,
    Template,
    count_segments,
    load_template,
)
from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType
from src.shared.schemas import DraftItemsRequest, NewCollectionItem

# 与对话包写剧本用同一个模型：中文好、Q4 约 5GB，8GB 显存放得下；
# 不用 Qwen3.5 这类思考型模型：会先写上千 token 的思考，4096 上下文里经常写不到正文
DRAFT_MODEL = "Qwen2.5-7B"
MAX_DRAFT_COUNT = 20
# LLM 上下文是 4096，扣掉提示词后留给输出的上限；每个条目按字段数估算需要的长度
_MAX_TOKENS = 3000
_TOKENS_PER_TEXT = 60

# 动作包（模板声明了 source）的条目是给图生视频模型的动作描述，主体来自绑定的角色。
# 视频模型不知道主体是什么时会往人形上猜，球、箱子也会长出手脚，所以要把主体和它没有的部位写明
_MOTION_RULES = (
    "- 这是动作包：每个条目是一个动作，主体就是用户给的「绑定的物体」。"
    f'"{PROMPT_FIELD}" 要写成给图生视频模型看的动作描述：先点明主体是什么，再写这个动作具体怎么动。\n'
    "- 只写主体实际有的部位。球、箱子、史莱姆、道具这类没有四肢、没有脸的物体，"
    "绝不能出现手、脚、腿、胳膊、脸、走路、挥手这类人形动作，"
    "要改用弹跳、滚动、挤压拉伸、旋转、摇摆、闪烁来表现（如「跑」写成快速向前滚动并轻微弹起）。\n"
    "- 动作在原地循环，镜头不动。\n"
    "- 走、跑、滚动这类周期动作循环方式选 loop；待机、呼吸、漂浮这类小幅动作选 pingpong。"
)


# 生图模型（Z-Image-Turbo）蒸馏版引导系数为 0，不读负面提示词，"没有手脚"只是正向文字里的
# 几个字（实测写不写这句，球形角色照样长脚）；所以只写看得见的特征，不写"没有什么"
_IMAGE_PROMPT_RULES = (
    "\n- 描述只写画面里有什么，不要写“没有……”“不带……”：生图模型不读否定，"
    "写了也不起作用。想要球形就直接写“圆球”，别写“没有手脚”。"
)
# 主体会被抠出来单独使用（角色、图标、道具、花草）：生成时后缀要求纯白背景，
# 描述里再写城堡、森林，模型两头都照做，抠图时背景就连着主体一起留下（实测）
_CUTOUT_RULES = (
    "\n- 这些条目会被单独抠出来使用，描述里不要写背景、场景、地面、阴影、周围的环境，"
    "只写主体本身。"
)


def _prompt_rules(template: Template) -> str:
    types = {step.type for step in template.steps}
    return (_IMAGE_PROMPT_RULES * ("image.generate" in types)
            + _CUTOUT_RULES * ("image.generate" in types and "image.remove_bg" in types))


def _draft_fields(template: Template) -> list[FieldSpec]:
    """交给模型填的字段：标了 draft=false 的（如一张要十几分钟的分层）留给用户自己选，
    保持默认值。"""
    return [spec for spec in template.fields if spec.draft]


def build_draft_messages(template: Template, req: DraftItemsRequest,
                         source_description: str = "") -> list[dict]:
    prompt_label = template.prompt_label or "描述"
    keys = [f'- "{PROMPT_FIELD}"（{prompt_label}）：给生成模型用的具体描述，'
            "写清外观、材质、颜色、形状等看得见的特征，一两句话"]
    for spec in _draft_fields(template):
        line = f'- "{spec.id}"（{spec.label or spec.id}）'
        if spec.options:
            # 空值是"请选择"占位，不是可选答案
            choices = "、".join(f"{value}（{label}）" if label else value
                               for value, label in spec.options if value)
            line += f"：只能取以下值之一 {choices}"
        elif spec.default:
            line += f"：参考写法「{spec.default}」"
        keys.append(line)
    system = (
        f"你是游戏策划，为资源包「{template.name or template.id}」起草条目。{template.description}\n"
        f"每个条目是一个 JSON 对象，包含这些键：\n" + "\n".join(keys) + "\n"
        f'只输出 JSON，格式：{{"items": [条目1, 条目2, ..., 条目{req.count}]}}。\n'
        f"要求：\n- 正好 {req.count} 个条目，每一个都必须属于用户给的主题，不能跑题；"
        "条目多时宁可在主题内换形态、材质、大小，也不要写主题以外的东西。\n"
        "- 条目之间差异明显，不要只换颜色或换个词；各条目的开头措辞也要不同。\n"
        "- 不要写画风、渲染等风格词，风格由资源包统一添加。"
        + _prompt_rules(template)
    )
    if template.source:
        system += "\n" + _MOTION_RULES
    # 数量在用户消息里再说一遍：小模型更听用户消息，只写在 system 里常常一条就收尾
    user = f"主题：{req.theme.strip()}\n请写 {req.count} 个条目。"
    if source_description:
        user += f"\n绑定的物体：{source_description}"
    if req.style.strip():
        user += f"\n整体风格（只作参考，不要写进条目）：{req.style.strip()}"
    if req.cast:
        roster = "\n".join(f"- {member.name}：{member.description or '（无设定）'}"
                           for member in req.cast if member.name)
        user += f"\n出场角色（只能写这些角色）：\n{roster}"
    if req.exclude:
        user += "\n已有条目（不要重复）：\n" + "\n".join(f"- {text}" for text in req.exclude)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _salvage_objects(body: str) -> list[dict]:
    """整体解析不了时，逐个取出写完整的对象，不让写好的条目白写。

    不只应对末尾截断：小模型还会写出 ["{"prompt": "…"}", …] 这种对象外多套引号的坏 JSON，
    整体和逐元素都解析不了，但每个对象本身是完整的。所以在每个 { 处都试着解析一次，
    成功就跳过整个对象（不再拆它里面嵌套的对象），失败就找下一个 {。
    """
    decoder, objects = json.JSONDecoder(), []
    index = body.find("{")
    while index >= 0:
        try:
            value, end = decoder.raw_decode(body, index)
        except json.JSONDecodeError:
            index = body.find("{", index + 1)
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = body.find("{", end)
    return objects


def _extract_items(text: str) -> list:
    body = text.strip()
    # 思考型模型会先输出 <think> 段；有的模型把 JSON 包在代码块里或前后加说明
    if "</think>" in body:
        body = body.split("</think>", 1)[1]
    starts = [i for i in (body.find("{"), body.find("[")) if i >= 0]
    if not starts:
        raise ValueError("模型没有输出 JSON，请重试")
    body = body[min(starts):]
    try:
        data, _ = json.JSONDecoder().raw_decode(body)
    except json.JSONDecodeError:
        raw_items = _salvage_objects(body)
        if not raw_items:
            raise ValueError("模型输出不完整，请减少数量后重试") from None
        return raw_items
    raw_items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise ValueError("模型输出里没有 items 数组")
    return raw_items


def _align_segments(template: Template, fields: dict[str, str]) -> None:
    """小模型常把长度和分段描述写得对不上，建包时会被拒。以分段描述为准改长度（写的内容更具体），
    长度选项里没有这个段数时丢掉分段描述，退回整张统一描述。"""
    for segments_id, prompts_id in template.segment_fields():
        count = count_segments(fields.get(prompts_id, ""))
        if not count:
            continue
        spec = next((spec for spec in template.fields if spec.id == segments_id), None)
        allowed = [value for value, _ in spec.options] if spec else []
        if not allowed or str(count) in allowed:
            fields[segments_id] = str(count)
        else:
            fields.pop(prompts_id)


def parse_drafts(text: str, template: Template, exclude: list[str] = ()) -> list[NewCollectionItem]:
    """解析模型输出并按模板校验：未声明的键丢掉，下拉字段的取值不合法时留空走默认值。"""
    raw_items = _extract_items(text)

    seen = {value.strip().lower() for value in exclude}
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        prompt = str(raw.get(PROMPT_FIELD, "")).strip()
        if not prompt or prompt.lower() in seen:
            continue
        seen.add(prompt.lower())
        fields = {}
        for spec in _draft_fields(template):
            value = str(raw.get(spec.id, "")).strip()
            if spec.options:
                # 模型有时照抄显示名而不是值，两种都认
                value = next((v for v, label in spec.options if value in (v, label)), "")
            if value:
                fields[spec.id] = value
        _align_segments(template, fields)
        items.append(NewCollectionItem(prompt=prompt, fields=fields))
    if not items:
        raise ValueError("模型没有给出可用的条目，请换个主题描述再试")
    return items


def draft_items(req: DraftItemsRequest) -> list[NewCollectionItem]:
    if not req.theme.strip():
        raise ValueError("请先填写主题")
    if not 1 <= req.count <= MAX_DRAFT_COUNT:
        raise ValueError(f"数量需在 1~{MAX_DRAFT_COUNT} 之间")
    template = load_template(req.template)
    source_description = ""
    if template.source:
        if not req.source:
            raise ValueError("这类资源包需要先选择一个角色")
        source_description = character_description(req.source)
    messages = build_draft_messages(template, req, source_description)
    text_fields = sum(1 for spec in _draft_fields(template) if not spec.options)
    max_tokens = min(_MAX_TOKENS, 200 + req.count * _TOKENS_PER_TEXT * (1 + text_fields))

    with GeneratorFactory.acquire(FactoryType.Text, req.model_name or DRAFT_MODEL) as generator:
        generator.ensure_model_loaded()
        reply = generator.complete_chat(messages, max_tokens=max_tokens,
                                        temperature=0.9, seed=random.randrange(2**31))
    try:
        items = parse_drafts(reply, template, req.exclude)[:req.count]
    except ValueError:
        print(f"⚠️ AI 起草的模型输出无法使用，开头：{reply[:400]!r}")
        raise

    print(f"✅ AI 起草了 {len(items)} 个条目（{template.id}：{req.theme.strip()}）")
    return items
