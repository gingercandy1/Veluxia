"""AI 起草条目：按模板的字段让文本模型一次写出一批条目，填进新建表单，用户改完再建包。

提示词从模板的名称、说明和字段声明拼出来，所以场景、角色、对话等各类模板都能直接用，
新增模板不需要改这里。模型自由输出 JSON，解析时容忍代码块、思考段和截断，再按模板逐项校验。
"""
import json
import random

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


def _draft_fields(template: Template) -> list[FieldSpec]:
    """交给模型填的字段：标了 draft=false 的（如一张要十几分钟的分层）留给用户自己选，
    保持默认值。"""
    return [spec for spec in template.fields if spec.draft]


def build_draft_messages(template: Template, req: DraftItemsRequest) -> list[dict]:
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
        "- 主题里的风格、氛围词（如发光、某某游戏风格）是整包共有的，不必每条都重复。\n"
        "- 不要写画风、渲染等风格词，风格由资源包统一添加。"
    )
    # 数量在用户消息里再说一遍：小模型更听用户消息，只写在 system 里常常一条就收尾
    user = f"主题：{req.theme.strip()}\n请写 {req.count} 个条目。"
    if req.style.strip():
        user += f"\n整体风格（只作参考，不要写进条目）：{req.style.strip()}"
    if req.cast:
        roster = "\n".join(f"- {member.name}：{member.description or '（无设定）'}"
                           for member in req.cast if member.name)
        user += f"\n出场角色（只能写这些角色）：\n{roster}"
    if req.exclude:
        user += "\n已有条目（不要重复）：\n" + "\n".join(f"- {text}" for text in req.exclude)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _salvage_objects(body: str) -> list:
    """输出在数组中途被截断时，逐个取出已经写完整的对象，不让前面的条目白写。"""
    start = body.find("[")
    if start < 0:
        return []
    decoder, index, objects = json.JSONDecoder(), start + 1, []
    while True:
        while index < len(body) and body[index] in " \t\r\n,":
            index += 1
        try:
            value, index = decoder.raw_decode(body, index)
        except json.JSONDecodeError:
            return objects
        objects.append(value)


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
    messages = build_draft_messages(template, req)
    text_fields = sum(1 for spec in _draft_fields(template) if not spec.options)
    max_tokens = min(_MAX_TOKENS, 200 + req.count * _TOKENS_PER_TEXT * (1 + text_fields))

    # 走租约：生成任务在跑时直接拒绝，空闲时会先卸载驻留的生图模型再加载 LLM
    with GeneratorFactory.acquire(FactoryType.Text, req.model_name or DRAFT_MODEL) as generator:
        generator.ensure_model_loaded()
        # 温度偏高 + 每次随机种子：同一主题多点几次"刷新"要能拿到不同的条目
        # 不用 JSON 语法约束：实测会把生成速度拖慢 6~10 倍，改为解析时容错
        reply = generator.complete_chat(messages, max_tokens=max_tokens, temperature=0.9,
                                        seed=random.randrange(2**31))
    items = parse_drafts(reply, template, req.exclude)[:req.count]
    print(f"✅ AI 起草了 {len(items)} 个条目（{template.id}：{req.theme.strip()}）")
    return items
