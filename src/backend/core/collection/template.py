"""资源包模板：JSON 步骤图（ADR 0004）。

模板就是界面上的"默认节点"：每一步声明 id、步骤类型、参数和输入来自哪一步。
输入只能引用前面的步骤，这样按列表顺序执行就一定满足依赖，不需要拓扑排序。

条目除了主提示词 prompt，还可以带模板声明的附加字段（如角色的音色描述）。
步骤参数里的字符串可以写 "{字段 id}" 占位，执行时换成条目的字段值。
"""
import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.schemas import PACK_TYPES

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
# 主提示词也能在参数里引用：语音步骤要的是条目原文，而不是拼了风格锁的出图提示词
PROMPT_FIELD = "prompt"
# 步骤输入里的特殊引用：来源角色的立绘（ADR 0006），只有声明了 source 的模板能用
SOURCE_INPUT = "@source"


@dataclass(frozen=True)
class FieldSpec:
    id: str
    label: str = ""
    required: bool = False
    default: str = ""
    # (值, 显示名)：值直接拼进提示词，所以通常是英文构图描述，显示名给界面看
    options: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class StepSpec:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()
    label: str = ""
    deliverable: bool = False
    review: bool = False


@dataclass(frozen=True)
class Template:
    id: str
    version: int
    type: str
    steps: tuple[StepSpec, ...]
    name: str = ""
    description: str = ""
    prompt_label: str = ""
    cover: str = ""
    fields: tuple[FieldSpec, ...] = ()
    # 需要绑定的来源资源包类型，如 "character"；空表示不绑定
    source: str = ""

    def step_ids(self) -> list[str]:
        return [step.id for step in self.steps]

    def step(self, step_id: str) -> StepSpec:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise ValueError(f"模板 {self.id} 没有步骤：{step_id}")

    def downstream_of(self, step_id: str) -> list[str]:
        """直接或间接依赖 step_id 的步骤：上游重新生成后，这些步骤的旧产物都已过期。"""
        affected = {step_id}
        result = []
        for step in self.steps:
            if affected.intersection(step.inputs):
                affected.add(step.id)
                result.append(step.id)
        return result

    def field_values(self, prompt: str, fields: dict[str, str]) -> dict[str, str]:
        """条目的全部字段值：没填的用模板默认值，供步骤参数里的占位符替换。"""
        values = {spec.id: fields.get(spec.id) or spec.default for spec in self.fields}
        values[PROMPT_FIELD] = prompt
        return values

    def check_item_fields(self, fields: dict[str, str]) -> None:
        declared = {spec.id for spec in self.fields}
        unknown = sorted(set(fields) - declared)
        if unknown:
            raise ValueError(f"模板 {self.id} 没有这些字段：{', '.join(unknown)}")
        missing = [spec.label or spec.id for spec in self.fields
                   if spec.required and not fields.get(spec.id, "").strip()]
        if missing:
            raise ValueError(f"缺少必填字段：{', '.join(missing)}")
        for spec in self.fields:
            allowed = [value for value, _ in spec.options]
            if allowed and fields.get(spec.id) and fields[spec.id] not in allowed:
                raise ValueError(f"字段 {spec.label or spec.id} 的取值不在可选范围内：{fields[spec.id]}")
        self._check_segments(fields)

    def _check_segments(self, fields: dict[str, str]) -> None:
        """分段描述的段数要和屏数一致，否则要跑到生成那一步才报错，条目多时很难定位是哪条填错了。

        按执行器实际用的方式（render_params）解析占位符，保证这里的判断和生成时一致。
        """
        values = self.field_values("", fields)
        for step in self.steps:
            if "segment_prompts" not in step.params or "segments" not in step.params:
                continue
            rendered = render_params(step.params, values)
            prompts = [part.strip() for part in str(rendered["segment_prompts"]).replace("｜", "|").split("|")
                      if part.strip()]
            segments = rendered["segments"]
            if prompts and len(prompts) != int(segments):
                raise ValueError(f"分段描述有 {len(prompts)} 段，但长度选的是 "
                                 f"{segments} 屏，两者要一致")


def render_params(params: dict[str, Any], values: dict[str, str]) -> dict[str, Any]:
    """只替换模板里写的占位符；用户填的值不会被再次解析，里面有花括号也不会出错。"""
    return {key: value.format_map(values) if isinstance(value, str) else value
            for key, value in params.items()}


def _placeholders(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {name for _, name, _, _ in string.Formatter().parse(value) if name is not None}


def parse_template(data: dict) -> Template:
    """校验并构造模板；不合法时抛 ValueError，避免跑到一半才发现模板写错。"""
    template_id = data.get("id")
    if not template_id:
        raise ValueError("模板缺少 id")
    pack_type = data.get("type", "scene")
    if pack_type not in PACK_TYPES:
        raise ValueError(f"模板 {template_id} 的类型不支持：{pack_type}")
    raw_steps = data.get("steps") or []
    if not raw_steps:
        raise ValueError(f"模板 {template_id} 没有任何步骤")

    fields = tuple(
        FieldSpec(raw["id"], raw.get("label", ""), bool(raw.get("required", False)),
                  raw.get("default", ""),
                  tuple((opt["value"], opt.get("label", "")) for opt in raw.get("options", [])))
        for raw in data.get("fields", [])
    )
    for spec in fields:
        if spec.id == PROMPT_FIELD:
            raise ValueError(f"模板 {template_id} 的字段 id 不能是保留名：{PROMPT_FIELD}")
        if spec.options and spec.default not in [value for value, _ in spec.options]:
            raise ValueError(f"模板 {template_id} 的字段 {spec.id} 默认值不在可选范围内")
    known_fields = {spec.id for spec in fields} | {PROMPT_FIELD}
    source = data.get("source", "")
    if source and source not in PACK_TYPES:
        raise ValueError(f"模板 {template_id} 的来源类型不支持：{source}")

    seen: set[str] = set()
    steps = []
    for raw in raw_steps:
        step_id, step_type = raw.get("id"), raw.get("type")
        if not step_id or not step_type:
            raise ValueError(f"模板 {template_id} 的步骤缺少 id 或 type：{raw}")
        if step_id in seen:
            raise ValueError(f"模板 {template_id} 的步骤 id 重复：{step_id}")
        inputs = tuple(raw.get("inputs", []))
        for name in inputs:
            if name == SOURCE_INPUT:
                if not source:
                    raise ValueError(f"模板 {template_id} 没有声明 source，步骤 {step_id} 不能引用 {SOURCE_INPUT}")
            elif name not in seen:
                raise ValueError(f"模板 {template_id} 的步骤 {step_id} 引用了未在它之前定义的步骤：{name}")
        params = dict(raw.get("params", {}))
        unknown = set().union(*(_placeholders(v) for v in params.values())) - known_fields
        if unknown:
            raise ValueError(f"模板 {template_id} 的步骤 {step_id} 引用了未声明的字段：{', '.join(sorted(unknown))}")
        seen.add(step_id)
        steps.append(StepSpec(step_id, step_type, params, inputs,
                              raw.get("label", ""), bool(raw.get("deliverable", False)),
                              bool(raw.get("review", False))))

    cover = data.get("cover", "")
    if cover and cover not in seen:
        raise ValueError(f"模板 {template_id} 的封面步骤不存在：{cover}")

    return Template(
        id=template_id,
        version=int(data.get("version", 1)),
        type=pack_type,
        steps=tuple(steps),
        name=data.get("name", ""),
        description=data.get("description", ""),
        prompt_label=data.get("prompt_label", ""),
        cover=cover,
        fields=fields,
        source=source,
    )


def list_templates() -> list[Template]:
    return [load_template(path.stem) for path in sorted(TEMPLATE_DIR.glob("*.json"))]


def load_template(template_id: str) -> Template:
    path = TEMPLATE_DIR / f"{template_id}.json"
    # 模板 id 来自请求，限定在模板目录内，防止 "../" 读到别处的文件
    if path.resolve().parent != TEMPLATE_DIR or not path.is_file():
        raise ValueError(f"模板不存在：{template_id}")
    return parse_template(json.loads(path.read_text(encoding="utf-8")))
