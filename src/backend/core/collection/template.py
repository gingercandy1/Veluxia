"""资源包模板：JSON 步骤图（ADR 0004）。

模板就是界面上的"默认节点"：每一步声明 id、步骤类型、参数和输入来自哪一步。
输入只能引用前面的步骤，这样按列表顺序执行就一定满足依赖，不需要拓扑排序。
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class StepSpec:
    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Template:
    id: str
    version: int
    type: str
    steps: tuple[StepSpec, ...]

    def step_ids(self) -> list[str]:
        return [step.id for step in self.steps]

    def downstream_of(self, step_id: str) -> list[str]:
        """直接或间接依赖 step_id 的步骤：上游重新生成后，这些步骤的旧产物都已过期。"""
        affected = {step_id}
        result = []
        for step in self.steps:
            if affected.intersection(step.inputs):
                affected.add(step.id)
                result.append(step.id)
        return result


def parse_template(data: dict) -> Template:
    """校验并构造模板；不合法时抛 ValueError，避免跑到一半才发现模板写错。"""
    template_id = data.get("id")
    if not template_id:
        raise ValueError("模板缺少 id")
    raw_steps = data.get("steps") or []
    if not raw_steps:
        raise ValueError(f"模板 {template_id} 没有任何步骤")

    seen: set[str] = set()
    steps = []
    for raw in raw_steps:
        step_id, step_type = raw.get("id"), raw.get("type")
        if not step_id or not step_type:
            raise ValueError(f"模板 {template_id} 的步骤缺少 id 或 type：{raw}")
        if step_id in seen:
            raise ValueError(f"模板 {template_id} 的步骤 id 重复：{step_id}")
        inputs = tuple(raw.get("inputs", []))
        for source in inputs:
            if source not in seen:
                raise ValueError(f"模板 {template_id} 的步骤 {step_id} 引用了未在它之前定义的步骤：{source}")
        seen.add(step_id)
        steps.append(StepSpec(step_id, step_type, dict(raw.get("params", {})), inputs))

    return Template(
        id=template_id,
        version=int(data.get("version", 1)),
        type=data.get("type", "scene"),
        steps=tuple(steps),
    )


def list_templates() -> list[Template]:
    return [load_template(path.stem) for path in sorted(TEMPLATE_DIR.glob("*.json"))]


def load_template(template_id: str) -> Template:
    path = TEMPLATE_DIR / f"{template_id}.json"
    # 模板 id 来自请求，限定在模板目录内，防止 "../" 读到别处的文件
    if path.resolve().parent != TEMPLATE_DIR or not path.is_file():
        raise ValueError(f"模板不存在：{template_id}")
    return parse_template(json.loads(path.read_text(encoding="utf-8")))
