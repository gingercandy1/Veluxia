"""步骤类型注册表（ADR 0004）：每种步骤类型对应一个 runner。

runner 分两段：
- open()：整组条目开始前调用一次。GPU 步骤在这里取租约、加载模型，
  这样一组条目只加载一次模型，组内也不会被别的请求插进来换掉模型；
  CPU 步骤不占租约（ADR 0003），默认什么都不做。
- run()：逐条目调用，把产物写进 ctx.out_dir，返回产物的绝对路径；
  生成细节（模型、实际提示词等）写进 ctx.meta，执行器会存进 manifest。
"""
import threading
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.shared.schemas import CollectionItem


@dataclass(frozen=True)
class CastVoice:
    """对话出场角色解析后的结果：有声线样本就克隆它，没有就按描述现场设计。"""
    name: str
    description: str = ""
    voice_prompt: str = ""
    sample_audio: Path | None = None
    sample_text: str = ""


@dataclass
class StepContext:
    item: CollectionItem
    # 拼好风格锁的提示词，给出图步骤用；语音等步骤用 values["prompt"] 取条目原文
    prompt: str
    negative_prompt: str
    inputs: list[Path]
    # 已替换过 "{字段}" 占位符
    params: dict[str, Any]
    out_dir: Path
    step_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    values: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    cast: list[CastVoice] = field(default_factory=list)
    # 单独的风格锁：出图步骤用参数 prompt 换掉条目主提示词时，风格仍要加上
    style_prompt: str = ""


class StepRunner:
    type_name: str = ""

    def open(self, params: dict[str, Any],
             cancel_event: threading.Event) -> AbstractContextManager:
        return nullcontext()

    def run(self, ctx: StepContext) -> list[Path]:
        raise NotImplementedError

    def validate_edit(self, content: str) -> str:
        """审阅时用户提交了修改后的内容：校验并返回要写回产物文件的文本。"""
        raise ValueError(f"步骤类型 {self.type_name} 的产物不支持修改")


_RUNNERS: dict[str, StepRunner] = {}


def register_runner(runner: StepRunner) -> None:
    _RUNNERS[runner.type_name] = runner


def registered_runners() -> dict[str, StepRunner]:
    _register_builtin_runners()
    return dict(_RUNNERS)


def _register_builtin_runners() -> None:
    # 延迟导入：内置 runner 模块反过来依赖本模块，放在顶部会循环导入
    from src.backend.core.collection import (
        audio_steps,
        dialogue_steps,
        edit_steps,
        frame_steps,
        image_steps,
    )

    for module in (image_steps, audio_steps, dialogue_steps, frame_steps, edit_steps):
        for runner in module.BUILTIN_RUNNERS:
            _RUNNERS.setdefault(runner.type_name, runner)
