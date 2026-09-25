"""资源包执行器（ADR 0004）。

外层循环按步骤、内层循环按条目：同一步骤的所有条目共用一次 runner.open()，
8GB 单驻留下换模型的次数只和步骤数有关，和条目数无关。
每处理完一个条目就写一次 manifest，断点续跑和界面进度都只读它。
"""
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

from src.backend.core.collection.manifest import load_manifest, save_manifest
from src.backend.core.collection.steps import (
    CastVoice,
    StepContext,
    StepRunner,
    registered_runners,
)
from src.backend.core.collection.template import StepSpec, Template, render_params
from src.backend.core.exceptions import GenerationCancelled, GeneratorBusyError
from src.shared.schemas import CollectionItem, Manifest, StepState


def compose_prompt(prompt: str, style_prompt: str) -> str:
    return ", ".join(part.strip() for part in (prompt, style_prompt) if part.strip())


class CollectionExecutor:
    def __init__(self, pack_dir: Path, template: Template,
                 runners: Mapping[str, StepRunner] | None = None,
                 cancel_event: threading.Event | None = None,
                 busy_wait_seconds: float = 5.0,
                 cast: list[CastVoice] | None = None):
        self.pack_dir = Path(pack_dir)
        self.template = template
        self.cast = list(cast or [])
        self.runners = dict(runners) if runners is not None else registered_runners()
        self.cancel_event = cancel_event or threading.Event()
        self.busy_wait_seconds = busy_wait_seconds

        missing = [s.type for s in template.steps if s.type not in self.runners]
        if missing:
            raise ValueError(f"模板 {template.id} 用到了未注册的步骤类型：{', '.join(missing)}")

    def run(self) -> Manifest:
        manifest = load_manifest(self.pack_dir)
        # 先给每个条目补齐全部步骤的状态，界面一开始就能画出完整的进度表格
        for item in manifest.items:
            for step_id in self.template.step_ids():
                self._state(item, step_id)
        save_manifest(self.pack_dir, manifest)

        for step in self.template.steps:
            pending = [item for item in manifest.items if self._needs_run(item, step)]
            if not pending:
                continue
            with self._open_with_retry(step):
                for item in pending:
                    self._check_cancelled()
                    self._run_item(manifest, item, step)
        return manifest

    def _needs_run(self, item: CollectionItem, step: StepSpec) -> bool:
        """上游全部完成、且本步没有完整产物时才执行；上游失败的条目留在 pending。
        需要审阅的上游还要等用户确认，避免在没确认的内容上跑耗时的下游。"""
        for source in step.inputs:
            source_state = self._state(item, source)
            if source_state.status != "done":
                return False
            if self.template.step(source).review and not source_state.approved:
                return False
        state = self._state(item, step.id)
        if state.status != "done":
            return True
        return not state.outputs or not all((self.pack_dir / p).exists() for p in state.outputs)

    def _run_item(self, manifest: Manifest, item: CollectionItem, step: StepSpec) -> None:
        state = self._state(item, step.id)
        state.status, state.error, state.outputs = "running", None, []
        # 重新生成的内容用户还没看过，之前的确认作废
        state.meta, state.approved = {}, False
        save_manifest(self.pack_dir, manifest)

        out_dir = self.pack_dir / item.id
        out_dir.mkdir(parents=True, exist_ok=True)
        values = self.template.field_values(item.prompt, item.fields)
        ctx = StepContext(
            item=item,
            prompt=compose_prompt(item.prompt, manifest.style.prompt),
            negative_prompt=manifest.style.negative,
            inputs=[self.pack_dir / p for source in step.inputs
                    for p in self._state(item, source).outputs],
            params=render_params(step.params, values),
            out_dir=out_dir,
            step_id=step.id,
            cancel_event=self.cancel_event,
            values=values,
            cast=self.cast,
        )
        started = time.monotonic()
        try:
            outputs = self.runners[step.type].run(ctx)
            if not outputs:
                raise RuntimeError("步骤没有产出任何文件")
            state.outputs = [self._relative(path) for path in outputs]
            state.meta = {**ctx.meta, "elapsed": round(time.monotonic() - started, 1)}
            state.status = "done"
            # 本步重新生成后，下游的旧产物已过期，要跟着重跑
            for downstream in self.template.downstream_of(step.id):
                item.steps[downstream] = StepState()
        except GenerationCancelled:
            state.status = "pending"
            raise
        except Exception as exc:  # noqa: BLE001
            # 单条失败只影响这一条：记下原因，下游跳过它，其他条目照常执行
            state.status = "error"
            state.error = str(exc) or exc.__class__.__name__
            print(f"❌ 资源包 {manifest.id} 条目 {item.id} 步骤 {step.id} 失败：{state.error}")
        finally:
            save_manifest(self.pack_dir, manifest)

    def _open_with_retry(self, step: StepSpec) -> ExitStack:
        """遇到别的任务占用模型时等待重试：不判失败，也不抢占正在运行的任务。"""
        runner = self.runners[step.type]
        while True:
            self._check_cancelled()
            stack = ExitStack()
            try:
                stack.enter_context(runner.open(dict(step.params), self.cancel_event))
                return stack
            except GeneratorBusyError as exc:
                stack.close()
                print(f"⚠️ 步骤 {step.id} 等待模型空闲：{exc}")
                if self.cancel_event.wait(self.busy_wait_seconds):
                    raise GenerationCancelled() from exc

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise GenerationCancelled()

    def _relative(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.pack_dir.resolve()).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"产物不在资源包目录内：{path}") from exc

    @staticmethod
    def _state(item: CollectionItem, step_id: str) -> StepState:
        return item.steps.setdefault(step_id, StepState())
