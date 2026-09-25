"""由生成器驱动的步骤 runner 基类（ADR 0004）：图片、语音、文本步骤共用同一套租约和产物处理。

生成器的生成方法有的是协程、有的是同步函数，这里统一用 asyncio.run 驱动协程，
所以执行器必须跑在没有事件循环的线程里（任务里用 asyncio.to_thread 调用）。
"""
import asyncio
import inspect
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.model_base import GeneratorFactory
from src.shared.enum_type import FactoryType

# 这些参数是本机路径，写进 meta 没有参考价值，还会暴露目录结构
_PATH_PARAMS = ("input_path", "reference_audio_path", "output_dir")


class GeneratorStepRunner(StepRunner):
    """一组条目共用一次租约：open() 取租约（并按需加载模型），run() 逐条生成。"""
    factory_type = FactoryType.Image
    # None 表示沿用生成器产物的扩展名
    output_suffix: str | None = ".png"
    # False 时 open() 只取租约、不加载模型：有的条目可能根本用不到这个模型（如对话里全员已有声线）
    load_on_open = True

    def __init__(self):
        self._generator = None
        self._model_name = ""

    @contextmanager
    def open(self, params: dict[str, Any], cancel_event: threading.Event):
        model_name = params.get("model_name")
        if not model_name:
            raise ValueError(f"步骤 {self.type_name} 缺少 model_name 参数")
        with GeneratorFactory.acquire(self.factory_type, model_name) as generator:
            generator.cancel_event = cancel_event
            if self.load_on_open:
                generator.ensure_model_loaded()
            self._generator, self._model_name = generator, model_name
            try:
                yield
            finally:
                self._generator, self._model_name = None, ""

    @property
    def generator(self):
        if self._generator is None:
            raise RuntimeError(f"步骤 {self.type_name} 未在 open() 内调用")
        return self._generator

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        path = self.generate_once(params)
        suffix = self.output_suffix or path.suffix
        ctx.meta.update(self.describe(params))
        return [move_into(path, ctx.out_dir / f"{ctx.step_id}{suffix}")]

    def generate_once(self, params: dict[str, Any]) -> Path:
        """调用一次生成器并确认真的产出了文件。"""
        generator = self.generator
        generator.check_cancelled()
        if not self.load_on_open:
            generator.ensure_model_loaded()
        generator.parse_params(params)
        path = call_blocking(self.invoke(generator))
        if not path or not Path(path).is_file():
            # 生成器失败时只打印日志并返回 None，这里必须转成异常，否则会被当成成功
            raise RuntimeError(f"{self.type_name} 没有生成文件，请查看后端日志")
        return Path(path)

    def invoke(self, generator):
        return generator.generate()

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        raise NotImplementedError

    def describe(self, params: dict[str, Any]) -> dict[str, Any]:
        """写进 manifest 的生成细节，供界面侧栏展示。"""
        meta: dict[str, Any] = {"model": self._model_name}
        if params.get("content"):
            meta["prompt"] = params["content"]
        if params.get("negative_prompt"):
            meta["negative_prompt"] = params["negative_prompt"]
        extra = {k: v for k, v in params.items()
                 if k not in ("content", "negative_prompt", *_PATH_PARAMS)}
        if extra:
            meta["params"] = extra
        return meta


def call_blocking(result):
    """生成方法可能是协程也可能是普通函数，统一拿到返回值。"""
    return asyncio.run(result) if inspect.iscoroutine(result) else result


def move_into(source: Path, target: Path) -> Path:
    """生成器写在默认输出目录，移进资源包并用固定文件名，manifest 里的路径才稳定。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, target)
    return target
