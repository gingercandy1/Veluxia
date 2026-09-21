import enum
import functools
import gc
import importlib
import json
import os.path
import threading
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_utils import get_device
from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT


@functools.lru_cache(maxsize=1)
def load_models_config() -> dict:
    """读取 models.json（模型清单的唯一来源），进程内只读一次。"""
    try:
        with open(os.path.join(PROJECT_ROOT, "models.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Models read error:", e)
        return {}


class SingletonMeta(ABCMeta):
    """實現單例的元類（支援抽象基類）"""
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class BaseGenerator(ABC, metaclass=SingletonMeta):
    type: enum.Enum = None

    def __init__(self, model_name: str, device: str):
        self.pipe = None
        self.model_name = model_name
        # 由 Router 在拿到 Job 后注入；None 表示这次调用不支持/不需要取消。
        self.cancel_event: Optional[threading.Event] = None
        # 模型就绪前的分阶段进度，前端据此显示"下载 42%"/"加载模型中"。
        # 加载跑在线程池里，路由那边靠读这个字典来推送进度。
        self.load_stage: Dict[str, Any] = {"stage": "idle", "progress": 0.0, "detail": ""}

        type_id = FactoryType.convert_to_text(self.type)
        model_info = load_models_config().get(type_id, {}).get(self.model_name, None)
        if isinstance(model_info, dict):
            model_info = dict(model_info)
            model_info.pop("generator", None)
            self.model_id = model_info.pop("repo_id", None)
            self.model_filename = model_info.pop("filename", None)
            self.model_extra = model_info
        elif isinstance(model_info, str):
            self.model_id = model_info
            self.model_filename = None
        else:
            self.model_id = None
            self.model_filename = None

        self.device = get_device(device)
        print(self.model_name, self.model_id, type_id)

    @abstractmethod
    def _check_model_file(self):
        pass

    @abstractmethod
    def _load_model(self):
        pass

    def report_load_stage(self, stage: str, progress: float = 0.0, detail: str = ""):
        """更新加载进度。字典整个替换（而不是逐键改），读的那一侧就不用加锁。"""
        self.load_stage = {"stage": stage, "progress": progress, "detail": detail}

    def ensure_model_loaded(self):
        """确保模型已加载（供外部调用）"""
        if self.pipe is None:
            try:
                self.report_load_stage("checking", detail=self.model_name)
                self._check_model_file()
                self.report_load_stage("loading", detail=self.model_name)
                self._load_model()
            except BaseException:
                self.report_load_stage("idle")
                raise
        self.report_load_stage("ready", 1.0)

    @abstractmethod
    def parse_params(self, raw: dict):
        pass

    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.torch.cuda.empty_cache()
            gc.collect()
            print("✅ ACE-Step1.5 已卸载")

    @property
    def torch(self):
        import torch  # 后台注册线程早已 import 过，这里只是拿缓存，不会重新触发加载
        return torch

    def check_cancelled(self):
        """在生成循环的可中断点调用：用户点了停止就在这里抛出，中断当前推理。"""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise GenerationCancelled()

    def make_cancel_callback(self):
        """
        返回一个 diffusers `callback_on_step_end` 兼容的回调：
        每个去噪步结束时检查一次取消标记，命中则抛出异常提前结束 pipe() 调用。
        """
        def _callback(pipe, step, timestep, callback_kwargs):
            self.check_cancelled()
            return callback_kwargs
        return _callback

class BaseTextGenerator(BaseGenerator):
    """所有图片生成模型的基类（文本 → 图像）"""
    type = FactoryType.Text

    @abstractmethod
    async def generate(self):
        pass


class BaseImageGenerator(BaseGenerator):
    """所有图片生成模型的基类（文本 → 图像）"""
    type = FactoryType.Image

    @abstractmethod
    async def generate(self):
        pass


class BaseImageFrameGenerator(BaseGenerator):
    type = FactoryType.ImageFrame

    @abstractmethod
    async def generate(self):
        pass


class BaseAnimationGenerator(BaseGenerator):
    """图像转动画序列的基类（单张参考图 → 多帧动画）"""
    type = FactoryType.Animation

    @abstractmethod
    async def generate_animation(self) -> tuple[List[Path], Path]:
        pass


class BaseSpeechGenerator(BaseGenerator):
    """图像转动画序列的基类（单张参考图 → 多帧动画）"""
    type = FactoryType.Speech

    @abstractmethod
    async def generate_music(self) -> Path:
        pass


@dataclass(frozen=True)
class GeneratorSpec:
    """生成器的延迟加载描述：只记录位置，真正用到时才 import（避免启动时加载 torch 等重型依赖）。"""
    module: str
    cls: str

    @classmethod
    def parse(cls, path: str) -> "GeneratorSpec":
        """解析 models.json 里的 "generator" 值：'包.模块:类名'，模块路径相对 src.backend.core。"""
        module, _, name = path.partition(":")
        return cls(f"src.backend.core.{module}", name)

    def load(self) -> type:
        return getattr(importlib.import_module(self.module), self.cls)


class GeneratorFactory:
    """生成器工厂，方便后续扩展模型"""
    _generators: Dict[FactoryType, Dict[str, type]] = {t: {} for t in FactoryType}
    _model: Dict[FactoryType, Dict[str, list]] = {t: {} for t in FactoryType}
    _device: str = "cpu"
    _ready_event: threading.Event = threading.Event()
    _resolve_lock: threading.Lock = threading.Lock()

    @classmethod
    def mark_ready(cls):
        """注册表已填充后调用，供 /ready 探针使用。"""
        cls._ready_event.set()

    @classmethod
    def is_ready(cls) -> bool:
        return cls._ready_event.is_set()

    @classmethod
    def apply_setting(cls, setting: dict):
        """
        运行时可反复调用。
        只更新配置，不重新注册模型。
        """
        new_device = setting.get("gpu", {}).get("backend", "cpu")
        if new_device != cls._device:
            print(f"⚙️ device 变更: {cls._device} → {new_device}")
            cls._device = new_device

    @classmethod
    def register_generator(cls, ty, name: str, generator_cls):
        cls._generators[ty].update({name: generator_cls})

    @classmethod
    def register_model_info(cls, ty, tag: str, name: str):
        names = cls._model[ty].setdefault(tag, [])
        if name not in names:
            names.append(name)

    @classmethod
    def build_generator(cls, ty, name: str):
        entry = cls._generators.get(ty, {}).get(name)
        if entry is None:
            raise ValueError(f"未知的生成器: {name}")
        if isinstance(entry, GeneratorSpec):
            with cls._resolve_lock:
                entry = cls._generators[ty].get(name)
                if isinstance(entry, GeneratorSpec):
                    entry = entry.load()
                    cls._generators[ty][name] = entry
        return entry(model_name=name, device=cls._device)

    @classmethod
    def get_generator_names(cls, ty) -> list:
        return list(cls._generators[ty].keys())

    @classmethod
    def get_model_info(cls, ty) -> dict[str, list]:
        return cls._model[ty]
