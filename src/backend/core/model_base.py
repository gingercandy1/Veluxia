import enum
import functools
import gc
import importlib
import json
import os.path
import threading
from contextlib import contextmanager
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.backend.core.exceptions import GenerationCancelled, GeneratorBusyError
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
    """單例元類（支援抽象基類）：同一個類只有一個實例，也就只占一份顯存。

    多個模型名可以共用同一個生成器類（如 LTX-2.3 / LTX-2.5），所以再次取實例時
    如果模型名變了，要先卸載舊權重再按新名字重新初始化；否則會拿到上一個模型的實例。
    """
    _instances: Dict[type, Any] = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        with SingletonMeta._lock:
            instance = cls._instances.get(cls)
            if instance is None:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            elif kwargs.get("model_name", instance.model_name) != instance.model_name:
                instance.unload()
                instance.__init__(*args, **kwargs)
            return instance


class BaseGenerator(ABC, metaclass=SingletonMeta):
    type: enum.Enum = None
    # 持有已加载模型的属性名，unload() 会逐个释放。
    # 子类在 pipe 之外还持有模型对象（img2img 管线、rembg 会话等）时覆盖此项。
    _model_attrs: tuple[str, ...] = ("pipe",)

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
        """释放已加载的模型并复位到未加载状态；显存必须先 gc 再 empty_cache 才会真正归还。"""
        loaded = [attr for attr in self._model_attrs if getattr(self, attr, None) is not None]
        if not loaded:
            return
        for attr in loaded:
            setattr(self, attr, None)
        gc.collect()
        self.torch.cuda.empty_cache()
        print(f"✅ {self.model_name} 已卸载，显存已释放")

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
    # 当前持有租约的使用者名称；None 表示空闲。见 acquire() / exclusive()
    _holder: Optional[str] = None

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
    def _require_idle(cls):
        if cls._holder is not None:
            raise GeneratorBusyError(
                f"正在运行 {cls._holder}，请等它完成后再试（刚点了停止的话，模型仍在收尾）")

    @classmethod
    def _unload_residents(cls, keep=None):
        for instance in list(SingletonMeta._instances.values()):
            if instance is not keep:
                instance.unload()

    @classmethod
    @contextmanager
    def acquire(cls, ty, name: str):
        """
        生成期间独占生成器的租约：8GB 显存只够驻留一个模型（ADR 0001），所以
        - 有任务在跑时，任何取用（包括同一个模型）都被拒绝：切换会卸载正在使用的权重，
          同一个实例并发使用又会互相覆盖 parse_params 写入的状态；
        - 空闲时才允许取用，并先卸载其他所有驻留的生成器。
        整个生成过程（含 ensure_model_loaded）必须放在 with 块里。
        """
        with SingletonMeta._lock:
            cls._require_idle()
            generator = cls.build_generator(ty, name)
            cls._unload_residents(keep=generator)
            cls._holder = name
        try:
            yield generator
        finally:
            with SingletonMeta._lock:
                cls._holder = None

    @classmethod
    @contextmanager
    def exclusive(cls, holder: str = "提示词优化"):
        """给不是生成器、但同样占显存的模型（如提示词优化器）用：占用期间没有生成器驻留。"""
        with SingletonMeta._lock:
            cls._require_idle()
            cls._unload_residents()
            cls._holder = holder
        try:
            yield
        finally:
            with SingletonMeta._lock:
                cls._holder = None

    @classmethod
    def get_generator_names(cls, ty) -> list:
        return list(cls._generators[ty].keys())

    @classmethod
    def get_model_info(cls, ty) -> dict[str, list]:
        return cls._model[ty]
