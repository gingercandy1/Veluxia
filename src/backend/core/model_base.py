import enum
import gc
import json
import os.path
from abc import ABC, abstractmethod, ABCMeta
from pathlib import Path
from typing import List, Dict, Any

from core.preloader import preloader
from src.backend.core.model_utils import get_device
from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT


class SingletonMeta(ABCMeta):
    """實現單例的元類（支援抽象基類）"""
    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class BaseGenerator(ABC, metaclass=SingletonMeta):
    names: Dict[str, Dict] = {}
    dynamic: bool = False
    type: enum.Enum = None
    config: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            if not cls.config:
                config_path = os.path.join(PROJECT_ROOT, "models.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                cls._config = config
        except Exception as e:
            print("Models read error:", e)
            return
        cls.register_to_factory(cls._config)

    @classmethod
    def register_to_factory(cls, config: dict):
        """
        静态 names 或动态从 config 读取，统一注册到工厂。
        """
        if cls.dynamic and cls.type is not None:
            type_id = FactoryType.convert_to_text(cls.type)
            type_dict = config.get(type_id, {})
            for name, info in type_dict.items():
                GeneratorFactory.register_generator(cls.type, name, cls)
                tag = str(info.get("tag", ""))
                GeneratorFactory.register_model_info(cls.type, tag, name)
            cls._config = config   # 存下来供 __init__ 用
            cls.names = type_dict

        elif cls.names and not cls.dynamic:
            for name, info in cls.names.items():
                GeneratorFactory.register_generator(cls.type, name, cls)
                tag = str(info.get("tag", ""))
                GeneratorFactory.register_model_info(cls.type, tag, name)

    def __init__(self, model_name: str, device: str):
        self.pipe = None
        self.model_name = model_name

        type_id = FactoryType.convert_to_text(self.type)
        model_info = self._config.get(type_id, {}).get(self.model_name, None)
        if isinstance(model_info, dict):
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

    def ensure_model_loaded(self):
        """确保模型已加载（供外部调用）"""
        if self.pipe is None:
            self._check_model_file()
            self._load_model()

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
        return preloader.get("torch")

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


class GeneratorFactory:
    """生成器工厂，方便后续扩展模型"""
    _generators: Dict[FactoryType, Dict[str, type]] = {t: {} for t in FactoryType}
    _model: Dict[FactoryType, Dict[str, list]] = {t: {} for t in FactoryType}
    _device: str = "cpu"

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
        if cls._model[ty].get(tag):
            cls._model[ty][tag].append(name)
        else:
            cls._model[ty][tag] = [name]

    @classmethod
    def build_generator(cls, ty, name: str):
        if name not in cls._generators.get(ty):
            raise ValueError(f"未知的生成器: {name}")
        return cls._generators[ty][name](model_name=name, device=cls._device)

    @classmethod
    def get_generator_names(cls, ty) -> list:
        return list(cls._generators[ty].keys())

    @classmethod
    def get_model_info(cls, ty) -> dict[str, list]:
        return cls._model[ty]
