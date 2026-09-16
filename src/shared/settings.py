import json
import os
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_URL = "http://127.0.0.1:8765"
SETTING_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "setting.json")

DEFAULT_CONFIG = {
    "general": {
        "language": "zh",
        "theme": "dark",
    },
    "gpu": {
        "backend": "cpu",
    },
    "translation": {
        "source": "auto",
        "target": "en",
    },
    "output": {
        "image_dir": "",
        "video_dir": "",
        "audio_dir": "",
    },
}


class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._config: dict = {}
        self._dirty: bool = False
        self.load()
        self._initialized = True

    def load(self):
        if os.path.exists(SETTING_CONFIG_PATH):
            try:
                with open(SETTING_CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                self._config = self._deep_merge(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"[ConfigManager] 配置加载失败，使用默认值: {e}")
                self._config = DEFAULT_CONFIG.copy()
        else:
            self._config = DEFAULT_CONFIG.copy()
            self.save()
        self._dirty = False

    def save(self):
        try:
            os.makedirs(os.path.dirname(SETTING_CONFIG_PATH), exist_ok=True)
            with open(SETTING_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(self._config, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception as e:
            print(f"[ConfigManager] 配置保存失败: {e}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._config.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any):
        if section not in self._config:
            self._config[section] = {}
        if self._config[section].get(key) != value:
            self._config[section][key] = value
            self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_clean(self):
        self._dirty = False

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_section(self, section: str) -> dict:
        """获取整个分组"""
        return self._config.get(section, {}).copy()

    def get_config(self):
        return self._config

    def get_backend_config(self) -> dict:
        """只返回后端需要的配置项，前端 UI 配置不透传"""
        return {
            "gpu": self.get_section("gpu"),
            "translation": self.get_section("translation")
        }