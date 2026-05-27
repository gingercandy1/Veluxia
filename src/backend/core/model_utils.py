import importlib
import os
import tempfile
import threading
from typing import Any

from core.preloader import preloader

huggingface_token = ""

project_name = "material_generation"


class LazyModule:
    """异步预加载模块，访问属性时自动等待加载完成"""

    def __init__(self, module_name: str, timeout: float = 60):
        self._module_name = module_name
        self._timeout = timeout
        self._module = None
        self._ready = threading.Event()
        self._error = None

        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            self._module = importlib.import_module(self._module_name)
        except Exception as e:
            self._error = e
        finally:
            self._ready.set()

    def _wait(self):
        if not self._ready.is_set():
            import traceback
            print(f"\n[LazyModule] ⚠️  {self._module_name} 被同步阻塞等待！")
            print("调用位置：")
            traceback.print_stack(limit=10)
            print("-" * 50)

        self._ready.wait(timeout=self._timeout)
        if self._error:
            raise self._error
        if self._module is None:
            raise TimeoutError(f"Module {self._module_name} failed to load within {self._timeout}s")

    def __getattr__(self, name: str) -> Any:
        self._wait()
        return getattr(self._module, name)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._error is None

def get_temp_dir(output_dir_name):
    temp_dir = tempfile.gettempdir()
    output_dir = os.path.join(temp_dir, project_name, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_device(device="auto"):
    if device == "auto":
        torch = preloader.get("torch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("⚠️  未检测到 CUDA，将使用 CPU 推理（非常慢）")
        return device
    else:
        return device

def print_vram_usage():
    torch = preloader.get("torch")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"   GPU 显存：{used:.1f} GB / {total:.1f} GB")



