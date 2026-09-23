import importlib
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.shared.settings import PROJECT_ROOT

# 不在源码里存 token：留空/None 时 huggingface_hub 会自动读 HF_TOKEN 环境变量，
# 或 `huggingface-cli login` 缓存的登录态，gated 仓库照样能下。
huggingface_token = os.environ.get("HF_TOKEN")

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

class _ProgressProxy:
    """
    代替 huggingface_hub 那条 tqdm 交给下载代码，下载代码只会调用 update()。

    自己累计字节数，不读 tqdm 的 n：进度条在非终端环境（我们这种子进程）里是
    禁用状态，禁用的 tqdm update() 直接返回、根本不累加，读它永远是 0。
    """

    def __init__(self, bar, total: int, callback):
        self._bar = bar
        self._total = total
        self._callback = callback
        self.n = getattr(bar, "n", 0) or 0

    def update(self, n=1):
        self.n += int(n or 0)
        try:
            self._callback(self.n, self._total)
        except Exception:
            pass  # 进度只是展示用，回调出错不能影响下载本身
        return self._bar.update(n)

    def __getattr__(self, name):
        return getattr(self._bar, name)


@contextmanager
def hf_download_progress(callback):
    """
    让 hf_hub_download / snapshot_download 的下载进度可以被前端看到。

    huggingface_hub 没有提供进度回调，只有内部那条 tqdm 进度条；这里临时接管它的
    创建过程，把拿到的进度条对象的 update 包一层，每次更新顺带回调一次
    (已下载字节, 总字节)。退出时一定还原，免得影响其它地方的进度条。
    """
    from huggingface_hub import file_download

    original = getattr(file_download, "_get_progress_bar_context", None)
    if original is None:  # 版本不匹配：没有进度就没有，别把下载本身搞挂
        yield
        return

    @contextmanager
    def _reporting_context(**kwargs):
        with original(**kwargs) as bar:
            yield _ProgressProxy(bar, kwargs.get("total") or 0, callback)

    file_download._get_progress_bar_context = _reporting_context
    try:
        yield
    finally:
        file_download._get_progress_bar_context = original


def get_media_root() -> Path:
    """生成素材的持久化存储根目录（随项目安装位置，不会被系统清理临时文件时删除）。"""
    root = Path(PROJECT_ROOT) / "output" / project_name
    root.mkdir(parents=True, exist_ok=True)
    return root

def get_temp_dir(output_dir_name):
    output_dir = get_media_root() / output_dir_name if output_dir_name else get_media_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)

def to_media_url(path) -> str:
    """把生成文件的本地绝对路径转换成前端可通过 /media 静态路由访问的相对 URL。"""
    p = Path(path).resolve()
    root = get_media_root().resolve()
    try:
        rel = p.relative_to(root)
    except ValueError:
        # 不在媒体根目录下（不应发生），退化为原始路径
        return str(p)
    return "/media/" + rel.as_posix()

def get_device(device="auto"):
    if device == "auto":
        import torch  # 后台注册线程早已 import 过，这里只是拿缓存，不会重新触发加载
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("⚠️  未检测到 CUDA，将使用 CPU 推理（非常慢）")
        return device
    else:
        return device

def print_vram_usage():
    import torch
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"   GPU 显存：{used:.1f} GB / {total:.1f} GB")



