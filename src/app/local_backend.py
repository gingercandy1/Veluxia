"""本机后端的安装与启动（ADR 0005）。

前端打包成 exe 后自身没有后端所需的 Python，改为从安装目录（Release 里的 veluxia-backend.zip 解压而来）
拉起后端。源码运行时仍用当前解释器，开发流程不变。所有函数都是阻塞的，须在 Worker 线程里调用。
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import httpx

from src.shared.settings import ConfigManager

RELEASE_URL = (
    "https://github.com/gingercandy1/Veluxia/releases/latest/download/veluxia-backend.zip"
)
BACKEND_MODULE = "src.backend.server"
PACKAGE_DIR_NAME = "veluxia-backend"  # zip 内的顶层目录
INSTALLED_MARKER = ".installed"       # install_backend.bat 成功后写入
# 从 venv / 其他 Python 环境启动时继承这些变量，会让包内解释器误用别处的标准库和 site-packages
INHERITED_PYTHON_ENV = ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "__PYVENV_LAUNCHER__")
_DOWNLOAD_CHUNK = 1024 * 1024

ProgressCallback = Callable[[str], None]


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def default_install_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Veluxia" / "backend"


def install_dir() -> Path:
    configured = ConfigManager().get("server", "install_dir", "")
    return Path(configured) if configured else default_install_dir()


def is_installed(root: Path) -> bool:
    return (root / INSTALLED_MARKER).exists() and (root / "python" / "python.exe").exists()


def _clean_env(root: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in INHERITED_PYTHON_ENV}
    env["PYTHONPATH"] = str(root / "app")
    # 让 pip / 后端日志按 UTF-8 输出，前端解码时才不会乱码
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def backend_command(root: Path, port: int) -> list[str]:
    return [str(root / "python" / "python.exe"), "-m", BACKEND_MODULE, "--port", str(port)]


def start(root: Path, port: int) -> subprocess.Popen:
    """启动已安装的本机后端，只监听 127.0.0.1，不需要 token。"""
    return subprocess.Popen(backend_command(root, port), cwd=str(root), env=_clean_env(root))


def download(url: str, dest: Path, on_progress: ProgressCallback) -> Path:
    """下载到 .part 临时文件，完成后再改名，避免中断后留下看似完整的 zip。"""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done, last_step = 0, -1
        with open(part, "wb") as f:
            for chunk in resp.iter_bytes(_DOWNLOAD_CHUNK):
                f.write(chunk)
                done += len(chunk)
                # 每 5% 报一次；拿不到总大小时每 50MB 报一次
                step = done * 20 // total if total else done // (50 * _DOWNLOAD_CHUNK)
                if step != last_step:
                    last_step = step
                    on_progress(f"⬇️ {done // _DOWNLOAD_CHUNK} / {total // _DOWNLOAD_CHUNK} MB")
    part.replace(dest)
    return dest


def extract(archive: Path, root: Path, on_progress: ProgressCallback) -> None:
    """解压并覆盖到安装目录。

    只覆盖不清空：安装目录的 app/ 下还有用户的模型权重、生成结果和后端设置，重装不能丢。
    """
    on_progress(f"📦 {archive.name} → {root}")
    root.mkdir(parents=True, exist_ok=True)
    # 先去掉标记，解压或安装中途失败时下次启动会提示重新安装，而不是拉起半装好的后端
    (root / INSTALLED_MARKER).unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as staging:
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
        package = Path(staging) / PACKAGE_DIR_NAME
        if not (package / "install_backend.bat").exists():
            raise RuntimeError(f"不是有效的后端安装包（缺少 {PACKAGE_DIR_NAME}/install_backend.bat）")
        shutil.copytree(package, root, dirs_exist_ok=True)


def run_installer(root: Path, on_progress: ProgressCallback) -> None:
    """运行包内 install_backend.bat（离线装 wheels，联网装 torch），逐行回传日志。"""
    proc = subprocess.Popen(
        ["cmd", "/c", str(root / "install_backend.bat")],
        cwd=str(root), env=_clean_env(root),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    for raw in proc.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            on_progress(line)
    if proc.wait() != 0:
        raise RuntimeError(f"install_backend.bat 失败（退出码 {proc.returncode}）")


def install(root: Path, on_progress: ProgressCallback, archive: Path | None = None) -> None:
    """安装本机后端：archive 为空时从 GitHub Release 下载最新版，否则用本地 zip。"""
    downloaded = None
    if archive is None:
        on_progress(f"⬇️ {RELEASE_URL}")
        downloaded = download(RELEASE_URL, root.parent / f"{PACKAGE_DIR_NAME}.zip", on_progress)
        archive = downloaded
    extract(archive, root, on_progress)
    run_installer(root, on_progress)
    # 只删自己下载的 zip，用户选的本地文件保留
    if downloaded is not None:
        downloaded.unlink(missing_ok=True)
