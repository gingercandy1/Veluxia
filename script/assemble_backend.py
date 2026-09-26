"""组装后端绿色 zip（ADR 0002 / 0005）。

目标布局 veluxia-backend/：
  python/                  ← python-build-standalone（uv 托管的 3.12，自带 pip、认 PYTHONPATH）
  wheels/                  ← 除 torch 三件套外的全部依赖 wheel（按 uv.lock 冻结）
  app/src/backend          ← 后端源码
  app/src/shared           ← 共享源码（构建时捆入）
  app/models.json
  backend.requirements.txt ← 离线装 wheels/ 的清单
  torch.requirements.txt   ← torch cu128，首次安装时联网装（体积超出 Release 单文件上限）
  install_backend.bat      ← 安装依赖（前端一键安装也调它）
  run_backend.bat          ← 未安装时先装，再起 :8765；额外参数透传（--host / --token）

用法：python script/assemble_backend.py [--out dist] [--bundle]
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765
BACKEND_MODULE = "src.backend.server"
PYTHON_VERSION = "3.12"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
# 单个 wheel 就接近 Release 2GB 上限，不进包，首次安装时联网装
TORCH_PACKAGES = ("torch", "torchvision", "torchaudio")
INHERITED_PYTHON_ENV = ("PYTHONHOME", "PYTHONPATH", "UV_INTERNAL__PYTHONHOME", "__PYVENV_LAUNCHER__")

# 打进 zip 的源码（相对 ROOT）；目录表示整棵复制（跳过 __pycache__）
SOURCE_PATHS = [
    "src/backend",
    "src/shared",
    "models.json",
]

# 体积巨大且独立安装的第三方，不进包（见 backend/pyproject 注释）
EXCLUDE_PREFIXES = (
    "src/backend/core/speech/ACE_Step",
)


def build_manifest(root: Path) -> list:
    """返回应打进包的相对路径清单（/ 分隔），供装配与单测共用。"""
    out: list = []
    for rel in SOURCE_PATHS:
        p = root / rel
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and "__pycache__" not in f.parts:
                    posix = f.relative_to(root).as_posix()
                    if posix.startswith(EXCLUDE_PREFIXES):
                        continue
                    out.append(posix)
        elif p.is_file():
            out.append(rel)
    return out


def write_installer(pkg_dir: Path) -> Path:
    """生成 install_backend.bat，返回路径。

    lock 导出的清单已含全部传递依赖，所以一律 --no-deps：
    否则 sentence-transformers 等会在离线阶段去找 torch。
    .bat 只写 ASCII（cmd 按 GBK 解析 UTF-8 中文会拆坏命令行）。
    """
    bat = pkg_dir / "install_backend.bat"
    bat.write_text(
        "@echo off\n"
        "cd /d %~dp0\n"
        "if not exist python\\python.exe (\n"
        "    echo [backend] python runtime is missing: python\\python.exe\n"
        "    exit /b 1\n"
        ")\n"
        "echo [backend] installing bundled wheels ...\n"
        "python\\python.exe -m pip install --no-index --no-deps --find-links wheels"
        " -r backend.requirements.txt\n"
        "if errorlevel 1 exit /b 1\n"
        "echo [backend] downloading torch (cu128), this may take a while ...\n"
        f"python\\python.exe -m pip install --no-deps --index-url {TORCH_INDEX}"
        " -r torch.requirements.txt\n"
        "if errorlevel 1 exit /b 1\n"
        "echo ok> .installed\n"
        "echo [backend] install finished\n",
        encoding="ascii",
    )
    return bat


def write_launcher(pkg_dir: Path, port: int = DEFAULT_PORT) -> Path:
    """生成 run_backend.bat，返回路径。"""
    bat = pkg_dir / "run_backend.bat"
    bat.write_text(
        "@echo off\n"
        "cd /d %~dp0\n"
        "set PYTHONPATH=%CD%\\app\n"
        "if not exist .installed (\n"
        "    call install_backend.bat\n"
        "    if errorlevel 1 exit /b 1\n"
        ")\n"
        # %* 放在默认端口之后，--port / --host / --token 可被调用方覆盖
        f"python\\python.exe -m {BACKEND_MODULE} --port {port} %*\n",
        encoding="ascii",
    )
    return bat


def split_requirements(text: str) -> tuple[list[str], list[str]]:
    """把 uv export 的清单拆成（离线 wheels，torch 联网装）两份。"""
    base, torch = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("==", 1)[0].split(" ", 1)[0].split(";", 1)[0].strip().lower()
        (torch if name in TORCH_PACKAGES else base).append(line)
    return base, torch


def export_requirements(root: Path) -> str:
    """从 uv.lock 导出后端全部依赖（含传递依赖，版本钉死）。"""
    result = subprocess.run(
        ["uv", "export", "--frozen", "--package", "veluxia-backend",
         "--no-hashes", "--no-annotate", "--no-header",
         "--no-emit-project", "--no-emit-workspace"],
        cwd=str(root), check=True, capture_output=True, text=True,
    )
    return result.stdout


def find_python_runtime() -> Path:
    """返回 uv 托管的 python-build-standalone 根目录（不是 .venv）。"""
    subprocess.run(["uv", "python", "install", PYTHON_VERSION], check=True)
    result = subprocess.run(
        ["uv", "python", "find", PYTHON_VERSION, "--managed-python", "--no-project", "--system"],
        check=True, capture_output=True, text=True,
    )
    return Path(result.stdout.strip()).parent


def copy_python_runtime(runtime: Path, dst: Path) -> Path:
    """复制运行时，并去掉 uv 的 EXTERNALLY-MANAGED 标记，否则 pip install 会被拒绝。"""
    shutil.copytree(runtime, dst, ignore=shutil.ignore_patterns("__pycache__"))
    (dst / "Lib" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)
    return dst / "python.exe"


def build_wheels(python: Path, requirements: Path, wheels_dir: Path) -> None:
    """用包内同版本 python 预构建 wheel；sdist-only 的依赖也会被编成 wheel，离线可装。"""
    # 从 uv venv 里启动时会继承 PYTHONHOME 等变量，子进程会误用原解释器（并被 EXTERNALLY-MANAGED 拒绝）
    env = {k: v for k, v in os.environ.items() if k not in INHERITED_PYTHON_ENV}
    subprocess.run(
        [str(python), "-m", "pip", "wheel", "--no-deps",
         "-r", str(requirements), "-w", str(wheels_dir)],
        check=True, env=env,
    )


def bundle(pkg: Path, root: Path = ROOT) -> None:
    """放入运行时、依赖清单和 wheels，使 zip 解压即可安装运行。"""
    base, torch = split_requirements(export_requirements(root))
    if not torch:
        raise RuntimeError("uv export 里没有 torch，检查 veluxia-backend 依赖")
    req = pkg / "backend.requirements.txt"
    req.write_text("\n".join(base) + "\n", encoding="utf-8")
    (pkg / "torch.requirements.txt").write_text("\n".join(torch) + "\n", encoding="utf-8")

    shutil.rmtree(pkg / "python", ignore_errors=True)
    python = copy_python_runtime(find_python_runtime(), pkg / "python")
    build_wheels(python, req, pkg / "wheels")


def assemble(out_dir: Path, with_bundle: bool = False) -> Path:
    pkg = out_dir / "veluxia-backend"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "app").mkdir(parents=True)
    (pkg / "wheels").mkdir(parents=True)

    for rel in build_manifest(ROOT):
        src, dst = ROOT / rel, pkg / "app" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    write_installer(pkg)
    write_launcher(pkg)
    if with_bundle:
        bundle(pkg)
    print(f"✅ 装配完成: {pkg}")
    if not with_bundle:
        print("⚠️ 未带 --bundle：包内没有 python 运行时和 wheels，仅用于检查源码布局。")
    return pkg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist", help="输出目录")
    ap.add_argument("--bundle", action="store_true", help="放入 python 运行时和依赖 wheels")
    args = ap.parse_args(argv)
    assemble(ROOT / args.out, with_bundle=args.bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
