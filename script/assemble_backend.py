"""组装后端绿色 zip（ADR 0002）。

目标布局 veluxia-backend/：
  python/          ← 手动放入 cpython embed-amd64 解包内容
  wheels/          ← uv pip download --dest wheels -r backend/requirements.frozen
  app/src/backend  ← 后端源码
  app/src/shared   ← 共享源码（构建时捆入）
  app/models.json
  run_backend.bat  ← 首次运行装 wheels，以后直起 :8765

用法：python script/assemble_backend.py [--out dist]
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8765
BACKEND_MODULE = "src.backend.server"

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


def write_launcher(pkg_dir: Path, port: int = DEFAULT_PORT) -> Path:
    """生成 run_backend.bat，返回路径。"""
    bat = pkg_dir / "run_backend.bat"
    bat.write_text(
        "@echo off\n"
        "cd /d %~dp0\n"
        "set PYTHONPATH=%CD%\\app\n"
        "if not exist python\\python.exe (\n"
        "    echo [backend] python\\ 目录缺 embed python，请先解包放入。\n"
        "    exit /b 1\n"
        ")\n"
        "if not exist .installed (\n"
        "    echo [backend] 首次运行，安装 wheels ...\n"
        "    python\\python.exe -m pip install --no-index --find-links wheels -r backend.requirements.txt\n"
        "    if errorlevel 1 exit /b 1\n"
        "    echo ok > .installed\n"
        ")\n"
        f"python\\python.exe -m {BACKEND_MODULE} --port {port}\n",
        encoding="utf-8",
    )
    return bat


def write_backend_requirements(pkg_dir: Path) -> Path:
    """从 backend/pyproject 导出冻结写法提示（真冻结靠 uv lock，人审后写入）。"""
    req = pkg_dir / "backend.requirements.txt"
    req.write_text(
        "# 由 uv pip compile backend/pyproject.toml 生成（cu128）。\n"
        "# 另需独立安装（与 torch 主线版本互斥，不进 lock）：\n"
        "#   uv pip install -r src/backend/core/speech/ACE_Step/requirements.txt\n",
        encoding="utf-8",
    )
    return req


def assemble(out_dir: Path) -> Path:
    pkg = out_dir / "veluxia-backend"
    if pkg.exists():
        shutil.rmtree(pkg)
    (pkg / "app").mkdir(parents=True)
    (pkg / "wheels").mkdir(parents=True)

    for rel in build_manifest(ROOT):
        src, dst = ROOT / rel, pkg / "app" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    write_launcher(pkg)
    write_backend_requirements(pkg)
    (pkg / "python").mkdir(exist_ok=True)
    (pkg / "python" / "README.txt").write_text(
        "将 cpython-3.12.x-embed-amd64.zip 解包到本目录。\n", encoding="utf-8")
    print(f"✅ 装配完成: {pkg}")
    print("后续人工步骤: 下载 wheels 到 wheels/，放入 embed python，打 zip。")
    return pkg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist", help="输出目录")
    args = ap.parse_args(argv)
    assemble(ROOT / args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
