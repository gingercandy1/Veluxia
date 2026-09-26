"""Veluxia 打包统一入口（ADR 0002）。

用法：
  python script/package.py front          # 前端 exe（PyInstaller）
  python script/package.py backend        # 后端绿色 zip
  python script/package.py all            # 全部
  python script/package.py backend --out dist --no-zip   # 只要装配目录
  python script/package.py backend --bundle  # 发布包：含 python 运行时和 wheels

约定：src/ 单源；shared 构建时各捆一份；torch 只锁 cu128。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

import assemble_backend as ab


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Veluxia 打包：front | backend | all")
    ap.add_argument("flow", choices=("front", "backend", "all"))
    ap.add_argument("--out", default="dist", help="输出目录（相对 ROOT）")
    ap.add_argument("--no-zip", action="store_true", help="后端只装配不打 zip")
    ap.add_argument("--bundle", action="store_true",
                    help="后端放入 python 运行时和依赖 wheels（发布用，需要 uv 和网络）")
    return ap.parse_args(argv)


def build_frontend(root: Path, out: Path, runner=subprocess.run) -> Path:
    """跑 PyInstaller，返回 dist 目录。"""
    spec = root / "app" / "veluxia.spec"
    runner([sys.executable, "-m", "PyInstaller",
            "--distpath", str(out / "front"),
            "--workpath", str(out / "build-front"),
            str(spec)],
           cwd=str(root), check=True)
    return out / "front"


def build_backend(root: Path, out: Path, zip_it: bool = True, with_bundle: bool = False) -> Path:
    """装配后端目录，可选打 zip，返回装配目录。"""
    out.mkdir(parents=True, exist_ok=True)
    pkg = ab.assemble(out, with_bundle=with_bundle)
    if zip_it:
        archive = shutil.make_archive(
            str(out / "veluxia-backend"), "zip", root_dir=str(out),
            base_dir=pkg.name)
        print(f"✅ 后端 zip: {archive}")
    return pkg


def main(argv=None) -> int:
    args = parse_args(argv)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.flow in ("front", "all"):
        print("—— 前端 exe ——")
        build_frontend(ROOT, out)
    if args.flow in ("backend", "all"):
        print("—— 后端绿色 zip ——")
        build_backend(ROOT, out, zip_it=not args.no_zip, with_bundle=args.bundle)
    print(f"✅ 完成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
