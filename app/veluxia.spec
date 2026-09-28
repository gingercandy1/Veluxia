# PyInstaller spec：Veluxia 前端（ADR 0002）
# 用法（root 下）：pyinstaller app/veluxia.spec
# 说明：后端由 exe 启动后经 work.py::start_backend 拉起，不打进 exe。

import os
from pathlib import Path

# SPECPATH 是 spec 所在目录（app/），不是 spec 文件本身，上一级才是项目根
ROOT = Path(os.path.abspath(SPECPATH)).parent

a = Analysis(
    [str(ROOT / "src" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "resource"), "resource"),
        (str(ROOT / "models.json"), "."),
        (str(ROOT / "config" / "setting.json"), "config"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "diffusers", "transformers", "accelerate"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="veluxia-app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "resource" / "icons" / "veluxia.ico"),  # 由 script/build_icon.py 生成
)
