"""从 src/app/ui/mark.py 的图形导出应用图标文件（改了图标几何后重跑一次）。

用法（项目根目录）：.venv\\Scripts\\python.exe script/build_icon.py
输出：resource/icons/veluxia.ico（exe 图标，16–256 多尺寸）、veluxia.png（1024，README 等用）
"""
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QBuffer, QIODevice  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.app.ui.mark import render_icon_pixmap  # noqa: E402

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
OUT_DIR = ROOT / "resource" / "icons"


def _to_pil(px_size: int) -> Image.Image:
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    render_icon_pixmap(px_size).save(buffer, "PNG")
    return Image.open(BytesIO(bytes(buffer.data()))).convert("RGBA")


def main() -> int:
    QApplication(sys.argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 每个尺寸都单独绘制再打包：让 Pillow 从 256 缩小的话，16/24px 会糊
    images = [_to_pil(size) for size in ICO_SIZES]
    ico_path = OUT_DIR / "veluxia.ico"
    images[-1].save(ico_path, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=images[:-1])
    png_path = OUT_DIR / "veluxia.png"
    _to_pil(1024).save(png_path)
    print(f"✅ {ico_path}")
    print(f"✅ {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
