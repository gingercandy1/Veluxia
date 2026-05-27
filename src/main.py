import os
import sys
from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from ui.window import MainWindow
from ui.setting.page.log_page import log_info, log_error


class Application(QApplication):
    """
    自定义 QApplication，负责：
    - 加载 QSS 样式表
    - 设置全局字体
    - 高 DPI 配置
    - 运行时热重载 QSS（开发模式）
    """
    QSS_DIR_PATH = Path(__file__).parent.parent / "resource" / "qss"

    def __init__(self, argv: list[str]):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        super().__init__(argv)
        self.setApplicationName(self.tr("Material Generation"))
        self.setOrganizationName("YourOrg")
        self.setApplicationVersion("1.0.0")

        self._setup_font()
        self._load_qss()

    def _setup_font(self):
        """
        按平台优先级设置 UI 字体。
        Windows → Segue UI / Microsoft YaHei UI
        macOS   → PingFang SC / SF Pro
        Linux   → Noto Sans
        """
        candidates = [
            "Segoe UI",
            "Microsoft YaHei UI",
            "PingFang SC",
            "Noto Sans",
            "sans-serif",
        ]
        available = QFontDatabase.families()
        chosen = next((f for f in candidates if f in available), "sans-serif")

        font = QFont(chosen, 13)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)

    def _load_qss(self, path: Path | List[Path] | None = None) -> bool:
        """
        从文件加载 QSS 并应用到整个应用。
        返回是否成功。
        """
        target = path
        if path is None:
            target = []
            for i in os.listdir(self.QSS_DIR_PATH):
                i_path = Path(os.path.join(self.QSS_DIR_PATH, i))
                target.append(i_path)

        if path and not target.exists():
            log_error(f"[Application] QSS 文件不存在：{target}")
            return False

        try:
            if isinstance(target, list):
                qss = ""
                for path in target:
                    qss += path.read_text(encoding="utf-8")
            else:
                qss = target.read_text(encoding="utf-8")
            self.setStyleSheet(qss)
            log_info(f"[Application] QSS 已加载")
            return True
        except Exception as e:
            log_error(f"[Application] 加载 QSS 失败：{e}")
            return False

    def reload_qss(self) -> bool:
        """
        热重载 QSS（绑定到快捷键使用，开发时无需重启）。
        用法：在 MainWindow 里按 Ctrl+Shift+R 调用 app.reload_qss()
        """
        ok = self._load_qss()
        if ok:
            # 强制所有顶层窗口重新应用样式
            for widget in self.topLevelWidgets():
                widget.style().unpolish(widget)
                widget.style().polish(widget)
                widget.update()
        return ok


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    os.environ["CUDA_PATH"] = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"

    app = Application(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
