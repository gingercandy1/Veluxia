import faulthandler
import os
import sys
from pathlib import Path
from typing import List

# 放在界面库导入之前：导入 PySide6 等本身就要几秒，这行能让人知道程序已经在动了
if __name__ == "__main__":
    print("[startup] 正在加载界面模块…", flush=True)
    # 30 秒窗口还没出来就把每个线程卡在哪一行打到控制台，卡住时直接看得出原因
    STARTUP_HANG_DUMP_SECONDS = 30
    faulthandler.dump_traceback_later(STARTUP_HANG_DUMP_SECONDS)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QStyleFactory

from src.app.i18n import DEFAULT_LANGUAGE, install_language
from src.app.ui.window import MainWindow
from src.app.ui.mark import build_app_icon
from src.app.ui.setting.page.log_page import log_info, log_error
from src.app.work import startup_log
from src.shared.settings import ConfigManager


class Application(QApplication):
    """
    自定义 QApplication，负责：
    - 加载 QSS 样式表
    - 设置全局字体
    - 高 DPI 配置
    - 运行时热重载 QSS（开发模式）
    """
    QSS_DIR_PATH = Path(__file__).parent.parent / "resource" / "qss"
    ROBOTO_FONT_PATH = Path(__file__).parent.parent / "resource" / "fonts" / "Roboto.ttf"

    def __init__(self, argv: list[str]):
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

        super().__init__(argv)
        # 必须在创建任何界面之前装好翻译器：各控件的 tr() 只在构造时求值一次
        self._setup_language()
        # 应用名是 QStandardPaths / QSettings 的目录标识，不能随界面语言变化，所以不翻译
        self.setApplicationName("Material Generation")
        self.setOrganizationName("YourOrg")
        self.setApplicationVersion("1.0.0")

        # Windows 原生（windowsvista）风格会用系统主题绘制部分控件（如 QSpinBox 的上下箭头），
        # 忽略我们 QSS 里自定义的 ::up-arrow/::down-arrow，在暗色主题下就变成了实心方块。
        # 换成 Fusion 后所有控件都完全按 QSS 绘制，样式表才能生效。
        self.setStyle(QStyleFactory.create("Fusion"))

        self._setup_font()
        self._load_qss()

    def _setup_language(self):
        code = ConfigManager().get("general", "language", DEFAULT_LANGUAGE)
        for problem in install_language(self, code):
            log_error(f"[Application] {problem}")

    def _setup_font(self):
        """
        英文/数字走随包的 Roboto Regular（resource/fonts/，OFL 协议可商用），
        中文走系统自带字体。用 setFamilies 声明完整回退链，而不是只选一个
        家族——单选家族时，中文字符会退回 Qt/系统挑的默认字体（往往是宋体），
        跟主字体风格不搭。
        Windows → Microsoft YaHei UI
        macOS   → PingFang SC
        Linux   → Noto Sans
        """
        roboto_family = self._load_bundled_roboto()

        system_candidates = [
            "Microsoft YaHei UI",
            "PingFang SC",
            "Noto Sans",
            "sans-serif",
        ]
        available = QFontDatabase.families()
        chain = ([roboto_family] if roboto_family else []) + \
                [f for f in system_candidates if f in available]
        if not chain:
            chain = ["sans-serif"]

        font = QFont(chain[0], 13)
        font.setFamilies(chain)
        font.setWeight(QFont.Weight.Normal)
        font.setHintingPreference(QFont.HintingPreference.PreferDefaultHinting)
        self.setFont(font)

    def _load_bundled_roboto(self) -> str | None:
        if not self.ROBOTO_FONT_PATH.exists():
            return None
        font_id = QFontDatabase.addApplicationFont(str(self.ROBOTO_FONT_PATH))
        if font_id == -1:
            log_error(f"[Application] 字体加载失败：{self.ROBOTO_FONT_PATH}")
            return None
        families = QFontDatabase.applicationFontFamilies(font_id)
        return families[0] if families else None

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
    startup_log("界面模块已加载")

    app = Application(sys.argv)
    app_icon = build_app_icon()
    app.setWindowIcon(app_icon)

    startup_log("正在创建主窗口…")
    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    faulthandler.cancel_dump_traceback_later()
    startup_log("窗口已显示，后端在后台启动")
    sys.exit(app.exec())
