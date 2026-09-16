import logging
from datetime import datetime
from enum import Enum

from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QLabel, QComboBox
)


class LogLevel(Enum):
    DEBUG   = ("DEBUG",   "#888888")
    INFO    = ("INFO",    "#d4d4d4")
    SUCCESS = ("SUCCESS", "#4ec994")
    WARNING = ("WARNING", "#e5c07b")
    ERROR   = ("ERROR",   "#e06c75")
    SYSTEM  = ("SYSTEM",  "#61afef")


class _LogSignals(QObject):
    """跨线程信号，后台线程也可以安全调用"""
    append = Signal(str, str, str)  # level_name, color, message


class LogPanel(QWidget):
    """
    全局日志面板，支持：
    - 不同级别颜色区分
    - 级别过滤
    - 清空 / 复制
    - 跨线程安全写入
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parent=None):
        if hasattr(self, "_initialized"):
            return
        super().__init__(parent)
        self._initialized = True
        self._signals = _LogSignals()
        self._signals.append.connect(self._append_to_view)
        self._min_level = LogLevel.DEBUG
        self._build_ui()
        self._setup_python_logging()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        toolbar.addWidget(QLabel("日志级别："))
        self._level_combo = QComboBox()
        self._level_combo.addItems([lv.value[0] for lv in LogLevel])
        self._level_combo.currentTextChanged.connect(self._on_level_changed)
        toolbar.addWidget(self._level_combo)

        toolbar.addStretch()

        self._clear_btn = QPushButton("清空")
        self._clear_btn.setFixedWidth(56)
        self._clear_btn.clicked.connect(self._text_view.clear
                                        if hasattr(self, "_text_view") else lambda: None)
        toolbar.addWidget(self._clear_btn)

        self._copy_btn = QPushButton("复制全部")
        self._copy_btn.setFixedWidth(72)
        self._copy_btn.clicked.connect(self._copy_all)
        toolbar.addWidget(self._copy_btn)

        layout.addLayout(toolbar)

        # 日志文本区
        self._text_view = QTextEdit()
        self._text_view.setReadOnly(True)
        self._text_view.setObjectName("log_text_view")
        self._text_view.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self._text_view)

        # 修正 clear_btn 的 connect（build_ui 里 _text_view 还未创建）
        self._clear_btn.clicked.disconnect()
        self._clear_btn.clicked.connect(self._text_view.clear)

    # ── 写入 ──────────────────────────────────────────────────────────

    def _append_to_view(self, level_name: str, color: str, message: str):
        """主线程槽，安全操作 QTextEdit"""
        cursor = self._text_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)

        ts = datetime.now().strftime("%H:%M:%S")
        cursor.insertText(f"[{ts}] [{level_name}] {message}\n")

        # 自动滚到底部
        self._text_view.setTextCursor(cursor)
        self._text_view.ensureCursorVisible()

    # ── 公开日志方法 ──────────────────────────────────────────────────

    def _log(self, level: LogLevel, message: str):
        if level.value[0] < self._min_level.value[0]:
            return
        self._signals.append.emit(level.value[0], level.value[1], message)

    def debug(self, message: str):
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str):
        self._log(LogLevel.INFO, message)

    def success(self, message: str):
        self._log(LogLevel.SUCCESS, message)

    def warning(self, message: str):
        self._log(LogLevel.WARNING, message)

    def error(self, message: str):
        self._log(LogLevel.ERROR, message)

    def system(self, message: str):
        self._log(LogLevel.SYSTEM, message)

    # ── 过滤 ──────────────────────────────────────────────────────────

    def _on_level_changed(self, level_name: str):
        for lv in LogLevel:
            if lv.value[0] == level_name:
                self._min_level = lv
                break

    # ── 工具 ──────────────────────────────────────────────────────────

    def _copy_all(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self._text_view.toPlainText())

    # ── 接管 Python logging ───────────────────────────────────────────

    def _setup_python_logging(self):
        """把标准 logging 也接入 LogPanel"""
        handler = _PanelHandler(self)
        handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.DEBUG)


class _PanelHandler(logging.Handler):
    """把 Python logging 转发到 LogPanel"""

    def __init__(self, panel: LogPanel):
        super().__init__()
        self._panel = panel

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        level = record.levelno
        if level >= logging.ERROR:
            self._panel.error(msg)
        elif level >= logging.WARNING:
            self._panel.warning(msg)
        elif level >= logging.INFO:
            self._panel.info(msg)
        else:
            self._panel.debug(msg)


# ── 全局快捷函数 ──────────────────────────────────────────────────────
def get_log_panel() -> LogPanel:
    """获取全局 LogPanel 单例"""
    return LogPanel()


def log_info(msg: str):    LogPanel().info(msg)
def log_success(msg: str): LogPanel().success(msg)
def log_warning(msg: str): LogPanel().warning(msg)
def log_error(msg: str):   LogPanel().error(msg)
def log_system(msg: str):  LogPanel().system(msg)
def log_debug(msg: str):   LogPanel().debug(msg)