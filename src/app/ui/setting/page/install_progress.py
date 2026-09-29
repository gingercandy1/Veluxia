from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.library.activity import LoadingStrip

LOG_HEIGHT = 160
# 安装 torch 时 pip 会刷几千行，只留最近的，避免日志越积越卡
LOG_MAX_LINES = 2000


def scrollable_layout(page: QWidget) -> QVBoxLayout:
    """设置页的内容放进滚动区：窗口矮时整页可以滚动，而不是把下面的分组挤扁、切掉。"""
    outer = QVBoxLayout(page)
    outer.setContentsMargins(0, 0, 0, 0)
    scroll = QScrollArea()
    scroll.setObjectName("setting_scroll")
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content = QWidget()
    content.setObjectName("setting_scroll_content")
    scroll.setWidget(content)
    outer.addWidget(scroll)
    return QVBoxLayout(content)


class InstallProgress(QWidget):
    """安装进度：没开始安装时整块隐藏，不留一个空的黑框；
    安装中显示流光进度条和最新一行输出，完整日志在下面，安装结束后保留供查看。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        self.strip = LoadingStrip(height=3)
        layout.addWidget(self.strip)
        self.status_label = QLabel()
        self.status_label.setObjectName("install_status")
        self.status_label.setWordWrap(False)
        layout.addWidget(self.status_label)

        self.log = QPlainTextEdit()
        self.log.setObjectName("install_log")
        self.log.setReadOnly(True)
        self.log.setFixedHeight(LOG_HEIGHT)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        # 内边距放在文档上而不是 qss 的 padding：后者会把滚动条也往里推、盖住最后一列字
        self.log.document().setDocumentMargin(8)
        layout.addWidget(self.log)

        self.strip.set_loading(False)
        self.hide()

    def start(self, message: str):
        self.log.clear()
        self.show()
        self.strip.set_loading(True)
        self._set_status(message, "running")
        self.append(message)

    def append(self, line: str):
        self.show()
        self.log.appendPlainText(line)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        if self.strip.is_active() and line.strip():
            self._set_status(line.strip(), "running")

    def finish(self, ok: bool, message: str):
        self.strip.set_loading(False)
        self.append(("✓ " if ok else "✗ ") + message)
        self._set_status(message, "done" if ok else "error")

    def is_running(self) -> bool:
        return self.strip.is_active()

    def _set_status(self, text: str, state: str):
        metrics = self.status_label.fontMetrics()
        width = max(self.status_label.width(), 200)
        self.status_label.setText(metrics.elidedText(text, Qt.TextElideMode.ElideMiddle, width))
        self.status_label.setToolTip(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
