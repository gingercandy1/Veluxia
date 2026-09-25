from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from src.app.ui.mark import MarkWidget


class LoadingPage(QWidget):
    """启动加载页：后端进程与模型加载完成前展示，避免用户看到不可用的主界面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loading_page")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        # 后端启动常要几十秒，动画一直转着才不像卡死
        self._mark = MarkWidget(color=QColor("#F2F3F5"), fill_ratio=0.72, loop=True)
        self._mark.setFixedSize(160, 160)

        self._title = QLabel(self.tr("Veluxia"))
        self._title.setObjectName("loading_title")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._status = QLabel(self.tr("Starting, please wait…"))
        self._status.setObjectName("loading_status")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._mark, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._title)
        layout.addWidget(self._status)

    def showEvent(self, event):
        super().showEvent(event)
        self._mark.play()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._mark.stop()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_error(self, text: str) -> None:
        self._mark.show_final_frame()
        self._status.setObjectName("loading_status_error")
        self._status.setStyleSheet("color: #E5A5A5;")
        self._status.setText(text)
