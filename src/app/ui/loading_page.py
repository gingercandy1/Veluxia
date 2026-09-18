from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar


class LoadingPage(QWidget):
    """启动加载页：后端进程与模型加载完成前展示，避免用户看到不可用的主界面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("loading_page")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(18)

        self._title = QLabel(self.tr("Veluxia"))
        self._title.setObjectName("loading_title")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress = QProgressBar()
        self._progress.setObjectName("loading_progress")
        self._progress.setFixedWidth(240)
        self._progress.setRange(0, 0)  # 不确定进度：跑马灯效果
        self._progress.setTextVisible(False)

        self._status = QLabel(self.tr("正在启动，请稍候…"))
        self._status.setObjectName("loading_status")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self._title)
        layout.addWidget(self._progress, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._status)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_error(self, text: str) -> None:
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._status.setObjectName("loading_status_error")
        self._status.setStyleSheet("color: #E5A5A5;")
        self._status.setText(text)
