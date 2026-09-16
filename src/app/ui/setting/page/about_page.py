from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame
)

APP_VERSION = "1.0.0"
APP_NAME    = "Veluxia"
COPYRIGHT   = "© 2025 Veluxia Team. All rights reserved."
DESCRIPTION = "AI 驱动的多模态创作工具，支持文本、图像、动画、语音生成。"

class AboutPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("about_page")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Logo
        self._logo = QLabel()
        self._logo.setObjectName("about_Logo")
        self._logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._logo.setFixedSize(96, 96)
        self._try_load_logo()
        layout.addWidget(self._logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        # 软件名
        name_label = QLabel(APP_NAME)
        name_label.setObjectName("about_app_name")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        # 版本号
        version_label = QLabel(f"版本  {APP_VERSION}")
        version_label.setObjectName("about_version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("about_separator")
        layout.addWidget(sep)

        # 简介
        desc_label = QLabel(DESCRIPTION)
        desc_label.setObjectName("about_description")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        layout.addStretch()

        # 版权
        copy_label = QLabel(COPYRIGHT)
        copy_label.setObjectName("about_copy_right")
        copy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copy_label)

    def _try_load_logo(self):
        """尝试加载 logo，失败则显示占位文字"""
        import os
        from src.shared.settings import PROJECT_ROOT
        logo_path = os.path.join(PROJECT_ROOT, "src", "app", "resources", "logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path).scaled(
                96, 96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._logo.setPixmap(pixmap)
        else:
            self._logo.setText("🎨")
            self._logo.setStyleSheet("font-size: 48px;")
