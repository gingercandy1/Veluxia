from PySide6.QtCore import Qt, QT_TRANSLATE_NOOP
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QFrame
)

from src.app.ui.mark import render_icon_pixmap

APP_NAME    = "Veluxia"
COPYRIGHT   = "© 2026 Veluxia"
TAGLINE     = QT_TRANSLATE_NOOP("AboutPage", "Local AI studio for game assets")
DESCRIPTION = QT_TRANSLATE_NOOP(
    "AboutPage",
    "Generate text, images, animation, voice, music and sound effects entirely on your own "
    "computer. Private, free to run, and optimized for 8 GB GPUs.")
LICENSE_NOTE = QT_TRANSLATE_NOOP(
    "AboutPage",
    "Released under the MIT License. AI models are subject to their own licenses.")

LOGO_SIZE = 96


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

        self._logo = QLabel()
        self._logo.setObjectName("about_logo")
        self._logo.setFixedSize(LOGO_SIZE, LOGO_SIZE)
        self._logo.setPixmap(self._logo_pixmap())
        layout.addWidget(self._logo, alignment=Qt.AlignmentFlag.AlignHCenter)

        name_label = QLabel(APP_NAME)
        name_label.setObjectName("about_app_name")
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(name_label)

        tagline_label = QLabel(self.tr(TAGLINE))
        tagline_label.setObjectName("about_tagline")
        tagline_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(tagline_label)

        # 版本号统一取 Application 里设置的那个，避免两处写死不一致
        version_label = QLabel(self.tr("Version") + f"  {QApplication.applicationVersion()}")
        version_label.setObjectName("about_version")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("about_separator")
        layout.addWidget(sep)

        desc_label = QLabel(self.tr(DESCRIPTION))
        desc_label.setObjectName("about_description")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        layout.addStretch()

        license_label = QLabel(self.tr(LICENSE_NOTE))
        license_label.setObjectName("about_copy_right")
        license_label.setWordWrap(True)
        license_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license_label)

        copy_label = QLabel(COPYRIGHT)
        copy_label.setObjectName("about_copy_right")
        copy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copy_label)

    def _logo_pixmap(self):
        # 按屏幕缩放比渲染，高 DPI 下图标不糊
        ratio = self.devicePixelRatioF()
        pixmap = render_icon_pixmap(round(LOGO_SIZE * ratio))
        pixmap.setDevicePixelRatio(ratio)
        return pixmap
