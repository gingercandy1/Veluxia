from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QWidget


class ImagePreviewOverlay(QWidget):
    """点击缩略图后铺满宿主窗口的看图浮层，类似网页里的 lightbox：
    暗色背景 + 居中大图，点击任意位置或按 Esc 关闭。"""

    MAX_FRACTION = 0.88  # 相对宿主窗口尺寸的最大占比，给四周留出呼吸空间

    def __init__(self, path: str, host: QWidget):
        super().__init__(host)
        self.setObjectName("image_preview_overlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("#image_preview_overlay { background-color: rgba(6, 7, 9, 0.86); }")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._close_btn = QPushButton("✕", self)
        self._close_btn.setFixedSize(32, 32)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setStyleSheet("""
            QPushButton {
                color: rgba(255,255,255,0.75);
                background: rgba(255,255,255,0.08);
                border: none;
                border-radius: 16px;
                font-size: 14px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.16); color: #FFFFFF; }
        """)
        self._close_btn.clicked.connect(self.close)

        self._host = host
        self._load_pixmap(path)

        host.installEventFilter(self)
        self.setGeometry(host.rect())
        self._layout_children()
        self.raise_()
        self.show()
        self.setFocus()

    def _load_pixmap(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._label.setText("⚠️ 图片加载失败")
            self._label.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 14px;")
            self._source_pixmap = None
        else:
            self._source_pixmap = pix
            self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self):
        if self._source_pixmap is None:
            return
        bound = self._host.size()
        max_w = max(1, int(bound.width() * self.MAX_FRACTION))
        max_h = max(1, int(bound.height() * self.MAX_FRACTION))
        scaled = self._source_pixmap.scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def _layout_children(self):
        self._apply_scaled_pixmap()
        self._label.move(
            (self.width() - self._label.width()) // 2,
            (self.height() - self._label.height()) // 2,
        )
        self._close_btn.move(self.width() - self._close_btn.width() - 16, 16)

    def eventFilter(self, obj, event):
        if obj is self._host and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._host.rect())
            self._layout_children()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._host.removeEventFilter(self)
        super().closeEvent(event)
        self.deleteLater()
