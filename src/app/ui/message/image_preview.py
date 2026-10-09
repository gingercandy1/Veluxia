from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import QLabel, QMenu, QPushButton, QWidget


class ImagePreviewOverlay(QWidget):
    """点击缩略图后铺满宿主窗口的看图浮层，类似网页里的 lightbox：
    暗色背景 + 居中大图，点击任意位置或按 Esc 关闭。"""

    MAX_FRACTION = 0.88  # 相对宿主窗口尺寸的最大占比，给四周留出呼吸空间
    # 缩放上下限：相对"适应窗口"的倍数。下限不让图缩到看不见，上限够看像素细节
    MIN_ZOOM = 0.2
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.15  # 每格滚轮的倍率
    DRAG_THRESHOLD = 4  # 位移超过这个像素数才算拖拽，否则算单击

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
        self._path = path
        # 当前缩放倍数（相对适应窗口的尺寸）和图的中心偏移（相对浮层中心，像素）
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._drag_origin: QPoint | None = None
        self._drag_pan_origin = QPoint(0, 0)
        self._dragged = False
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
            self._label.setText("⚠️ " + self.tr("Failed to load image"))
            self._label.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 14px;")
            self._source_pixmap = None
        else:
            self._source_pixmap = pix
            self._apply_scaled_pixmap()

    def _fit_size(self):
        """适应窗口时图的尺寸（缩放倍数为 1 时）。"""
        bound = self._host.size()
        max_w = max(1, int(bound.width() * self.MAX_FRACTION))
        max_h = max(1, int(bound.height() * self.MAX_FRACTION))
        return self._source_pixmap.size().scaled(
            max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio)

    def _apply_scaled_pixmap(self):
        if self._source_pixmap is None:
            return
        size = self._fit_size() * self._zoom
        # 放大到超过窗口时仍按缩放倍数渲染，所以这里不再限制到窗口内
        scaled = self._source_pixmap.scaled(
            max(1, round(size.width())), max(1, round(size.height())),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def _layout_children(self):
        self._apply_scaled_pixmap()
        self._label.move(
            (self.width() - self._label.width()) // 2 + self._pan.x(),
            (self.height() - self._label.height()) // 2 + self._pan.y(),
        )
        self._close_btn.move(self.width() - self._close_btn.width() - 16, 16)

    def _zoom_at(self, factor: float, anchor: QPoint):
        """以 anchor（浮层坐标）为中心缩放：锚点下方的图像素保持在鼠标下面。"""
        if self._source_pixmap is None:
            return
        new_zoom = min(self.MAX_ZOOM, max(self.MIN_ZOOM, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        # 图中心在浮层里的位置 = 浮层中心 + pan；锚点相对图中心的偏移按倍率同比例放大
        center = QPoint(self.width() // 2, self.height() // 2) + self._pan
        ratio = new_zoom / self._zoom
        offset = anchor - center
        self._pan = self._pan + QPoint(round(offset.x() * (1 - ratio)),
                                       round(offset.y() * (1 - ratio)))
        self._zoom = new_zoom
        self._layout_children()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._layout_children()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            self._zoom_at(self.ZOOM_STEP ** steps, event.position().toPoint())
        event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def eventFilter(self, obj, event):
        if obj is self._host and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._host.rect())
            self._layout_children()
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self._drag_pan_origin = QPoint(self._pan)
            self._dragged = False

    def mouseMoveEvent(self, event):
        if self._drag_origin is None:
            return
        delta = event.position().toPoint() - self._drag_origin
        if not self._dragged and max(abs(delta.x()), abs(delta.y())) < self.DRAG_THRESHOLD:
            return
        self._dragged = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self._pan = self._drag_pan_origin + delta
        self._layout_children()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._drag_origin is None:
            return
        was_click = not self._dragged
        self._drag_origin = None
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 拖过就不关闭，只有没拖动的单击才关闭（和之前的点击关闭行为一致）
        if was_click:
            self.close()

    def contextMenuEvent(self, event):
        if self._source_pixmap is None:
            return
        menu = QMenu(self)
        copy_action = menu.addAction(self.tr("Copy image"))
        copy_action.triggered.connect(self._copy_to_clipboard)
        menu.exec(event.globalPos())

    def _copy_to_clipboard(self):
        if self._source_pixmap is not None:
            QGuiApplication.clipboard().setPixmap(self._source_pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._host.removeEventFilter(self)
        super().closeEvent(event)
        self.deleteLater()
