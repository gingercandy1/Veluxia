"""分段后的画布：一排段图按顺序排开，段间留间隔，不首尾相接。

交互与 RingStrip 一致（中键平移、滚轮以鼠标为中心缩放），但段不是环，所以平移不取模，
而是限制在内容范围内，免得拖丢。只负责绘制和点击，选中状态由外面通过 set_selected 传入。"""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy

from src.app.ui.base.widget import BaseWidget

GAP = 12
CLICK_SLOP = 4
ZOOM_MIN = 0.2
ZOOM_MAX = 8.0
ZOOM_STEP = 1.15
# 平移时内容至少留这么多像素在控件内
PAN_MARGIN = 40
# 与 RingStrip 的选中接缝 / 当前屏黄框同色
SELECTED_COLOR = QColor(255, 214, 102)


class SegmentCanvas(BaseWidget):
    segment_clicked = Signal(int, object)      # 段序号, 键盘修饰键

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("segment_canvas")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._images: list[QImage] = []
        self._pixmaps: list[QPixmap | None] = []
        self._selected: list[int] = []
        self._zoom = 1.0
        self._pan = [0.0, 0.0]             # 内容左上角相对控件的位置
        self._pan_start: tuple[float, float] | None = None
        self._drag_origin = (0.0, 0.0)
        self._click: tuple[float, float] | None = None
        self._cache_key: tuple[int, float] | None = None
        self._rects: list[QRectF] = []     # 上一次布局里各段的位置，供点击命中

    def set_images(self, images: list[QImage]):
        # 空图也占一个位置：段序号必须和外面的段一一对应，不能因为某张没加载就错位
        self._images = list(images)
        self._selected = [i for i in self._selected if i < len(self._images)]
        self._cache_key = None
        self.update()

    def segment_count(self) -> int:
        return len(self._images)

    def set_selected(self, indices: list[int]):
        self._selected = [i for i in indices if 0 <= i < len(self._images)]
        self.update()

    def selected(self) -> list[int]:
        return list(self._selected)

    def zoom(self) -> float:
        return self._zoom

    def reset_view(self):
        self._zoom = 1.0
        self._pan = [0.0, 0.0]
        self._cache_key = None
        self.update()

    def segment_rect(self, index: int) -> QRectF:
        """第 index 段在控件里的位置（按当前缩放和平移），测试和点击命中共用。"""
        self._layout()
        return QRectF(self._rects[index]) if 0 <= index < len(self._rects) else QRectF()

    # ---- 布局 ----
    def _image_height(self) -> float:
        return max(1.0, self.height() * self._zoom)

    def _layout(self):
        """按高度等比缩放；x 从平移量起依次排开。缩放后的位图按需生成并缓存。"""
        height = self._image_height()
        key = (self.height(), self._zoom)
        if self._cache_key != key:
            self._cache_key = key
            self._pixmaps = [None] * len(self._images)
        self._rects = []
        x = self._pan[0]
        for image in self._images:
            width = image.width() * height / image.height() if image.height() else 0.0
            self._rects.append(QRectF(x, self._pan[1], width, height))
            x += width + GAP

    def _content_width(self) -> float:
        return self._rects[-1].right() - self._rects[0].left() if self._rects else 0.0

    def _pixmap(self, index: int) -> QPixmap:
        pixmap = self._pixmaps[index]
        if pixmap is None:
            image = self._images[index]
            target = round(self._image_height())
            # 缩小先平滑缩到目标尺寸，避免直接降采样出锯齿；放大用原图
            pixmap = QPixmap.fromImage(
                image.scaledToHeight(target, Qt.TransformationMode.SmoothTransformation)
                if 0 < target < image.height() else image)
            self._pixmaps[index] = pixmap
        return pixmap

    def _clamp_pan(self):
        self._layout()
        width = self._content_width()
        self._pan[0] = min(self.width() - PAN_MARGIN, max(PAN_MARGIN - width, self._pan[0]))
        self._pan[1] = min(self.height() - PAN_MARGIN,
                           max(PAN_MARGIN - self._image_height(), self._pan[1]))

    # ---- 绘制 ----
    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self._images:
            return
        self._layout()
        pen = QPen(SELECTED_COLOR, 3)
        for index, rect in enumerate(self._rects):
            if rect.right() < 0 or rect.left() > self.width() or rect.isEmpty():
                continue
            pixmap = self._pixmap(index)
            painter.drawPixmap(rect, pixmap, QRectF(pixmap.rect()))
            if index in self._selected:
                painter.setPen(pen)
                painter.drawRect(rect.adjusted(1.5, 1.5, -1.5, -1.5))

    # ---- 交互 ----
    def set_zoom(self, zoom: float, anchor_x: float | None = None, anchor_y: float | None = None):
        """缩放以锚点为中心：锚点下方的画面位置保持不动。"""
        new_zoom = min(ZOOM_MAX, max(ZOOM_MIN, zoom))
        if new_zoom == self._zoom:
            return
        ax = self.width() / 2 if anchor_x is None else anchor_x
        ay = self.height() / 2 if anchor_y is None else anchor_y
        ratio = new_zoom / self._zoom
        # 段间隔是固定像素，不随缩放；只缩放图本身，所以锚点按"内容坐标"换算后重排
        self._pan = [ax - (ax - self._pan[0]) * ratio, ay - (ay - self._pan[1]) * ratio]
        self._zoom = new_zoom
        self._cache_key = None
        self._clamp_pan()
        self.update()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            position = event.position()
            self.set_zoom(self._zoom * ZOOM_STEP ** steps, position.x(), position.y())
        event.accept()

    def mousePressEvent(self, event):
        position = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = (position.x(), position.y())
            self._drag_origin = (self._pan[0], self._pan[1])
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._click = (position.x(), position.y())
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is None:
            return
        position = event.position()
        self._pan = [self._drag_origin[0] + position.x() - self._pan_start[0],
                     self._drag_origin[1] + position.y() - self._pan_start[1]]
        self._clamp_pan()
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton or self._click is None:
            return
        (press_x, press_y), self._click = self._click, None
        position = event.position()
        if abs(position.x() - press_x) > CLICK_SLOP or abs(position.y() - press_y) > CLICK_SLOP:
            return
        self._layout()
        for index, rect in enumerate(self._rects):
            if rect.contains(position):
                self.segment_clicked.emit(index, event.modifiers())
                return

    def mouseDoubleClickEvent(self, event):
        # 双击回到默认视角
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()

    def resizeEvent(self, event):
        self._cache_key = None
        super().resizeEvent(event)
