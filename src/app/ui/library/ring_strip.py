"""循环场景预览条带：只负责绘制，不读取资源包或发起请求。"""
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import FrameTicker

SCROLL_SPEED = 60.0
SEAM_HIT_RADIUS = 24
# 缩放倍数的上下限：相对"图高适应控件高度"的倍数
ZOOM_MIN = 0.5
ZOOM_MAX = 4.0
ZOOM_STEP = 1.15
DRAG_THRESHOLD = 4
# 切线比接缝线更优先被抓住：两条线挨得近时，按下去拖的是切线
CUT_HIT_RADIUS = 8


class RingStrip(BaseWidget):
    seam_selected = Signal(int)
    playing_changed = Signal(bool)
    cuts_edited = Signal(list)       # 拖完切线后的全部切点（原图像素坐标，升序）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_strip")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._images: list[QImage] = []
        self._pixmaps: list[QPixmap] = []
        self._cache_height = 0
        self._cache_key: tuple[int, float] | None = None
        self._total_width = 0
        self._offset = 0.0
        self._last_tick = time.monotonic()
        self._playing = True
        self._selected = -1
        self._zoom = 1.0
        # 拖拽状态：按下时的鼠标 x、按下时的偏移，以及是否真的拖动过（否则算点击选缝）
        self._drag_x: float | None = None
        self._drag_offset_origin = 0.0
        self._dragged = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        # 上一次绘制时各交界的 (x, 接缝序号)，供点击命中
        self._seam_hits: list[tuple[int, int]] = []
        # 切段位置：长图（各屏原图像素拼起来）上的 x；和接缝线一样随条带滚动、缩放
        self._cuts: list[int] = []
        self._cut_hits: list[tuple[int, int]] = []
        self._drag_cut: int | None = None
        self._ticker = FrameTicker(self)
        self._ticker.set_running(False)

    def set_images(self, images: list[QImage]):
        self._images = [image for image in images if not image.isNull()]
        if self._selected >= len(self._images):
            self._selected = -1
        self._cache_height = 0
        self._offset = 0.0
        self._last_tick = time.monotonic()
        self._ticker.set_running(self._playing and bool(self._images))
        self.update()

    def set_playing(self, playing: bool):
        if self._playing == playing:
            return
        self._advance()
        self._playing = playing
        self._last_tick = time.monotonic()
        self._ticker.set_running(playing and bool(self._images))
        self.update()
        self.playing_changed.emit(playing)

    def is_playing(self) -> bool:
        return self._playing

    def set_cuts(self, cuts: list[int]):
        self._cuts = sorted(int(cut) for cut in cuts)
        self._drag_cut = None
        self.update()

    def cuts(self) -> list[int]:
        return list(self._cuts)

    def source_width(self) -> int:
        """长图总宽（原图像素），切点坐标的取值范围。"""
        return sum(image.width() for image in self._images)

    def _source_scale(self) -> float:
        source = self.source_width()
        return self._total_width / source if source else 1.0

    def seam_count(self) -> int:
        return len(self._images)

    def set_offset(self, offset: float):
        """设置可复现的测试/预览偏移；实际播放偏移仍由单调时钟驱动。"""
        self._offset = float(offset)
        self._last_tick = time.monotonic()
        self.update()

    def _advance(self):
        now = time.monotonic()
        if self._playing and self._total_width:
            self._offset = (self._offset + (now - self._last_tick) * SCROLL_SPEED) \
                % self._total_width
        self._last_tick = now

    def _ensure_pixmaps(self):
        # 缩放倍数也要进缓存判断：滚轮改了倍数，图要按新尺寸重新生成
        key = (self.height(), self._zoom)
        if self._cache_key == key:
            return
        self._cache_key = key
        self._cache_height = self.height()
        target_height = max(1, round(self._cache_height * self._zoom))
        self._pixmaps = [
            QPixmap.fromImage(image.scaledToHeight(
                target_height, Qt.TransformationMode.SmoothTransformation))
            for image in self._images
        ]
        self._total_width = sum(pixmap.width() for pixmap in self._pixmaps)
        if self._total_width:
            self._offset %= self._total_width

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self._images:
            painter.setPen(QColor(255, 255, 255, 105))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             self.tr("Add at least two finished scene screens, then plan the ring."))
            return

        self._ensure_pixmaps()
        self._advance()
        if not self._total_width:
            return
        offset = self._offset % self._total_width
        x = -offset
        index = 0
        seam_pen = QColor(123, 157, 188, 180)
        selected_pen = QPen(QColor(255, 214, 102), 3)
        self._seam_hits = []
        while x < self.width():
            pixmap = self._pixmaps[index]
            position = round(x)
            painter.drawPixmap(position, 0, pixmap)
            # 当前图左边的交界属于上一张图的接缝（接缝归左屏）
            seam = (index - 1) % len(self._pixmaps)
            self._seam_hits.append((position, seam))
            painter.setPen(selected_pen if seam == self._selected else seam_pen)
            painter.drawLine(position, 0, position, self.height())
            x += pixmap.width()
            index = (index + 1) % len(self._pixmaps)
        self._paint_cuts(painter, offset)

    def _paint_cuts(self, painter: QPainter, offset: float):
        self._cut_hits = []
        if not self._cuts:
            return
        scale = self._source_scale()
        pen = QPen(QColor(255, 82, 82), 2, Qt.PenStyle.DashLine)
        handle = QColor(255, 82, 82)
        for index, cut in enumerate(self._cuts):
            # 条带是环：同一个切点在可见范围里可能出现不止一次
            x = cut * scale - offset
            while x < self.width():
                if x >= -CUT_HIT_RADIUS:
                    position = round(x)
                    self._cut_hits.append((position, index))
                    painter.setPen(pen)
                    painter.drawLine(position, 0, position, self.height())
                    painter.fillRect(position - 5, 0, 10, 14, handle)
                x += self._total_width

    def set_selected(self, seam: int):
        """选中的接缝序号：第 i 个接缝是第 i 张图和下一张图（环，最后一张接回第一张）的交界。"""
        self._selected = seam if 0 <= seam < len(self._images) else -1
        self.update()

    def selected_seam(self) -> int:
        return self._selected

    def set_zoom(self, zoom: float, anchor_x: float | None = None):
        """缩放以 anchor_x 为中心：锚点下方的画面位置保持不动，方便对着接缝细看。"""
        new_zoom = min(ZOOM_MAX, max(ZOOM_MIN, zoom))
        if new_zoom == self._zoom or not self._total_width:
            return
        self._advance()
        anchor = self.width() / 2 if anchor_x is None else anchor_x
        # 锚点在图像里的位置（相对总宽的比例）在缩放前后保持不变
        world = (self._offset + anchor) / self._total_width
        self._zoom = new_zoom
        self._cache_key = None
        self._ensure_pixmaps()
        self._offset = world * self._total_width - anchor
        self._offset %= self._total_width
        self.update()

    def zoom(self) -> float:
        return self._zoom

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            self.set_zoom(self._zoom * ZOOM_STEP ** steps, event.position().x())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x = event.position().x()
        nearest = min(self._cut_hits, key=lambda hit: abs(hit[0] - x), default=None)
        if nearest is not None and abs(nearest[0] - x) <= CUT_HIT_RADIUS:
            self._drag_cut = nearest[1]
            self.set_playing(False)
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            return
        self._drag_x = x
        self._drag_offset_origin = self._offset
        self._dragged = False

    def mouseMoveEvent(self, event):
        if self._drag_cut is not None:
            self._move_cut(event.position().x())
            return
        if self._drag_x is None:
            return
        delta = event.position().x() - self._drag_x
        if not self._dragged and abs(delta) < DRAG_THRESHOLD:
            return
        self._dragged = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        # 拖动时暂停自动滚动，松开后停在拖到的位置，不再继续走
        self._advance()
        if self._playing:
            self.set_playing(False)
        self._offset = (self._drag_offset_origin - delta) % self._total_width
        self._last_tick = time.monotonic()
        self.update()

    def _move_cut(self, screen_x: float):
        if not self._total_width:
            return
        source = self.source_width()
        world = (screen_x + self._offset) % self._total_width
        self._cuts[self._drag_cut] = min(source - 1, max(0, round(world / self._source_scale())))
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_cut is not None:
            self._drag_cut = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._cuts.sort()
            self.cuts_edited.emit(list(self._cuts))
            return
        if event.button() != Qt.MouseButton.LeftButton or self._drag_x is None:
            return
        was_click = not self._dragged
        self._drag_x = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if was_click:
            # 滚动时交界一直在动，点中线附近的一小段就算选中
            x = event.position().x()
            nearest = min(self._seam_hits, key=lambda hit: abs(hit[0] - x), default=None)
            if nearest is not None and abs(nearest[0] - x) <= SEAM_HIT_RADIUS:
                self.set_selected(nearest[1])
                self.seam_selected.emit(nearest[1])

    def mouseDoubleClickEvent(self, event):
        # 双击回到默认缩放，偏移归零，便于重新对齐
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom = 1.0
            self._cache_key = None
            self._offset = 0.0
            self.update()

    def resizeEvent(self, event):
        self._cache_key = None
        super().resizeEvent(event)
