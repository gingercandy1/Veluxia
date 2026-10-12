"""循环场景预览条带：只负责绘制，不读取资源包或发起请求。

像画布一样用：中键拖动平移（横向首尾相接成环，纵向自由），滚轮以鼠标为中心缩放；
左键点接缝线选中、拖红色切线调整切点。
"""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.app.ui.base.widget import BaseWidget

SEAM_HIT_RADIUS = 24
# 浮层离条带右上角的距离
OVERLAY_MARGIN = 10
CLICK_SLOP = 4
# 缩放倍数的上下限：相对"图高适应控件高度"的倍数
ZOOM_MIN = 0.2
ZOOM_MAX = 8.0
ZOOM_STEP = 1.15
# 切线比接缝线更优先被抓住：两条线挨得近时，按下去拖的是切线
CUT_HIT_RADIUS = 8
# 删除时点得松一点：细虚线不好瞄准，误删的代价又比误拖大
CUT_REMOVE_RADIUS = 16
# 纵向平移时图至少留这么多像素在控件内，免得拖丢
PAN_MARGIN = 40


class RingStrip(BaseWidget):
    seam_selected = Signal(int)
    cuts_edited = Signal(list)       # 拖完切线后的全部切点（原图像素坐标，升序）
    seams_selected = Signal(list)    # 选中接缝集合变化后的全部序号（单击重置为一条，Ctrl+单击加减）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_strip")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._images: list[QImage] = []
        self._pixmaps: list[QPixmap | None] = []
        self._widths: list[float] = []
        self._image_height = 0
        self._cache_key: tuple[int, float] | None = None
        self._total_width = 0.0
        # 画布视角：横向偏移（环上位置）、图顶部的纵坐标、缩放倍数
        self._offset = 0.0
        self._pan_y = 0.0
        self._zoom = 1.0
        self._selected = -1
        # 多选的接缝；_selected 仍是"最近一次选中的那条"，保持旧接口语义
        self._selected_seams: list[int] = []
        # 与 images 一一对应：该屏上下共补齐的原图像素，用半透明遮罩标出"不是真画面"
        self._padding: list[int] = []
        # 卡片条上点中的那一屏：在条带里用黄框圈出来（和选中的接缝线同色）
        self._active_image = -1
        # 中键平移：按下时的鼠标位置与视角；左键点击：按下时的 x
        self._pan_start: tuple[float, float] | None = None
        self._drag_origin = (0.0, 0.0)
        self._click_pos: float | None = None
        # 上一次绘制时各交界的 (x, 接缝序号)，供点击命中
        self._seam_hits: list[tuple[int, int]] = []
        # 切段位置：长图（各屏原图像素拼起来）上的 x；和接缝线一样随条带平移、缩放
        self._cuts: list[int] = []
        self._cut_hits: list[tuple[int, int]] = []
        self._drag_cut: int | None = None
        self._overlay: QWidget | None = None

    def set_overlay(self, widget: QWidget):
        """挂一个浮在条带右上角的控件（如前/后开关）：它盖在画面上，不占条带的行高。"""
        self._overlay = widget
        widget.setParent(self)
        widget.show()
        self._place_overlay()

    def _place_overlay(self):
        if self._overlay is None:
            return
        self._overlay.adjustSize()
        self._overlay.move(self.width() - self._overlay.width() - OVERLAY_MARGIN, OVERLAY_MARGIN)
        self._overlay.raise_()

    def set_images(self, images: list[QImage]):
        self._images = [image for image in images if not image.isNull()]
        if self._selected >= len(self._images):
            self._selected = -1
        self._selected_seams = [s for s in self._selected_seams if s < len(self._images)]
        if self._active_image >= len(self._images):
            self._active_image = -1
        # 视角（偏移、纵向平移）保留：前后切换、轮询刷新都不该把画面拉回起点，换资源包时才 reset_view
        self._cache_key = None
        self.update()

    def set_padding(self, padding: list[int]):
        self._padding = [max(0, int(value)) for value in padding]
        self.update()

    def reset_view(self):
        self._offset = 0.0
        self._pan_y = 0.0
        self.update()

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
        """设置横向偏移，便于测试和预览复现同一画面。"""
        self._offset = float(offset)
        self.update()

    def _ensure_pixmaps(self):
        # 缩放倍数也要进缓存判断：滚轮改了倍数，缩小的图要按新尺寸重新生成
        key = (self.height(), self._zoom)
        if self._cache_key == key:
            return
        self._cache_key = key
        self._image_height = max(1, round(self.height() * self._zoom))
        # 位图按需生成（见 _pixmap）：滚轮缩放时只有画面里看得见的几屏要重新缩放，
        # 不必把整个环的每一屏都缩一遍
        self._pixmaps = [None] * len(self._images)
        self._widths = [image.width() * self._image_height / image.height()
                        for image in self._images]
        self._total_width = sum(self._widths)
        if self._total_width:
            self._offset %= self._total_width

    def _pixmap(self, index: int) -> QPixmap:
        pixmap = self._pixmaps[index]
        if pixmap is None:
            image = self._images[index]
            # 缩小时先平滑缩到目标尺寸，避免直接降采样出锯齿；放大时用原图，细节不丢
            pixmap = QPixmap.fromImage(
                image.scaledToHeight(self._image_height, Qt.TransformationMode.SmoothTransformation)
                if self._image_height < image.height() else image)
            self._pixmaps[index] = pixmap
        return pixmap

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if not self._images:
            painter.setPen(QColor(255, 255, 255, 105))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             self.tr("Add at least two finished scene screens, then plan the ring."))
            return

        self._ensure_pixmaps()
        if not self._total_width:
            return
        offset = self._offset % self._total_width
        x = -offset
        index = 0
        top = round(self._pan_y)
        bottom = top + self._image_height
        seam_pen = QPen(QColor(123, 157, 188, 150), 1)
        selected_pen = QPen(QColor(255, 214, 102), 2)
        active_pen = QPen(QColor(255, 214, 102), 3)
        self._seam_hits = []
        while x < self.width():
            pixmap = self._pixmap(index)
            width = self._widths[index]
            position = round(x)
            painter.drawPixmap(QRectF(x, top, width, self._image_height), pixmap,
                               QRectF(pixmap.rect()))
            self._paint_padding(painter, index, x, top, width)
            # 当前图左边的交界属于上一张图的接缝（接缝归左屏）
            seam = (index - 1) % len(self._images)
            self._seam_hits.append((position, seam))
            # 接缝线只画在两张图相接的那段高度上，不画满整个控件
            painter.setPen(selected_pen if seam in self._selected_seams else seam_pen)
            painter.drawLine(position, top, position, bottom)
            if index == self._active_image:
                painter.setPen(active_pen)
                painter.drawRect(QRectF(x + 1.5, top + 1.5, width - 3, self._image_height - 3))
            x += width
            index = (index + 1) % len(self._images)
        self._paint_cuts(painter, offset, top, bottom)

    def _paint_padding(self, painter: QPainter, index: int, x: float, top: int, width: float):
        """较矮的屏是用边缘行拉伸补齐到统一高度的：上下各一半，盖一层深色标出来。"""
        padding = self._padding[index] if index < len(self._padding) else 0
        if not padding:
            return
        band = padding / 2 * self._image_height / self._images[index].height()
        shade = QColor(0, 0, 0, 120)
        painter.fillRect(QRectF(x, top, width, band), shade)
        painter.fillRect(QRectF(x, top + self._image_height - band, width, band), shade)

    def _paint_cuts(self, painter: QPainter, offset: float, top: int, bottom: int):
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
                    painter.drawLine(position, top, position, bottom)
                    painter.fillRect(position - 5, top, 10, 14, handle)
                x += self._total_width

    def set_selected(self, seam: int):
        """选中的接缝序号：第 i 个接缝是第 i 张图和下一张图（环，最后一张接回第一张）的交界。"""
        self._selected = seam if 0 <= seam < len(self._images) else -1
        self._selected_seams = [self._selected] if self._selected >= 0 else []
        self.update()

    def selected_seam(self) -> int:
        return self._selected

    def selected_seams(self) -> list[int]:
        return list(self._selected_seams)

    def set_selected_seams(self, seams: list[int]):
        """按集合选中接缝（越界的忽略）；不发信号，由调用方负责同步。"""
        valid = sorted({s for s in seams if 0 <= s < len(self._images)})
        self._selected_seams = valid
        self._selected = valid[-1] if valid else -1
        self.update()

    def _toggle_seam(self, seam: int):
        if seam in self._selected_seams:
            self._selected_seams.remove(seam)
            self._selected = self._selected_seams[-1] if self._selected_seams else -1
        else:
            self._selected_seams = sorted([*self._selected_seams, seam])
            self._selected = seam
        self.update()
        self.seams_selected.emit(self.selected_seams())

    def set_active_image(self, index: int, focus: bool = True):
        """给第 index 张图加黄框（-1 取消）；focus 时把它平移到控件中央，免得它在可视区外。"""
        self._active_image = index if 0 <= index < len(self._images) else -1
        if focus and self._active_image >= 0:
            self._ensure_pixmaps()
            if self._total_width:
                start = sum(self._widths[:self._active_image])
                center = start + self._widths[self._active_image] / 2
                self._offset = (center - self.width() / 2) % self._total_width
        self.update()

    def active_image(self) -> int:
        return self._active_image

    def _clamp_pan_y(self):
        self._pan_y = min(self.height() - PAN_MARGIN,
                          max(PAN_MARGIN - self._image_height, self._pan_y))

    def set_zoom(self, zoom: float, anchor_x: float | None = None, anchor_y: float | None = None):
        """缩放以锚点为中心：锚点下方的画面位置保持不动，方便对着某处细看。"""
        self._ensure_pixmaps()
        new_zoom = min(ZOOM_MAX, max(ZOOM_MIN, zoom))
        if new_zoom == self._zoom or not self._total_width:
            return
        ax = self.width() / 2 if anchor_x is None else anchor_x
        ay = self.height() / 2 if anchor_y is None else anchor_y
        # 锚点在图像里的位置（相对总宽的比例）在缩放前后保持不变
        world = (self._offset + ax) / self._total_width
        ratio = new_zoom / self._zoom
        self._pan_y = ay - (ay - self._pan_y) * ratio
        self._zoom = new_zoom
        self._cache_key = None
        self._ensure_pixmaps()
        self._offset = (world * self._total_width - ax) % self._total_width
        self._clamp_pan_y()
        self.update()

    def zoom(self) -> float:
        return self._zoom

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120
        if steps:
            position = event.position()
            self.set_zoom(self._zoom * ZOOM_STEP ** steps, position.x(), position.y())
        event.accept()

    def mousePressEvent(self, event):
        x = event.position().x()
        if event.button() == Qt.MouseButton.MiddleButton:
            # 中键专管平移，和左键拖切线分开，免得按到切线附近时拖错对象
            self._pan_start = (x, event.position().y())
            self._drag_origin = (self._offset, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if shift and event.button() == Qt.MouseButton.LeftButton:
            self._add_cut(x)
            return
        if shift and event.button() == Qt.MouseButton.RightButton:
            self._remove_cut(x)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        nearest = min(self._cut_hits, key=lambda hit: abs(hit[0] - x), default=None)
        if nearest is not None and abs(nearest[0] - x) <= CUT_HIT_RADIUS:
            self._drag_cut = nearest[1]
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            return
        # 左键在空白处只算点击（选接缝），不拖动画面
        self._click_pos = x

    def mouseMoveEvent(self, event):
        if self._drag_cut is not None:
            self._move_cut(event.position().x())
            return
        if self._pan_start is None or not self._total_width:
            return
        dx = event.position().x() - self._pan_start[0]
        dy = event.position().y() - self._pan_start[1]
        self._offset = (self._drag_origin[0] - dx) % self._total_width
        self._pan_y = self._drag_origin[1] + dy
        self._clamp_pan_y()
        self.update()

    def _add_cut(self, screen_x: float):
        """Shift+左键：在点击处加一条切线；已经有切线的地方不重复加。"""
        if not self._total_width:
            return
        nearest = self._nearest_cut(screen_x)
        if nearest is not None and nearest[1] <= CUT_HIT_RADIUS:
            return
        world = (screen_x + self._offset) % self._total_width
        cut = min(self.source_width() - 1, max(0, round(world / self._source_scale())))
        self._cuts = sorted({*self._cuts, cut})
        self.update()
        self.cuts_edited.emit(list(self._cuts))

    def _remove_cut(self, screen_x: float):
        """Shift+右键：删掉离点击处最近的切线。至少留一条，后端不接受没有切点。"""
        nearest = self._nearest_cut(screen_x)
        if nearest is None or nearest[1] > CUT_REMOVE_RADIUS or len(self._cuts) <= 1:
            return
        del self._cuts[nearest[0]]
        self.update()
        self.cuts_edited.emit(list(self._cuts))

    def _nearest_cut(self, screen_x: float) -> tuple[int, float] | None:
        """离屏幕 x 最近的切线：(序号, 屏幕上的距离)。直接按切点坐标算，不用上次绘制记下的
        命中表（增删之后、重绘之前它的序号已经对不上）；条带是环，取两侧里近的那个。"""
        if not self._cuts or not self._total_width:
            return None
        scale = self._source_scale()
        world = (screen_x + self._offset) % self._total_width
        distances = []
        for index, cut in enumerate(self._cuts):
            gap = abs(cut * scale - world)
            distances.append((index, min(gap, self._total_width - gap)))
        return min(distances, key=lambda item: item[1])

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
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._cuts.sort()
            self.cuts_edited.emit(list(self._cuts))
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if event.button() != Qt.MouseButton.LeftButton or self._click_pos is None:
            return
        press_x, self._click_pos = self._click_pos, None
        x = event.position().x()
        # 按下和松开隔得远是拖了一下，不算点击
        if abs(x - press_x) > CLICK_SLOP:
            return
        # 点中接缝线附近的一小段就算选中
        nearest = min(self._seam_hits, key=lambda hit: abs(hit[0] - x), default=None)
        if nearest is not None and abs(nearest[0] - x) <= SEAM_HIT_RADIUS:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._toggle_seam(nearest[1])
                return
            self.set_selected(nearest[1])
            self.seam_selected.emit(nearest[1])
            self.seams_selected.emit(self.selected_seams())

    def mouseDoubleClickEvent(self, event):
        # 双击回到默认视角，便于重新对齐
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom = 1.0
            self._cache_key = None
            self._offset = 0.0
            self._pan_y = 0.0
            self.update()

    def resizeEvent(self, event):
        self._cache_key = None
        self._place_overlay()
        super().resizeEvent(event)
