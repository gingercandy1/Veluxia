from typing import Optional

from PySide6.QtCore import (
    Signal, QPoint, QPropertyAnimation,
    QParallelAnimationGroup, QEasingCurve,
    QAbstractAnimation, Property, Qt
)
from PySide6.QtGui import QPainter, QPainterPath, QColor, QLinearGradient, QPen
from PySide6.QtWidgets import QWidget, QHBoxLayout



class DotItem(QWidget):
    """ DotItem — 单个指示点（圆 ↔ 胶囊动画）"""
    clicked = Signal(int)

    def __init__(self, index: int, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._index    = index
        self._is_active = False
        self._scale_x  = 1.0

        self.setFixedSize(21, 21)
        self.setCursor(Qt.PointingHandCursor)

        self._anim = QPropertyAnimation(self, b"scaleX")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutQuad)

    def _get_scale_x(self) -> float:
        return self._scale_x

    def _set_scale_x(self, v: float):
        if self._scale_x != v:
            self._scale_x = v
            self.update()

    scaleX = Property(float, _get_scale_x, _set_scale_x)

    def set_active(self, active: bool):
        """点击或页面切换完成时调用，触发动画"""
        self._is_active = active
        self._anim.stop()
        if active:
            self._anim.setStartValue(self._scale_x)
            self._anim.setEndValue(1.6)
        else:
            self._anim.setStartValue(self._scale_x)
            self._anim.setEndValue(1.0)
        self._anim.start()

    def set_progress(self, progress: float, is_target: bool, is_finished: bool):
        if is_finished:
            return
        if self._is_active and not is_target:
            self._set_scale_x(2.0 - progress)   # 胶囊 → 圆
        elif is_target:
            self._set_scale_x(1.0 + progress)   # 圆 → 胶囊

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        base_w  = self.height() - 4
        cx      = self.width()  / 2.0
        cy      = self.height() / 2.0

        # 外圈
        outer_x = cx - base_w / 2.0
        p.setPen(QPen(QColor(180, 180, 180, 150), 1.5))
        p.setBrush(Qt.NoBrush)
        outer_path = QPainterPath()
        outer_path.addEllipse(outer_x, cy - base_w / 2.0, base_w, base_w)
        p.drawPath(outer_path)

        if self._scale_x > 1.0:
            fill_w = base_w * (self._scale_x - 1.0)
            fill_x = cx - fill_w / 2.0
            fill_h = fill_w

            fill_color = QColor(122, 122, 122, 150)
            grad = QLinearGradient(fill_x, 0, fill_x + fill_w, base_w)
            grad.setColorAt(0, fill_color.lighter(120))
            grad.setColorAt(1, fill_color)

            p.setPen(Qt.NoPen)
            p.setBrush(grad)

            # 用圆角矩形实现胶囊形状
            fill_path = QPainterPath()
            fill_path.addRoundedRect(fill_x, cy - fill_h / 2.0,
                                     fill_w, fill_h, fill_h / 2.0, fill_h / 2.0)
            p.drawPath(fill_path)

        p.end()

    def mousePressEvent(self, _):
        self.clicked.emit(self._index)


class DotIndicatorBar(QWidget):
    """
    DotIndicatorBar — 顶部指示器行
    点击 dot → switch_to(index)
    拖拽中 → update_drag(current, target, progress, finished)
    """
    switch_to = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("dot_indicator_bar")
        self.setFixedHeight(25)

        self._current = -1
        self._dots: list[DotItem] = []

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(8, 0, 8, 0)
        self._layout.setSpacing(6)

    def set_page_count(self, count: int):
        # 清除旧 dots
        for dot in self._dots:
            self._layout.removeWidget(dot)
            dot.deleteLater()
        self._dots.clear()

        # 重新插入
        for i in range(count):
            dot = DotItem(i, self)
            dot.clicked.connect(self._on_dot_clicked)
            self._dots.append(dot)
            self._layout.insertWidget(i + 1, dot)

        if count > 0:
            self.set_current_index(0)
        self._layout.addStretch()

    def set_current_index(self, index: int):
        if index < 0 or index >= len(self._dots):
            return
        self._current = index
        for i, dot in enumerate(self._dots):
            dot.set_active(i == index)

    def update_drag(self, current: int, target: int,
                    progress: float, finished: bool):
        if (current < 0 or target < 0
                or current >= len(self._dots)
                or target >= len(self._dots)):
            return
        self._dots[target].set_progress(progress, True,  finished)
        self._dots[current].set_progress(progress, False, finished)

    def _on_dot_clicked(self, index: int):
        self.set_current_index(index)
        self.switch_to.emit(index)


class SlideStackWidget(QWidget):
    """
    SlideStackWidget — 滑动切换核心
    - 页面以 setParent(self) + move() 管理
    - 支持鼠标拖拽切换（阈值 width/2）
    - 支持点击切换（带动画）
    """
    current_changed = Signal(int)
    drag_progress   = Signal(int, int, float, bool)

    _DRAG_THRESHOLD  = 10
    _SNAP_THRESHOLD  = 0.5

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("slide_stack_widget")

        self._pages:         list[QWidget] = []
        self._current_index: int           = -1
        self._current_page:  Optional[QWidget] = None
        self._target_page:   Optional[QWidget] = None
        self._target_index:  int           = -1

        # 拖拽状态
        self._start_pos:  QPoint = QPoint()
        self._dragging:   bool   = False
        self._cached_w:   int    = 0

        # 进行中的动画组
        self._anim_group: Optional[QParallelAnimationGroup] = None

    def add_page(self, page: QWidget, name: str = "") -> int:
        page.setParent(self)
        if name:
            page.setObjectName(name)
        page.hide()
        self._pages.append(page)

        if len(self._pages) == 1:
            self.set_current_index(0)

        return len(self._pages) - 1

    def remove_page(self, index: int):
        if index < 0 or index >= len(self._pages):
            return
        page = self._pages.pop(index)
        page.hide()
        page.setParent(None)
        page.deleteLater()

        if not self._pages:
            self._current_index = -1
            self._current_page  = None
        else:
            new_idx = min(self._current_index, len(self._pages) - 1)
            self.set_current_index(new_idx)

    def count(self) -> int:
        return len(self._pages)

    def current_index(self) -> int:
        return self._current_index

    def widget_at(self, index: int) -> Optional[QWidget]:
        if 0 <= index < len(self._pages):
            return self._pages[index]
        return None

    def index_of(self, name: str) -> int:
        for i, p in enumerate(self._pages):
            if p.objectName() == name:
                return i
        return -1

    def set_current_index(self, index: int):
        if index < 0 or index >= len(self._pages) or index == self._current_index:
            return
        if self._current_page:
            self._current_page.hide()

        self._current_index = index
        self._current_page  = self._pages[index]
        self._current_page.move(0, 0)
        self._current_page.show()
        self.current_changed.emit(index)

    def start_transition(self, target_index: int):
        if (target_index == self._current_index
                or target_index < 0
                or target_index >= len(self._pages)):
            return

        # 停止上一个动画
        if self._anim_group and self._anim_group.state() == QAbstractAnimation.Running:
            self._anim_group.stop()

        cur_page = self._pages[self._current_index]
        tgt_page = self._pages[target_index]

        cur_page.show()
        tgt_page.show()

        direction = 1 if target_index > self._current_index else -1
        w = self.width() or 400
        print("width：", w)

        self._create_transition(cur_page, tgt_page, target_index, direction, w)

    def _create_transition(self, cur: QWidget, tgt: QWidget,
                            target_index: int, direction: int, w: int):
        dur   = 300
        curve = QEasingCurve.Type.OutQuad

        cur_anim = QPropertyAnimation(cur, b"pos")
        cur_anim.setDuration(dur)
        cur_anim.setEasingCurve(curve)
        cur_anim.setEndValue(QPoint(-w if direction == 1 else w, 0))

        tgt_anim = QPropertyAnimation(tgt, b"pos")
        tgt_anim.setDuration(dur)
        tgt_anim.setEasingCurve(curve)
        # 目标页从屏幕外开始
        tgt.move(w if direction == 1 else -w, 0)
        tgt_anim.setEndValue(QPoint(0, 0))

        group = QParallelAnimationGroup(self)
        group.addAnimation(cur_anim)
        group.addAnimation(tgt_anim)

        def on_finished():
            cur.hide()
            tgt.move(0, 0)
            self._current_index = target_index
            self._current_page  = tgt
            self._target_page   = None
            self._anim_group    = None
            self.current_changed.emit(target_index)

        group.finished.connect(on_finished)
        self._anim_group = group
        group.start()

    def mousePressEvent(self, event):
        self._start_pos = event.pos()
        self._dragging  = False
        self._target_page = None
        if 0 <= self._current_index < len(self._pages):
            self._current_page = self._pages[self._current_index]
            if self._current_page:
                self._current_page.show()

    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self._start_pos.x()

        if not self._dragging and abs(dx) > self._DRAG_THRESHOLD:
            self._dragging  = True
            self._cached_w  = self.width()

            direction  = 1 if dx < 0 else -1
            new_index  = self._current_index + direction

            if 0 <= new_index < len(self._pages):
                self._target_index = new_index
                self._current_page = self._pages[self._current_index]
                self._target_page  = self._pages[new_index]

                if self._current_page and self._target_page:
                    self._current_page.show()
                    self._target_page.show()
                    self._current_page.move(0, 0)
                    # 目标页初始位置在屏幕外
                    self._target_page.move(
                        self._cached_w if direction == 1 else -self._cached_w, 0
                    )
            else:
                self._dragging = False

        elif self._dragging and self._current_page and self._target_page:
            w         = self.width()
            direction = 1 if dx < 0 else -1

            # 同向才移动（防止方向切换导致错位）
            if direction == (1 if self._target_index > self._current_index else -1):
                cur_x = dx if direction == 1 else -dx
                if direction == 1:
                    self._current_page.move(cur_x, 0)
                    self._target_page.move(w + cur_x, 0)
                else:
                    self._current_page.move(-cur_x, 0)
                    self._target_page.move(-(w + cur_x), 0)  # ← 修正C++中的 bug

                # 实时更新 dot 进度
                progress = min(abs(dx) / w, 1.0)
                self.drag_progress.emit(
                    self._current_index, self._target_index, progress, False
                )

    def mouseReleaseEvent(self, event):
        if not self._dragging:
            return

        dx = event.pos().x() - self._start_pos.x()
        w  = self.width()

        if abs(dx) > w * self._SNAP_THRESHOLD and self._target_index >= 0:
            # 超过阈值 → 完成切换
            self.start_transition(self._target_index)
        else:
            # 未超过 → 回弹
            if self._current_page:
                self._current_page.move(0, 0)
            if self._target_page:
                self._target_page.hide()
            self.current_changed.emit(self._current_index)

        self._dragging    = False
        self._target_page = None
        self.drag_progress.emit(
            self._current_index, self._target_index, 0.0, True
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def adjust_page_sizes(self, height):
        self.setFixedHeight(height)