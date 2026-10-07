"""界面通用的小动画：切页淡入滑动、展开收起、抖动、按钮回弹、交叉淡入。

统一 200ms 左右、OutCubic，和侧栏、思考气泡等现有动画一致。
切页的约定：平级、无先后的页面（主窗口的生成 / 设置 / 资源库）只淡入；
有进入 / 返回或上下顺序的（资源库内部、设置子页面）再按方向滑一小段。
控件还没显示时（首次构建、离屏测试）一律直接到终态：动画要靠事件循环推进，
不显示时既看不到，也不该让状态停在半路。
"""
import math
from collections.abc import Callable

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    QPoint,
    QPropertyAnimation,
    Qt,
    QVariantAnimation,
)
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

DURATION = 200
# 主窗口、设置页、资源库共用的底色（resource.qss / setting.qss / library.qss）；
# 切页盖层用它，颜色不一致的话淡入时会先闪一下别的颜色
WINDOW_BACKGROUND = QColor("#101215")
# Qt 的 QWIDGETSIZE_MAX：展开动画结束后恢复成"不限高"，内容再变高也不会被截
_UNLIMITED = 16777215


def discard(widget: QWidget):
    """从布局里拿掉并删除控件。先隐藏：deleteLater 要等回到事件循环才真正删，
    这期间它还会按没排过版的默认尺寸（顶层窗口默认 640×480）画在原处，盖住新内容。"""
    parent_layout = widget.parentWidget().layout() if widget.parentWidget() else None
    if parent_layout is not None:
        parent_layout.removeWidget(widget)
    widget.hide()
    widget.deleteLater()


def animate(owner: QWidget, start: float, end: float, setter: Callable[[float], None],
            duration: int = DURATION, on_done: Callable[[], None] | None = None,
            curve: QEasingCurve.Type = QEasingCurve.Type.OutCubic) -> QVariantAnimation | None:
    """在 start → end 之间逐帧调用 setter；owner 不可见时直接设终值。"""
    if not owner.isVisible():
        setter(end)
        if on_done is not None:
            on_done()
        return None
    anim = QVariantAnimation(owner)
    anim.setDuration(duration)
    anim.setStartValue(float(start))
    anim.setEndValue(float(end))
    anim.setEasingCurve(curve)
    anim.valueChanged.connect(setter)
    if on_done is not None:
        anim.finished.connect(on_done)
    anim.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return anim


def _settle_layout(widget: QWidget) -> QPoint:
    """让父布局立刻把控件摆回它该在的位置，返回这个位置。

    刚切换完内容时布局还没重排（要等事件循环），直接读 widget.pos() 拿到的是旧位置。
    也不能每帧去问布局项：QWidgetItem.geometry() 返回的就是控件当前的几何，
    滑动时读到的是上一帧挪过的位置，偏移一帧帧累加，最后整页停在偏右几十像素处、右边被切掉。
    invalidate 后再 activate 会强制布局重新 setGeometry，和用户拖一下窗口效果相同。
    """
    parent = widget.parentWidget()
    layout = parent.layout() if parent is not None else None
    if layout is not None:
        layout.invalidate()
        layout.activate()
    return widget.pos()


def fade_slide_in(widget: QWidget, offset_x: int = 0, duration: int = DURATION):
    """淡入，可同时从 offset_x 处滑到原位；结束后移除透明度特效（常驻会走离屏渲染，很吃性能）。

    滑动每一帧都按布局当前给的位置算，动画期间布局变了（内容增减、窗口缩放）也跟得上。
    """
    if not widget.isVisible():
        return
    # 同一控件上一轮动画还没结束时先停掉，免得两个动画抢着改位置和透明度
    for running in widget.findChildren(QParallelAnimationGroup, "fade_slide_in"):
        running.stop()
    home = _settle_layout(widget)
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.0)
    widget.setGraphicsEffect(effect)
    group = QParallelAnimationGroup(widget)
    group.setObjectName("fade_slide_in")
    fade = QPropertyAnimation(effect, b"opacity", group)
    fade.setDuration(duration)
    fade.setStartValue(0.0)
    fade.setEndValue(1.0)
    fade.setEasingCurve(QEasingCurve.Type.OutCubic)
    group.addAnimation(fade)
    if offset_x:
        slide = QVariantAnimation(group)
        slide.setDuration(duration)
        slide.setStartValue(1.0)
        slide.setEndValue(0.0)
        slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        slide.valueChanged.connect(
            lambda t: widget.move(home + QPoint(round(offset_x * t), 0)))
        group.addAnimation(slide)

    def finish():
        widget.setGraphicsEffect(None)
        if offset_x:
            # 动画期间布局可能变过（内容增减、窗口缩放）：以布局最新结果为准
            _settle_layout(widget)

    group.finished.connect(finish)
    group.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)


class _Cover(QWidget):
    """盖在控件上的一层底色，透明度从 1 降到 0，看起来就是内容淡入。"""

    def __init__(self, parent: QWidget, color: QColor):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._color = QColor(color)
        self.alpha = 1.0

    def set_alpha(self, alpha: float):
        self.alpha = alpha
        self.update()

    def paintEvent(self, event):
        color = QColor(self._color)
        color.setAlphaF(max(0.0, min(1.0, self.alpha)))
        QPainter(self).fillRect(self.rect(), color)


def cover_fade_in(widget: QWidget, background: QColor, duration: int = DURATION):
    """给含表格、下拉框的大块内容做淡入。

    QGraphicsOpacityEffect 会让整棵子控件树每帧离屏重绘，内容一多就卡；
    对滚动区域和表格里的单元格控件还常画错位置。这里改为盖一层底色、再让它变透明，
    内容本身完全不受影响。
    """
    if not widget.isVisible():
        return
    for old in widget.findChildren(_Cover, options=Qt.FindChildOption.FindDirectChildrenOnly):
        old.deleteLater()
    cover = _Cover(widget, background)
    cover.setGeometry(widget.rect())
    cover.show()
    cover.raise_()

    def step(alpha: float):
        # 动画期间内容可能变高（表格长高），盖层跟着撑满
        cover.setGeometry(widget.rect())
        cover.set_alpha(alpha)

    animate(cover, 1.0, 0.0, step, duration, on_done=cover.deleteLater)


def page_enter(widget: QWidget, background: QColor, offset_x: int = 0, offset_y: int = 0,
               duration: int = DURATION):
    """整页切换：盖层淡入 + 从 (offset_x, offset_y) 滑到原位。
    整页内容很多（卡片墙、聊天列表、表格），不用透明度特效。"""
    cover_fade_in(widget, background, duration)
    # 连续切页时上一轮滑动还没结束：停掉，免得两个动画抢着改位置
    for running in widget.findChildren(QVariantAnimation, "page_enter",
                                       Qt.FindChildOption.FindDirectChildrenOnly):
        running.stop()
    if not widget.isVisible() or not (offset_x or offset_y):
        return
    home = _settle_layout(widget)
    anim = animate(widget, 1.0, 0.0,
                   lambda t: widget.move(home + QPoint(round(offset_x * t), round(offset_y * t))),
                   duration, on_done=lambda: _settle_layout(widget))
    if anim is not None:
        anim.setObjectName("page_enter")


def stack_cross_fade(stack: QStackedWidget, index: int, duration: int = 150):
    """小块内容互换（如顶栏的侧栏按钮 ↔ 返回按钮）：旧页淡出一半时间，换成新页再淡入。

    内容很小，用透明度特效不卡；结束后移除特效，常驻会走离屏渲染。
    连点时先停掉上一轮：不然它淡出结束时会把页面又换回它的目标页。
    """
    for running in stack.findChildren(QVariantAnimation, "stack_cross_fade",
                                      Qt.FindChildOption.FindDirectChildrenOnly):
        running.stop()
    for i in range(stack.count()):
        stack.widget(i).setGraphicsEffect(None)
    if index == stack.currentIndex() or not stack.isVisible():
        stack.setCurrentIndex(index)
        return
    half = duration // 2
    outgoing = stack.currentWidget()
    effect = QGraphicsOpacityEffect(outgoing)
    outgoing.setGraphicsEffect(effect)

    def fade_in():
        outgoing.setGraphicsEffect(None)
        stack.setCurrentIndex(index)
        incoming = stack.currentWidget()
        incoming_effect = QGraphicsOpacityEffect(incoming)
        incoming_effect.setOpacity(0.0)
        incoming.setGraphicsEffect(incoming_effect)
        anim = animate(stack, 0.0, 1.0, incoming_effect.setOpacity, half,
                       on_done=lambda: incoming.setGraphicsEffect(None))
        if anim is not None:
            anim.setObjectName("stack_cross_fade")

    anim = animate(stack, 1.0, 0.0, effect.setOpacity, half, on_done=fade_in)
    if anim is not None:
        anim.setObjectName("stack_cross_fade")


def shake(widget: QWidget, amplitude: int = 6, duration: int = 320):
    """左右抖几下再停住：提示出错，比单纯变红更容易被注意到。"""
    if not widget.isVisible():
        return
    origin = widget.pos()

    def step(t: float):
        offset = math.sin(t * math.pi * 6) * amplitude * (1 - t)
        widget.move(origin + QPoint(round(offset), 0))

    animate(widget, 0.0, 1.0, step, duration, on_done=lambda: widget.move(origin),
            curve=QEasingCurve.Type.Linear)


def bounce(widget: QWidget, depth: float = 0.08, duration: int = 320):
    """按下时以中心缩一下、略微弹过头再停住，作为点击反馈。

    改的是几何尺寸而不是加图形特效：按钮在固定底栏里，动画期间布局不会重排，
    结束时恢复成原来的几何即可；特效会让按钮走离屏渲染，文字发虚。
    """
    if not widget.isVisible():
        return
    home = widget.geometry()

    def step(t: float):
        # 前半段缩小，后半段弹过头一点，振幅随时间衰减到 0
        scale = 1 - depth * math.sin(t * math.pi * 2) * (1 - t)
        width, height = round(home.width() * scale), round(home.height() * scale)
        widget.setGeometry(home.center().x() - width // 2, home.center().y() - height // 2,
                           width, height)

    animate(widget, 0.0, 1.0, step, duration, on_done=lambda: widget.setGeometry(home),
            curve=QEasingCurve.Type.Linear)


class FlowHost(QWidget):
    """装 FlowLayout 的容器：按当前宽度报告建议高度。

    Qt 默认按"建议宽度"（最窄）去算换行后的高度，每项各占一行，报出来的高度虚高好几倍，
    外层布局就照这个高度分配，下面留出一大片空白。
    """

    def sizeHint(self):
        hint = super().sizeHint()
        layout = self.layout()
        if layout is not None and self.width() > 0:
            hint.setHeight(layout.heightForWidth(self.width()))
        return hint


class Collapsible(QWidget):
    """可展开收起的容器：高度动画过渡，收起后隐藏内容，Tab 键不会跳进看不见的输入框。"""

    def __init__(self, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(content)
        self._expanded = expanded
        self._anim: QVariantAnimation | None = None
        self.setMaximumHeight(_UNLIMITED if expanded else 0)
        self.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self._expanded

    def set_expanded(self, expanded: bool):
        if expanded == self._expanded:
            return
        self._expanded = expanded
        # 连点时上一次还没播完：停掉，从当前高度往回走，免得两个动画抢着改高度
        start = self.height() if self._anim is not None else None
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        if expanded:
            self.setVisible(True)
            target = self.sizeHint().height()
            self._anim = animate(self, start or 0, target,
                                 lambda h: self.setMaximumHeight(round(h)),
                                 on_done=self._finish_expand)
        else:
            self._anim = animate(self, self.height(), 0,
                                 lambda h: self.setMaximumHeight(round(h)),
                                 on_done=self._finish_collapse)

    def _finish_expand(self):
        self._anim = None
        self.setMaximumHeight(_UNLIMITED)

    def toggle(self):
        self.set_expanded(not self._expanded)

    def _finish_collapse(self):
        self._anim = None
        # 动画期间又被展开了就不隐藏
        if not self._expanded:
            self.setMaximumHeight(0)
            self.setVisible(False)
