"""执行中的动效：跳动的三个点、带流光的进度条。

资源包一跑就是几分钟，纯文字"执行中"看不出程序还活着，动起来用户才放心。
相位都取自 time.monotonic()，不同控件各自刷新也能保持同步。
"""
import math
import time

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QSizePolicy, QWidget

from src.app.ui.base.widget import BaseWidget

ACCENT = QColor("#7B9DBC")
STATUS_COLOR = {
    "done": QColor("#4ec994"),
    "review": QColor("#e5c07b"),
    "error": QColor("#f87171"),
}
# 约 30 帧：动效够顺，空闲时计时器停掉不占 CPU
FRAME_MS = 33

DOT_RADIUS = 2.0
DOT_GAP = 5.0
DOTS_WIDTH = DOT_RADIUS * 2 * 3 + DOT_GAP * 2
_DOT_PERIOD = 1.2
_DOT_HOP = 3.0


def paint_dots(painter: QPainter, left: float, center_y: float, color: QColor = ACCENT):
    """三个点依次跳起再落下，像聊天里的"正在输入"。"""
    now = time.monotonic()
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    for index in range(3):
        # 每个点错开一点相位；周期前 40% 跳一下，其余时间停在原位
        phase = (now / _DOT_PERIOD - index * 0.15) % 1.0
        hop = math.sin(math.pi * phase / 0.4) * _DOT_HOP if phase < 0.4 else 0.0
        x = left + DOT_RADIUS + index * (DOT_RADIUS * 2 + DOT_GAP)
        painter.drawEllipse(QPointF(x, center_y - hop), DOT_RADIUS, DOT_RADIUS)
    painter.restore()


class FrameTicker:
    """按需启停的刷新计时器：只有真在动的控件才每帧重绘。"""

    def __init__(self, target: QWidget):
        self._timer = QTimer(target)
        self._timer.setInterval(FRAME_MS)
        self._timer.timeout.connect(target.update)

    def set_running(self, running: bool):
        if running and not self._timer.isActive():
            self._timer.start()
        elif not running and self._timer.isActive():
            self._timer.stop()

    def is_running(self) -> bool:
        return self._timer.isActive()


class ProgressStrip(BaseWidget):
    """细进度条：填充按状态着色；执行中叠一道来回扫过的流光，进度暂时不涨也能看出在跑。"""
    TRACK_COLOR = QColor(255, 255, 255, 15)
    _SWEEP_PERIOD = 1.6

    def __init__(self, height: int = 4, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value = 0
        self._status = "pending"
        self._ticker = FrameTicker(self)

    def value(self) -> int:
        return self._value

    def status(self) -> str:
        return self._status

    def is_active(self) -> bool:
        return self._ticker.is_running()

    def set_progress(self, value: int, status: str, active: bool = False):
        self._value = max(0, min(100, value))
        self._status = status
        self._ticker.set_running(active)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())
        radius = rect.height() / 2
        clip = QPainterPath()
        clip.addRoundedRect(rect, radius, radius)
        painter.setClipPath(clip)
        painter.fillRect(rect, self.TRACK_COLOR)

        if self._value:
            fill = QRectF(rect.left(), rect.top(), rect.width() * self._value / 100, rect.height())
            painter.fillRect(fill, STATUS_COLOR.get(self._status, ACCENT))

        if self.is_active():
            # 光带宽度取整条的 35%，从左侧外面扫到右侧外面，首尾不会突然出现
            band = rect.width() * 0.35
            phase = (time.monotonic() / self._SWEEP_PERIOD) % 1.0
            left = -band + (rect.width() + band) * phase
            gradient = QLinearGradient(left, 0, left + band, 0)
            glow = QColor(ACCENT.lighter(150))
            for stop, alpha in ((0.0, 0), (0.5, 170), (1.0, 0)):
                glow.setAlpha(alpha)
                gradient.setColorAt(stop, glow)
            painter.fillRect(QRectF(left, rect.top(), band, rect.height()), gradient)
