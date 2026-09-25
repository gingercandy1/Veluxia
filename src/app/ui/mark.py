"""
Veluxia 品牌标记：五片宽窄不一、朝同一方向甩出的月牙叶片组成的漩涡。

坐标系统一用「以中心为原点、边长 100 的正方形」，绘制时再按控件尺寸整体缩放。
不直接用单位圆坐标：QPainterPath 的布尔运算会按坐标单位把曲线拍平成折线，
单位太小时圆会变成多边形。

动画（加载页用）：
  1. 展开：中心核放大，五片叶片依次沿各自的螺线长出，整体同时旋入；
  2. 等待：展开后保持匀速慢转 + 背后光晕呼吸，表明程序还在工作，不会像定格那样像卡死。
旋转角 = 旋入缓动 + 匀速自转，两段叠加，所以从展开过渡到慢转没有速度突变。
"""
import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import (QColor, QIcon, QLinearGradient, QPainter, QPainterPath, QPixmap,
                           QRadialGradient)
from PySide6.QtWidgets import QWidget

# ── 几何（设计稿方案 D：5 叶、长甩尾） ──
_BLADE_COUNT = 5
_SEED = 5                     # 固定种子：叶片宽窄"随机"但每次绘制都一样
_TWIST = 2.1                  # 叶尖相对叶根沿旋转方向偏转的弧度
_RADIUS = 44.0                # 最长叶尖到中心的距离
_CORE_RADIUS = _RADIUS * 0.24
_WIDTH_RANGE = (0.9, 1.6)     # 叶根张角，占一个扇区角度的倍数；>1 时相邻叶根重叠成实心核
_LENGTH_RANGE = (0.78, 1.0)   # 叶长，占 _RADIUS 的比例
_SAMPLES = 48


@dataclass(frozen=True)
class _Blade:
    theta: float   # 叶根中线角度
    width: float   # 叶根张角
    length: float  # 叶尖半径


def _make_blades() -> list[_Blade]:
    rng = random.Random(_SEED)
    step = 2 * math.pi / _BLADE_COUNT
    blades = []
    for i in range(_BLADE_COUNT):
        theta = i * step + rng.uniform(-0.1, 0.1) * step
        width = step * rng.uniform(*_WIDTH_RANGE)
        length = _RADIUS * rng.uniform(*_LENGTH_RANGE)
        blades.append(_Blade(theta, width, length))
    return blades


BLADES = _make_blades()


def _polar(r: float, a: float) -> QPointF:
    return QPointF(r * math.cos(a), r * math.sin(a))


def blade_path(blade: _Blade, growth: float = 1.0) -> QPainterPath:
    """单片叶片。沿半径采样两条螺线：前缘转得早（t^0.7）外凸，后缘转得晚（t^1.6）内凹，
    两边在叶尖汇合成月牙。growth<1 时叶长和偏转一起按比例缩小，看起来是沿螺线"长出来"。"""
    length = _CORE_RADIUS + (blade.length - _CORE_RADIUS) * growth
    twist = _TWIST * growth
    lead, trail = [], []
    for s in range(_SAMPLES + 1):
        t = s / _SAMPLES
        r = _CORE_RADIUS + (length - _CORE_RADIUS) * t
        half = blade.width / 2 * (1 - t) ** 1.2
        lead.append(_polar(r, blade.theta + half + twist * t ** 0.7))
        trail.append(_polar(r, blade.theta - half + twist * t ** 1.6))
    path = QPainterPath()
    path.moveTo(trail[0])
    for point in lead:
        path.lineTo(point)
    for point in reversed(trail):
        path.lineTo(point)
    path.closeSubpath()
    return path


def _core_path(scale: float = 1.0) -> QPainterPath:
    path = QPainterPath()
    radius = _CORE_RADIUS * 1.08 * scale
    path.addEllipse(QPointF(0, 0), radius, radius)
    return path


def _swirl_path() -> QPainterPath:
    """完整的静态漩涡（叶片并集），用于图标。"""
    path = _core_path()
    for blade in BLADES:
        path = path.united(blade_path(blade))
    return path


SWIRL_PATH = _swirl_path()

# ── 配色 ──
GLYPH_COLOR = QColor("#F2F3F5")
_TILE_TOP = QColor("#9AC5EC")
_TILE_MID = QColor("#7B9DBC")
_TILE_BOTTOM = QColor("#6a5fb0")
_GLOW_COLOR = QColor(154, 197, 236)

# ── 时间线 ──
_CORE_DURATION_MS = 300
_BLADE_DELAY_MS = 120
_BLADE_STAGGER_MS = 90
_BLADE_DURATION_MS = 650
_SPIN_IN_DURATION_MS = 1100
_SPIN_IN_DEGREES = 140.0      # 旋入：从 -140° 转回正位
_SPIN_DEG_PER_MS = 360.0 / 6000.0  # 等待时慢转：6 秒一圈
_GLOW_PERIOD_MS = 2400.0

INTRO_DURATION_MS = _BLADE_DELAY_MS + _BLADE_STAGGER_MS * (_BLADE_COUNT - 1) + _BLADE_DURATION_MS


def _ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def _progress(elapsed_ms: float, delay_ms: float, duration_ms: float) -> float:
    return _ease_out_cubic((elapsed_ms - delay_ms) / duration_ms)


def paint_mark(painter: QPainter, size: float, color: QColor, elapsed_ms: Optional[float],
               fill_ratio: float = 0.72, spin: bool = True) -> None:
    """在 painter 当前坐标原点（应已平移到控件中心）绘制第 elapsed_ms 毫秒的画面。
    elapsed_ms=None 画静态终帧；spin=False 时展开后停在正位不再自转。"""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scale = size * fill_ratio / 100.0
    painter.scale(scale, scale)
    painter.setPen(Qt.PenStyle.NoPen)

    if elapsed_ms is None:
        painter.setBrush(color)
        painter.drawPath(SWIRL_PATH)
        painter.restore()
        return

    # 背后光晕：展开时淡入，之后缓慢呼吸
    intro = _progress(elapsed_ms, 0.0, INTRO_DURATION_MS)
    breath = 0.5 + 0.5 * math.sin(2 * math.pi * elapsed_ms / _GLOW_PERIOD_MS)
    glow = QRadialGradient(QPointF(0, 0), _RADIUS * 1.25)
    glow_color = QColor(_GLOW_COLOR)
    glow_color.setAlphaF(intro * (0.10 + 0.10 * breath))
    glow.setColorAt(0, glow_color)
    glow.setColorAt(1, QColor(0, 0, 0, 0))
    painter.setBrush(glow)
    painter.drawEllipse(QPointF(0, 0), _RADIUS * 1.25, _RADIUS * 1.25)

    rotation = -_SPIN_IN_DEGREES * (1 - _progress(elapsed_ms, 0.0, _SPIN_IN_DURATION_MS))
    if spin:
        rotation += elapsed_ms * _SPIN_DEG_PER_MS
    painter.rotate(rotation)

    # 叶片不透明且同色，重叠部分看不出来，所以逐片直接画，不用每帧做并集
    painter.setBrush(color)
    for i, blade in enumerate(BLADES):
        growth = _progress(elapsed_ms, _BLADE_DELAY_MS + i * _BLADE_STAGGER_MS, _BLADE_DURATION_MS)
        if growth > 0.01:
            painter.drawPath(blade_path(blade, growth))
    core = _progress(elapsed_ms, 0.0, _CORE_DURATION_MS)
    if core > 0.01:
        painter.drawPath(_core_path(core))

    painter.restore()


def render_icon_pixmap(px_size: int) -> QPixmap:
    """应用图标：渐变圆角底板 + 白色漩涡。带底板是为了在浅色任务栏上也看得见。"""
    pixmap = QPixmap(px_size, px_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    margin = px_size * 0.06
    tile = QRectF(margin, margin, px_size - 2 * margin, px_size - 2 * margin)
    gradient = QLinearGradient(tile.topLeft(), tile.bottomRight())
    gradient.setColorAt(0, _TILE_TOP)
    gradient.setColorAt(0.55, _TILE_MID)
    gradient.setColorAt(1, _TILE_BOTTOM)
    tile_path = QPainterPath()
    tile_path.addRoundedRect(tile, tile.width() * 0.22, tile.width() * 0.22)
    painter.fillPath(tile_path, gradient)

    painter.translate(px_size / 2, px_size / 2)
    paint_mark(painter, tile.width(), QColor("#ffffff"), None, fill_ratio=0.88)
    painter.end()
    return pixmap


def build_app_icon() -> QIcon:
    """多分辨率 QIcon，供窗口/任务栏使用；各尺寸单独绘制，小尺寸不靠缩放糊掉。"""
    icon = QIcon()
    for px_size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(render_icon_pixmap(px_size))
    return icon


class MarkWidget(QWidget):
    """播放漩涡展开动画的控件。loop=True 时展开后持续慢转（加载指示）；
    loop=False 时展开一次后停在正位并发出 finished。"""

    finished = Signal()

    def __init__(self, parent=None, color: QColor = GLYPH_COLOR, fill_ratio: float = 0.72,
                 loop: bool = True):
        super().__init__(parent)
        self._color = color
        self._fill_ratio = fill_ratio
        self._loop = loop
        self._elapsed_ms: Optional[float] = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._tick)

    def sizeHint(self):
        return QSize(160, 160)

    def play(self):
        self._elapsed_ms = 0.0
        self._timer.start()
        self.update()

    def stop(self):
        self._timer.stop()

    def show_final_frame(self):
        self.stop()
        self._elapsed_ms = None
        self.update()

    def _tick(self):
        self._elapsed_ms += self._timer.interval()
        if not self._loop and self._elapsed_ms >= INTRO_DURATION_MS:
            self.show_final_frame()
            self.finished.emit()
            return
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.translate(self.width() / 2, self.height() / 2)
        size = min(self.width(), self.height())
        paint_mark(painter, size, self._color, self._elapsed_ms, self._fill_ratio, spin=self._loop)
