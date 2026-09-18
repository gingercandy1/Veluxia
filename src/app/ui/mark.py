"""
Veluxia 品牌标记：一缕云雾向右上角散开，中央显露出四角星。
几何与时间线来自设计原型（Variant C · Parallax Gust）：
https://claude.ai/artifact/QxG3A4uknve6kHKogocziQ

坐标系统一用「以中心为原点、边长 100 的正方形」，绘制时再按控件尺寸整体缩放，
所以这里的路径/位移数值和原型 SVG 里的完全对应，改动时可以对照着改。
"""
import math
from typing import Optional

from PySide6.QtCore import QPointF, QSize, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QWidget

# ── 路径（相对中心点，原型 viewBox 100x100 平移 -50,-50 后的坐标） ──

def _star_path(radius: float = 44.0, steps_per_quadrant: int = 24) -> QPainterPath:
    """四角星走的是星形线（astroid）：x=R·cos³t，y=R·sin³t。
    这条曲线的凹陷弧度是数学上"刚刚好"的，比手调贝塞尔控制点更对称、更干净，
    这也是市面上大多数四角"sparkle"图标的真实几何。"""
    path = QPainterPath()
    steps = steps_per_quadrant * 4
    for i in range(steps + 1):
        t = (i / steps) * 2 * math.pi
        x = radius * math.cos(t) ** 3
        y = radius * math.sin(t) ** 3
        if i == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    path.closeSubpath()
    return path


def _wisp_path() -> QPainterPath:
    """飘散消失的云缕：运动中途就淡出了，细节不重要，形状简单即可。"""
    path = QPainterPath()
    path.moveTo(0, -19)
    path.cubicTo(13, -21, 25, -12, 25, -1)
    path.cubicTo(25, 9, 16, 15, 5, 11)
    path.cubicTo(-5, 7, -9, -4, -7, -12)
    path.cubicTo(-6, -16, -3, -18, 0, -19)
    path.closeSubpath()
    return path


def _cloud_path() -> QPainterPath:
    """留在角落、最终会定格进图标的那朵云：一串圆深度重叠揉成一朵蓬松的云。
    只用圆（不掺矩形底座）是为了避免相邻图形只是"擦边"重叠时，
    并集在两者之间露出一道背景色的接缝——圆与圆之间重叠深度都留够了半径的一半以上。
    """
    circles = [
        (-14.0, 3.0, 9.0),
        (-3.0, -4.0, 12.0),
        (9.0, 1.0, 10.0),
        (18.0, 5.0, 7.0),
    ]
    cloud = QPainterPath()
    for cx, cy, r in circles:
        bump = QPainterPath()
        bump.addEllipse(QPointF(cx, cy), r, r)
        cloud = cloud.united(bump)
    return cloud


STAR_PATH = _star_path()
WISP_PATH = _wisp_path()
CLOUD_PATH = _cloud_path()

# 背景薄雾（c-bg）：覆盖全图 -> 缩小飘向右上角并淡出
_BG_START = dict(pos=(0.0, 0.0), scale=1.9, opacity=1.0)
_BG_END = dict(pos=(30.0, -30.0), scale=0.36, opacity=0.0)
_BG_DURATION_MS = 1550

# 前景云缕（c-f）：两缕飘散消失
_FRONT_WISPS = [
    dict(start=(-14.0, 10.0), s0=0.5, end=(-40.0, 20.0), s1=0.14, o1=0.0),
    dict(start=(12.0, 14.0), s0=0.46, end=(16.0, 38.0), s1=0.12, o1=0.0),
]
# 留在右上角、最终定格进图标的那朵云（用更好认的云朵剪影，不用云缕那个水滴形）
_KEEP_CLOUD = dict(start=(10.0, -12.0), s0=0.42, end=(23.0, -26.0), s1=0.48, o1=1.0)
_FRONT_DURATION_MS = 1050

# 四角星：延迟出现，缩放+轻微旋转显露
_STAR_DELAY_MS = 350
_STAR_DURATION_MS = 1600
_STAR_SCALE_FROM = 0.86
_STAR_ROTATION_TO = 360.0  # 转一整圈回到正位：动画有旋转的动感，定格图标又是端正对称的

TOTAL_DURATION_MS = _STAR_DELAY_MS + _STAR_DURATION_MS  # 1950ms，覆盖所有分轨


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease_in_out_cubic(t: float) -> float:
    """近似原型里 CSS 的 cubic-bezier(.4,0,.2,1) / (.32,0,.2,1)：先快后慢的标准缓动。"""
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    p = -2 * t + 2
    return 1 - (p ** 3) / 2


def _track_progress(elapsed_ms: float, delay_ms: float, duration_ms: float) -> float:
    if duration_ms <= 0:
        return 1.0
    t = (elapsed_ms - delay_ms) / duration_ms
    return _ease_in_out_cubic(t) if t > 0 else 0.0


def paint_mark(painter: QPainter, size: float, color: QColor, elapsed_ms: float, fill_ratio: float = 0.72) -> None:
    """在 painter 当前坐标原点（应已平移到控件中心）绘制第 elapsed_ms 毫秒的画面。"""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scale = size * fill_ratio / 100.0
    painter.scale(scale, scale)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)

    # 背景薄雾
    e = _track_progress(elapsed_ms, 0.0, _BG_DURATION_MS)
    bg_opacity = _lerp(_BG_START["opacity"], _BG_END["opacity"], e)
    if bg_opacity > 0.004:
        bg_scale = _lerp(_BG_START["scale"], _BG_END["scale"], e)
        bg_pos = QPointF(
            _lerp(_BG_START["pos"][0], _BG_END["pos"][0], e),
            _lerp(_BG_START["pos"][1], _BG_END["pos"][1], e),
        )
        painter.save()
        painter.setOpacity(bg_opacity)
        painter.translate(bg_pos)
        painter.scale(bg_scale, bg_scale)
        painter.drawPath(WISP_PATH)
        painter.restore()

    # 前景云缕：两缕飘散消失
    e = _track_progress(elapsed_ms, 0.0, _FRONT_DURATION_MS)
    for wisp in _FRONT_WISPS:
        opacity = _lerp(1.0, wisp["o1"], e)
        if opacity <= 0.004:
            continue
        w_scale = _lerp(wisp["s0"], wisp["s1"], e)
        w_pos = QPointF(
            _lerp(wisp["start"][0], wisp["end"][0], e),
            _lerp(wisp["start"][1], wisp["end"][1], e),
        )
        painter.save()
        painter.setOpacity(opacity)
        painter.translate(w_pos)
        painter.scale(w_scale, w_scale)
        painter.drawPath(WISP_PATH)
        painter.restore()

    # 留在角落、最终定格进图标的那朵云
    cloud = _KEEP_CLOUD
    c_scale = _lerp(cloud["s0"], cloud["s1"], e)
    c_pos = QPointF(
        _lerp(cloud["start"][0], cloud["end"][0], e),
        _lerp(cloud["start"][1], cloud["end"][1], e),
    )
    painter.save()
    painter.translate(c_pos)
    painter.scale(c_scale, c_scale)
    painter.drawPath(CLOUD_PATH)
    painter.restore()

    # 四角星
    e = _track_progress(elapsed_ms, _STAR_DELAY_MS, _STAR_DURATION_MS)
    if elapsed_ms >= _STAR_DELAY_MS or e > 0:
        star_opacity = _lerp(0.0, 1.0, e)
        star_scale = _lerp(_STAR_SCALE_FROM, 1.0, e)
        star_rotation = _lerp(0.0, _STAR_ROTATION_TO, e)
        painter.save()
        painter.setOpacity(star_opacity)
        painter.rotate(star_rotation)
        painter.scale(star_scale, star_scale)
        painter.drawPath(STAR_PATH)
        painter.restore()

    painter.restore()


def render_mark_pixmap(px_size: int, color: QColor = QColor("#F2F3F5"), elapsed_ms: Optional[float] = None) -> QPixmap:
    """渲染某一帧到 QPixmap；elapsed_ms=None 时渲染终帧（星 + 右上角一缕云），用作静态图标。"""
    if elapsed_ms is None:
        elapsed_ms = TOTAL_DURATION_MS
    pixmap = QPixmap(px_size, px_size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.translate(px_size / 2, px_size / 2)
    paint_mark(painter, px_size, color, elapsed_ms, fill_ratio=0.8)
    painter.end()
    return pixmap


def build_app_icon(color: QColor = QColor("#F2F3F5")) -> QIcon:
    """终帧图标的多分辨率 QIcon，供窗口/任务栏使用。"""
    icon = QIcon()
    for px_size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(render_mark_pixmap(px_size, color))
    return icon


class MarkWidget(QWidget):
    """播放"云雾散去、四角星显露"动画的控件；播放结束后自动从头循环，作为加载指示器。"""

    finished = Signal()

    def __init__(self, parent=None, color: QColor = QColor("#F2F3F5"), fill_ratio: float = 0.72, loop: bool = True):
        super().__init__(parent)
        self._color = color
        self._fill_ratio = fill_ratio
        self._loop = loop
        self._elapsed_ms = 0.0
        self._running = False

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._tick)

        self._loop_gap = QTimer(self)
        self._loop_gap.setSingleShot(True)
        self._loop_gap.timeout.connect(self._restart)

    def sizeHint(self):
        return QSize(160, 160)

    def play(self):
        self._elapsed_ms = 0.0
        self._running = True
        self._timer.start()
        self.update()

    def stop(self):
        self._running = False
        self._timer.stop()
        self._loop_gap.stop()

    def show_final_frame(self):
        self.stop()
        self._elapsed_ms = TOTAL_DURATION_MS
        self.update()

    def _tick(self):
        self._elapsed_ms += self._timer.interval()
        if self._elapsed_ms >= TOTAL_DURATION_MS:
            self._elapsed_ms = TOTAL_DURATION_MS
            self._timer.stop()
            self.update()
            self.finished.emit()
            if self._loop:
                self._loop_gap.start(500)  # 定格片刻再重播，别一直转得让人心烦
            return
        self.update()

    def _restart(self):
        if self._running:
            self.play()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.translate(self.width() / 2, self.height() / 2)
        size = min(self.width(), self.height())
        paint_mark(painter, size, self._color, self._elapsed_ms, self._fill_ratio)
