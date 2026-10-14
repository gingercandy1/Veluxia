"""框选局部改图的矩形逻辑：归一化框 [x, y, w, h] 与控件坐标的互转、八手柄命中与拖拽。

纯几何，不碰任何控件：RingStrip 和 SegmentCanvas 共用，挪动缩放后框仍贴在原图上。
框以归一化坐标为真相（0~1，相对目标图），控件坐标只在绘制和鼠标交互时换算。
"""
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QPainter, QPen

# 手柄正方形边长的一半；命中时再外扩一点，好点中
HANDLE_HALF = 6.0
HANDLE_HIT_HALF = 10.0
# 归一化最小边长：再小的框改完贴回去也看不出，还浪费一次模型调用
MIN_NORM_SIDE = 0.02
# 换算和发射时保留的小数位，免得浮点噪声让后端校验失败
ROUND_DIGITS = 4

HANDLES = ("top_left", "top", "top_right", "right",
           "bottom_right", "bottom", "bottom_left", "left")


def valid(bbox: object) -> bool:
    """归一化框是否合法：4 个数、落在 0~1 内、宽高不小于最小边。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x, y, w, h = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return False
    return (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
            and MIN_NORM_SIDE <= w <= 1.0 and MIN_NORM_SIDE <= h <= 1.0
            and x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6)


def clamp(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """把框夹回 0~1：先定宽高下限，再收 x/y 让框不越界。"""
    x, y, w, h = (float(v) for v in bbox)
    w = min(1.0, max(MIN_NORM_SIDE, w))
    h = min(1.0, max(MIN_NORM_SIDE, h))
    x = min(1.0 - w, max(0.0, x))
    y = min(1.0 - h, max(0.0, y))
    return (round(x, ROUND_DIGITS), round(y, ROUND_DIGITS),
            round(w, ROUND_DIGITS), round(h, ROUND_DIGITS))


def to_widget(bbox: tuple[float, float, float, float], bounds: QRectF) -> QRectF:
    """归一化框 -> 控件坐标（bounds 是目标图在控件里的位置）。"""
    x, y, w, h = bbox
    return QRectF(bounds.x() + x * bounds.width(),
                  bounds.y() + y * bounds.height(),
                  w * bounds.width(), h * bounds.height())


def to_normalized(rect: QRectF, bounds: QRectF) -> tuple[float, float, float, float] | None:
    """控件坐标矩形 -> 归一化框；bounds 为空或矩形退化时返回 None。"""
    if bounds.width() <= 0 or bounds.height() <= 0:
        return None
    if rect.width() <= 0 or rect.height() <= 0:
        return None
    return clamp(((rect.x() - bounds.x()) / bounds.width(),
                  (rect.y() - bounds.y()) / bounds.height(),
                  rect.width() / bounds.width(),
                  rect.height() / bounds.height()))


def handle_rects(rect: QRectF) -> dict[str, QRectF]:
    """八个手柄的正方形：键是手柄名，值是控件坐标。"""
    cx = rect.x() + rect.width() / 2
    cy = rect.y() + rect.height() / 2
    points = {
        "top_left": (rect.left(), rect.top()),
        "top": (cx, rect.top()),
        "top_right": (rect.right(), rect.top()),
        "right": (rect.right(), cy),
        "bottom_right": (rect.right(), rect.bottom()),
        "bottom": (cx, rect.bottom()),
        "bottom_left": (rect.left(), rect.bottom()),
        "left": (rect.left(), cy),
    }
    return {name: QRectF(QPointF(px - HANDLE_HALF, py - HANDLE_HALF),
                         QPointF(px + HANDLE_HALF, py + HANDLE_HALF))
            for name, (px, py) in points.items()}


def hit(rect: QRectF, pos: QPointF) -> str:
    """命中测试：手柄名、在框内是 "move"、框外是 ""（调用方用来新建）。"""
    for name, square in handle_rects(rect).items():
        grown = square.adjusted(-(HANDLE_HIT_HALF - HANDLE_HALF), -(HANDLE_HIT_HALF - HANDLE_HALF),
                                HANDLE_HIT_HALF - HANDLE_HALF, HANDLE_HIT_HALF - HANDLE_HALF)
        if grown.contains(pos):
            return name
    return "move" if rect.contains(pos) else ""


def apply_drag(start: QRectF, handle: str, delta: QPointF) -> QRectF:
    """按鼠标位移推矩形：手柄缩放（最小 12px）、"move" 平移。越界由调用方收进目标图。"""
    if handle == "move":
        return QRectF(start.x() + delta.x(), start.y() + delta.y(),
                      start.width(), start.height())
    left, top, right, bottom = start.left(), start.top(), start.right(), start.bottom()
    if "left" in handle:
        left += delta.x()
    if "right" in handle:
        right += delta.x()
    if "top" in handle:
        top += delta.y()
    if "bottom" in handle:
        bottom += delta.y()
    if right - left < 12:
        # 压到最小宽度时按住哪边、哪边不动，另一边被推回去
        if "left" in handle:
            left = right - 12
        else:
            right = left + 12
    if bottom - top < 12:
        if "top" in handle:
            top = bottom - 12
        else:
            bottom = top + 12
    return QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()


def paint(painter: QPainter, rect: QRectF, bounds: QRectF):
    """画框选：框外压暗、黄色描边、八个白手柄。和选中接缝的黄框同色。"""
    painter.save()
    painter.setPen(QPen(QColor(0, 0, 0, 0)))
    painter.setBrush(QColor(0, 0, 0, 110))
    # 框外四个方向各一块，不盖住框里的画面
    painter.drawRect(QRectF(bounds.topLeft(), QPointF(bounds.right(), rect.top())))
    painter.drawRect(QRectF(QPointF(bounds.left(), rect.bottom()), bounds.bottomRight()))
    painter.drawRect(QRectF(QPointF(bounds.left(), rect.top()),
                            QPointF(rect.left(), rect.bottom())))
    painter.drawRect(QRectF(QPointF(rect.right(), rect.top()),
                            QPointF(bounds.right(), rect.bottom())))
    painter.setBrush(QColor(0, 0, 0, 0))
    painter.setPen(QPen(QColor(255, 214, 102), 2))
    painter.drawRect(rect)
    painter.setPen(QPen(QColor(30, 30, 30), 1))
    painter.setBrush(QColor(255, 255, 255))
    for square in handle_rects(rect).values():
        painter.drawRect(square)
    painter.restore()
