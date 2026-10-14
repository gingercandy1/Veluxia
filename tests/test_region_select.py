"""框选矩形几何契约：合法性、坐标互转、手柄命中与拖拽。不碰控件、不连后端。"""
from PySide6.QtCore import QPointF, QRectF

from src.app.ui.library import region_select
from src.app.ui.library.region_select import HANDLES


def _bounds():
    return QRectF(100.0, 50.0, 400.0, 200.0)


def test_valid_accepts_normalized_box_only():
    assert region_select.valid([0.1, 0.2, 0.3, 0.4])
    assert region_select.valid((0.0, 0.0, 1.0, 1.0))
    assert not region_select.valid([0.1, 0.2, 0.3])          # 不是 4 个数
    assert not region_select.valid([0.1, 0.2, 0.0, 0.4])     # 宽为 0
    assert not region_select.valid([0.1, 0.2, 0.01, 0.4])    # 小于最小边
    assert not region_select.valid([0.9, 0.9, 0.3, 0.3])     # 越界
    assert not region_select.valid(["a", 0, 1, 1])
    assert not region_select.valid(None)


def test_clamp_keeps_the_box_inside():
    assert region_select.clamp((0.9, 0.9, 0.3, 0.3)) == (0.7, 0.7, 0.3, 0.3)
    assert region_select.clamp((-0.1, -0.2, 0.5, 0.5)) == (0.0, 0.0, 0.5, 0.5)
    x, y, w, h = region_select.clamp((0.5, 0.5, 0.001, 2.0))
    assert (w, h) == (region_select.MIN_NORM_SIDE, 1.0) and x + w <= 1.0 and y + h <= 1.0


def test_widget_and_normalized_round_trip():
    bounds = _bounds()
    bbox = (0.25, 0.25, 0.5, 0.5)
    rect = region_select.to_widget(bbox, bounds)
    assert (rect.x(), rect.y(), rect.width(), rect.height()) == (200.0, 100.0, 200.0, 100.0)
    assert region_select.to_normalized(rect, bounds) == bbox
    assert region_select.to_normalized(rect, QRectF()) is None
    assert region_select.to_normalized(QRectF(1, 1, 0, 0), bounds) is None


def test_hit_reports_handles_move_and_outside():
    rect = QRectF(100.0, 100.0, 200.0, 100.0)
    assert region_select.hit(rect, QPointF(100, 100)) == "top_left"
    assert region_select.hit(rect, QPointF(300, 200)) == "bottom_right"
    assert region_select.hit(rect, QPointF(200, 100)) == "top"
    assert region_select.hit(rect, QPointF(200, 150)) == "move"
    assert region_select.hit(rect, QPointF(10, 10)) == ""
    assert set(region_select.handle_rects(rect)) == set(HANDLES)


def test_apply_drag_resizes_moves_and_floors_at_minimum():
    rect = QRectF(100.0, 100.0, 200.0, 100.0)
    grown = region_select.apply_drag(rect, "bottom_right", QPointF(10, 20))
    assert (grown.width(), grown.height()) == (210.0, 120.0)
    moved = region_select.apply_drag(rect, "move", QPointF(-5, 7))
    assert (moved.x(), moved.y(), moved.width(), moved.height()) == (95.0, 107.0, 200.0, 100.0)
    # 往里压过头时按住的边不动，最小 12px
    assert region_select.apply_drag(rect, "right", QPointF(-500, 0)).width() == 12.0
    assert region_select.apply_drag(rect, "top", QPointF(0, 500)).height() == 12.0
