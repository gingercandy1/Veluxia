"""分段画布契约。"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest

from src.app.ui.library.segment_canvas import GAP, SegmentCanvas


def _images():
    images = [QImage(w, 180, QImage.Format.Format_RGB32) for w in (100, 200, 50)]
    for image, color in zip(images, ("red", "green", "blue")):
        image.fill(QColor(color))
    return images


def _canvas():
    canvas = SegmentCanvas()
    canvas.resize(500, 180)
    canvas.set_images(_images())
    return canvas


def test_layout_gaps_and_count(qapp):
    canvas = _canvas()
    assert canvas.segment_count() == 3
    first, second = canvas.segment_rect(0), canvas.segment_rect(1)
    assert first.width() == 100 and first.height() == 180
    assert second.left() - first.right() == GAP
    assert canvas.segment_rect(5).isNull()


def test_click_hits_segment_index_with_modifiers(qapp):
    canvas = _canvas()
    got = []
    canvas.segment_clicked.connect(lambda index, mod: got.append((index, mod)))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(50, 90))
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier,
                     QPoint(200, 90))
    # 间隔里点不中任何段
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(106, 90))
    assert [index for index, _ in got] == [0, 1]
    assert got[1][1] == Qt.KeyboardModifier.ControlModifier


def test_selected_highlight_paints_and_ignores_out_of_range(qapp):
    canvas = _canvas()
    canvas.set_selected([1, 9])
    assert canvas.selected() == [1]
    shot = canvas.grab().toImage()
    rect = canvas.segment_rect(1)
    assert shot.pixelColor(int(rect.left()) + 1, 90) == QColor(255, 214, 102)
    canvas.set_images(_images()[:1])
    assert canvas.selected() == []
    canvas.grab()


def test_wheel_zoom_and_middle_pan(qapp):
    canvas = _canvas()
    canvas.set_zoom(2.0, 0, 0)
    assert canvas.zoom() == 2.0 and canvas.segment_rect(0).height() == 360
    canvas.reset_view()
    left = canvas.segment_rect(0).left()
    QTest.mousePress(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(100, 90))
    QTest.mouseMove(canvas, QPoint(60, 90))
    QTest.mouseRelease(canvas, Qt.MouseButton.MiddleButton, pos=QPoint(60, 90))
    assert canvas.segment_rect(0).left() == left - 40
    canvas.grab()
