"""看图浮层的缩放、拖拽和关闭行为（离屏，不连后端）。"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from src.app.ui.message.image_preview import ImagePreviewOverlay


def _host_with_image(tmp_path, qapp):
    host = QWidget()
    host.resize(800, 600)
    image = QImage(400, 300, QImage.Format.Format_RGB32)
    image.fill(QColor("green"))
    path = tmp_path / "shot.png"
    image.save(str(path))
    host.show()
    return host, str(path)


def test_wheel_zooms_in_and_out_within_limits(tmp_path, qapp):
    host, path = _host_with_image(tmp_path, qapp)
    overlay = ImagePreviewOverlay(path, host)
    start = overlay._label.width()
    overlay._zoom_at(1.15, QPoint(400, 300))
    assert overlay._label.width() > start
    for _ in range(200):
        overlay._zoom_at(1.15, QPoint(400, 300))
    assert overlay._zoom == ImagePreviewOverlay.MAX_ZOOM
    for _ in range(400):
        overlay._zoom_at(1 / 1.15, QPoint(400, 300))
    assert overlay._zoom == ImagePreviewOverlay.MIN_ZOOM
    overlay.close()
    host.close()


def test_drag_pans_without_closing_but_click_closes(tmp_path, qapp):
    host, path = _host_with_image(tmp_path, qapp)
    overlay = ImagePreviewOverlay(path, host)
    QTest.mousePress(overlay, Qt.MouseButton.LeftButton, pos=QPoint(400, 300))
    QTest.mouseMove(overlay, QPoint(440, 320))
    QTest.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=QPoint(440, 320))
    assert overlay._pan != QPoint(0, 0)
    assert overlay.isVisible()
    QTest.mousePress(overlay, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    QTest.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    assert not overlay.isVisible()
    host.close()


def test_double_click_resets_zoom_and_pan(tmp_path, qapp):
    host, path = _host_with_image(tmp_path, qapp)
    overlay = ImagePreviewOverlay(path, host)
    overlay._zoom_at(2.0, QPoint(400, 300))
    overlay._pan = QPoint(30, -10)
    overlay._layout_children()
    QTest.mouseDClick(overlay, Qt.MouseButton.LeftButton, pos=QPoint(10, 10))
    assert overlay._zoom == 1.0
    assert overlay._pan == QPoint(0, 0)
    overlay.close()
    host.close()
