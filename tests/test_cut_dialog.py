"""切段确认对话框契约。"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog

from src.app.ui.library.cut_dialog import CutDialog
from src.app.ui.library.ring_math import segment_widths


def _images():
    return [QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)]


def test_segment_widths_wraps_and_single_cut():
    assert segment_widths([100, 350], 600) == [250, 350]
    assert segment_widths([350], 600) == [600]


def test_initial_cuts_summary_and_warning(qapp):
    dialog = CutDialog(_images(), [350, 100])
    assert dialog.cuts() == [100, 350]
    assert dialog.warning_label.isHidden()
    assert "2 segments" in dialog.summary_label.text()
    assert "250" in dialog.summary_label.text() and "350" in dialog.summary_label.text()

    warned = CutDialog(_images(), [100], warning="3 seams are not ready")
    assert not warned.warning_label.isHidden()
    assert warned.warning_label.text() == "3 seams are not ready"
    assert "1 segments" in warned.summary_label.text()


def test_editing_strip_updates_cuts_and_summary(qapp):
    dialog = CutDialog(_images(), [100, 350])
    strip = dialog.strip
    strip.resize(400, 180)
    strip.set_offset(0)
    strip.grab()
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     QPoint(200, 90))
    # 条带最小高度 220，缩放比不是 1，新切点的具体值不固定，只断言多了一条且仍升序
    cuts = dialog.cuts()
    assert len(cuts) == 3 and cuts == sorted(cuts) and 100 in cuts and 350 in cuts
    assert "3 segments" in dialog.summary_label.text()


def test_ok_and_cancel_results(qapp):
    dialog = CutDialog(_images(), [100, 350])
    dialog.buttons.accepted.emit()
    assert dialog.result() == QDialog.DialogCode.Accepted
    dialog = CutDialog(_images(), [100, 350])
    dialog.buttons.rejected.emit()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert dialog.cuts() == [100, 350]
