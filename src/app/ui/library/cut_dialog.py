"""确认切段位置的对话框：条带上拖线 / Shift+左键加线 / Shift+右键删线，确定才提交。

自动切割只是建议，必须让用户在这里看过再提交；取消就什么都不变，所以对话框不直接改任何数据，
由调用方在 accepted 后读 cuts()。"""
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from src.app.ui.library.ring_math import segment_widths
from src.app.ui.library.ring_strip import RingStrip

STRIP_HEIGHT = 220


class CutDialog(QDialog):
    def __init__(self, images: list[QImage], cuts: list[int], warning: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Confirm segment cuts"))
        self.setObjectName("cut_dialog")
        self.resize(900, 360)
        self._total = sum(image.width() for image in images if not image.isNull())

        self.warning_label = QLabel(warning)
        self.warning_label.setObjectName("cut_dialog_warning")
        self.warning_label.setWordWrap(True)
        self.warning_label.setVisible(bool(warning))

        hint = QLabel(self.tr("Drag a red line to move it · Shift+left click adds a cut · "
                              "Shift+right click removes the nearest cut"))
        hint.setObjectName("detail_muted")
        hint.setWordWrap(True)

        self.strip = RingStrip()
        self.strip.setMinimumHeight(STRIP_HEIGHT)
        self.strip.set_images(images)
        self.strip.set_cuts(cuts)
        self.strip.cuts_edited.connect(self._refresh_summary)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("detail_muted")
        self.summary_label.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.buttons = buttons

        layout = QVBoxLayout(self)
        layout.addWidget(self.warning_label)
        layout.addWidget(hint)
        layout.addWidget(self.strip, 1)
        layout.addWidget(self.summary_label)
        layout.addWidget(buttons)
        self._refresh_summary()

    def cuts(self) -> list[int]:
        return self.strip.cuts()

    def _refresh_summary(self, *_):
        cuts = self.strip.cuts()
        if not cuts or not self._total:
            self.summary_label.setText("")
            return
        widths = segment_widths(cuts, self._total)
        self.summary_label.setText(
            self.tr("{0} segments: {1} px").format(
                len(widths), " · ".join(str(width) for width in widths)))
