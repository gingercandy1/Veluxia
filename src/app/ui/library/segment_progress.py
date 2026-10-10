"""分层进度条：每段一格，宽度按段在条带上的占比，颜色表示待做 / 进行中 / 完成。

只负责展示；哪一段处于什么状态由页面算好传进来。"""
from enum import Enum

from PySide6.QtWidgets import QFrame, QHBoxLayout, QWidget

CELL_HEIGHT = 8
CELL_GAP = 2


class SegmentState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


class SegmentProgressBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(CELL_HEIGHT)
        self._widths: list[int] = []
        self._cells: list[QFrame] = []
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(CELL_GAP)

    def set_segments(self, widths: list[int]):
        """widths：每段在条带上的像素宽度，只决定各格的相对长短。没变化时不重建。"""
        if widths == self._widths:
            return
        self._widths = list(widths)
        for cell in self._cells:
            self._layout.removeWidget(cell)
            cell.deleteLater()
        self._cells = []
        for index, width in enumerate(widths, start=1):
            cell = QFrame()
            cell.setObjectName("ring_segment_cell")
            cell.setToolTip(self.tr("Segment {0}").format(index))
            self._layout.addWidget(cell, max(width, 1))
            self._cells.append(cell)
        self.set_states([SegmentState.PENDING] * len(widths))

    def set_states(self, states: list[SegmentState]):
        for cell, state in zip(self._cells, states):
            if cell.property("state") == state.value:
                continue
            cell.setProperty("state", state.value)
            cell.style().unpolish(cell)
            cell.style().polish(cell)
