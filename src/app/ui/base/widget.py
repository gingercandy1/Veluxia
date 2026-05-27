from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget, QStyleOption, QStyle


class BaseWidget(QWidget):
    def __init__(self, parent=None):
        super(BaseWidget, self).__init__(parent)

    def paintEvent(self, event):
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
        super().paintEvent(event)