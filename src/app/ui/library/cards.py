from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.pack_status import Progress, status_text

PACK_THUMB = QSize(184, 124)
ASSET_THUMB = QSize(148, 116)


def repolish(widget: QWidget):
    """动态属性改了以后 qss 不会自己重算，要手动刷新样式。"""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class Card(BaseWidget):
    """卡片：缩略图 + 名称 + 进度。没有图片的卡片（对话、音效）在缩略图位置显示分类文字。"""
    clicked = Signal()
    double_clicked = Signal()

    def __init__(self, thumb_size: QSize, placeholder: str, category: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("library_card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._thumb_size = thumb_size

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.thumb = QLabel(placeholder)
        self.thumb.setObjectName("card_thumb")
        # 分类决定占位底色，qss 按 category 属性挂色
        self.thumb.setProperty("category", category)
        self.thumb.setFixedSize(thumb_size)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb)

        self.name_label = QLabel()
        self.name_label.setObjectName("card_name")
        self.name_label.setFixedWidth(thumb_size.width())
        layout.addWidget(self.name_label)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.meta_label = QLabel()
        self.meta_label.setObjectName("card_meta")
        self.badge = QLabel()
        self.badge.setObjectName("card_badge")
        row.addWidget(self.meta_label)
        row.addStretch()
        row.addWidget(self.badge)
        layout.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setObjectName("card_progress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        layout.addWidget(self.progress)

        self.setFixedWidth(thumb_size.width() + 16)

    def set_name(self, name: str):
        self.name_label.setToolTip(name)
        metrics = self.name_label.fontMetrics()
        self.name_label.setText(metrics.elidedText(
            name, Qt.TextElideMode.ElideRight, self._thumb_size.width()))

    def set_progress(self, progress: Progress, meta: str, percent: int | None = None):
        self.meta_label.setText(meta)
        self.progress.setValue(progress.percent if percent is None else percent)
        # 全部完成后不再显示角标，卡片墙里只有需要关注的卡片带标记
        show_badge = progress.status != "done" and not (
            progress.status == "pending" and progress.done == 0)
        self.badge.setText(status_text(progress.status) if show_badge else "")
        self.badge.setVisible(show_badge)
        self.badge.setProperty("status", progress.status)
        self.progress.setProperty("status", progress.status)
        repolish(self.badge)
        repolish(self.progress)

    def set_thumbnail(self, pixmap: QPixmap):
        self.thumb.setPixmap(pixmap.scaled(
            self._thumb_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def set_selected(self, selected: bool):
        self.setProperty("selected", selected)
        repolish(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class AddCard(BaseWidget):
    """每个分类末尾的"新建"占位卡：空分类也有入口，直接新建这一类的资源包。"""
    clicked = Signal()
    # qss 的 dashed 边框不抗锯齿、圆角处虚线会断开，所以虚线框自己画
    BORDER_COLOR = QColor(255, 255, 255, 36)
    HOVER_BORDER_COLOR = QColor(123, 157, 188, 150)
    HOVER_FILL_COLOR = QColor(123, 157, 188, 12)
    RADIUS = 8

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("library_add_card")
        # 鼠标进出时触发重绘，paintEvent 才能切换悬停颜色
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(PACK_THUMB.width() + 16, PACK_THUMB.height() + 16)
        layout = QVBoxLayout(self)
        label = QLabel("+\n" + text)
        label.setObjectName("add_card_label")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def paintEvent(self, event):
        hovered = self.underMouse()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 线宽 1 时描边落在像素中心，内缩半个像素才不会发虚
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        if hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self.HOVER_FILL_COLOR)
            painter.drawRoundedRect(rect, self.RADIUS, self.RADIUS)
        pen = QPen(self.HOVER_BORDER_COLOR if hovered else self.BORDER_COLOR, 1)
        pen.setDashPattern([4, 3])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, self.RADIUS, self.RADIUS)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
