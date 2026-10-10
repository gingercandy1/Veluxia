from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import ProgressStrip, Shimmer
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
    export_requested = Signal()
    delete_requested = Signal()
    layers_toggled = Signal(bool)

    def __init__(self, thumb_size: QSize, placeholder: str, category: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("library_card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._thumb_size = thumb_size
        self.thumb_token = ""
        # 带分层视图的卡片：原图和分层各一张缩略图，按钮切换显示哪张
        self.layer_token = ""
        self._pixmaps: dict[bool, QPixmap] = {}
        self._layered = False
        self.layer_btn: QPushButton | None = None

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
        self._shimmer = Shimmer(self.thumb)

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

        self.progress = ProgressStrip(height=3)
        layout.addWidget(self.progress)

        self.setFixedWidth(thumb_size.width() + 16)
        self._actions: QWidget | None = None

    def add_actions(self):
        """缩略图右上角的导出 / 删除按钮：只在鼠标停在卡片上时显示，平时不挡图。
        做成缩略图的子控件，点按钮不会冒泡成卡片的选中和双击放大。"""
        self._actions = QWidget(self.thumb)
        self._actions.setObjectName("card_actions")
        row = QHBoxLayout(self._actions)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.export_btn = self._action_button(
            ":/svg/export.svg", "card_export_btn", self.export_requested)
        self.delete_btn = self._action_button(
            ":/svg/trash.svg", "card_delete_btn", self.delete_requested)
        row.addWidget(self.export_btn)
        row.addWidget(self.delete_btn)
        self._actions.adjustSize()
        self._actions.move(self._thumb_size.width() - self._actions.width() - 6, 6)
        self._actions.hide()

    def add_layer_toggle(self):
        """缩略图左上角的"分层"开关：常驻显示（有分层的标记），点一下在原图和分层之间切换。
        同样是缩略图的子控件，点它不会冒泡成卡片的选中。"""
        self.layer_btn = QPushButton(self.tr("Layers"), self.thumb)
        self.layer_btn.setObjectName("card_layer_btn")
        self.layer_btn.setCheckable(True)
        self.layer_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.layer_btn.setFixedHeight(22)
        self.layer_btn.move(6, 6)
        self.layer_btn.adjustSize()
        self.layer_btn.toggled.connect(self._on_layer_toggled)
        self.layer_btn.hide()

    def set_layers_available(self, available: bool):
        """分层还没做完（或这条没分层）时按钮不出现，也退回显示原图。"""
        if self.layer_btn is None:
            return
        self.layer_btn.setVisible(available)
        if not available and self.layer_btn.isChecked():
            self.layer_btn.setChecked(False)

    def is_layered(self) -> bool:
        return self._layered

    def _on_layer_toggled(self, checked: bool):
        self._layered = checked
        self._show_pixmap()
        self.layers_toggled.emit(checked)

    def _action_button(self, icon: str, name: str, signal) -> QPushButton:
        button = QPushButton(QIcon(icon), "")
        button.setObjectName(name)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setIconSize(QSize(14, 14))
        button.setFixedSize(26, 26)
        button.clicked.connect(signal)
        return button

    def enterEvent(self, event):
        if self._actions is not None:
            self._actions.show()
            self._actions.raise_()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._actions is not None:
            self._actions.hide()
        super().leaveEvent(event)

    def set_name(self, name: str):
        self.name_label.setToolTip(name)
        metrics = self.name_label.fontMetrics()
        self.name_label.setText(metrics.elidedText(
            name, Qt.TextElideMode.ElideRight, self._thumb_size.width()))

    def set_progress(self, progress: Progress, meta: str, percent: int | None = None,
                     active: bool = False):
        """active：这张卡片对应的内容正在执行，进度条上走流光。"""
        self.meta_label.setText(meta)
        self.progress.set_progress(progress.percent if percent is None else percent,
                                   progress.status, active)
        # 全部完成后不再显示角标，卡片墙里只有需要关注的卡片带标记
        show_badge = progress.status != "done" and not (
            progress.status == "pending" and progress.done == 0)
        self.badge.setText(status_text(progress.status) if show_badge else "")
        self.badge.setVisible(show_badge)
        self.badge.setProperty("status", progress.status)
        repolish(self.badge)

    def set_thumbnail(self, pixmap: QPixmap, layered: bool = False):
        self.set_loading(False)
        self._pixmaps[layered] = pixmap.scaled(
            self._thumb_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._show_pixmap()

    def _show_pixmap(self):
        # 分层缩略图还没到时先留着原图，不闪成空白占位
        pixmap = self._pixmaps.get(self._layered, self._pixmaps.get(False))
        if pixmap is not None:
            self.thumb.setPixmap(pixmap)

    def set_loading(self, loading: bool):
        """缩略图在路上：占位上扫一道微光。"""
        self._shimmer.set_running(loading)

    def is_loading(self) -> bool:
        return self._shimmer.is_running()

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
