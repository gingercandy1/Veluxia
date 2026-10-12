"""往环里加屏的选择对话框。

环上的顺序、增删和拖动换位都在图层窗口里做（一屏一个节点，按环的顺序排）；
这里只负责"挑哪几屏加到末尾"。
"""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

PICKER_THUMB = QSize(144, 81)


def _fit_in_canvas(pixmap: QPixmap, size: QSize) -> QPixmap:
    """缩略图按各自比例缩小后大小不一：统一画到固定大小的底板中央，列表里才整齐。"""
    canvas = QPixmap(size)
    canvas.fill(QColor(255, 255, 255, 12))
    scaled = pixmap.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
    painter = QPainter(canvas)
    painter.drawPixmap((size.width() - scaled.width()) // 2,
                       (size.height() - scaled.height()) // 2, scaled)
    painter.end()
    return canvas


class RingItemPicker(QDialog):
    """从资源包里挑要加进环的屏；点哪几个就加哪几个，已经在环里的也可以再加。"""

    def __init__(self, entries: list[tuple[str, str, str, QPixmap | None]], parent=None):
        """entries：(条目 id, 标题, 提示文字, 缩略图)。"""
        super().__init__(parent)
        self.setWindowTitle(self.tr("Add screens to the ring"))
        layout = QVBoxLayout(self)
        hint = QLabel(self.tr(
            "Pick the screens to add. They are appended to the end of the ring; "
            "a screen already in the ring can be added again."))
        hint.setObjectName("detail_muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list.setIconSize(PICKER_THUMB)
        self.list.setSpacing(8)
        for item_id, title, tooltip, pixmap in entries:
            row = QListWidgetItem(QIcon(_fit_in_canvas(pixmap, PICKER_THUMB))
                                  if pixmap is not None else QIcon(),
                                  f"{item_id}\n{title}" if title and title != item_id
                                  else item_id)
            row.setData(Qt.ItemDataRole.UserRole, item_id)
            row.setToolTip(tooltip)
            row.setSizeHint(QSize(PICKER_THUMB.width() + 16, PICKER_THUMB.height() + 52))
            self.list.addItem(row)
        self.list.itemDoubleClicked.connect(self._pick_one)
        layout.addWidget(self.list, 1)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.list.itemSelectionChanged.connect(self._update_ok)
        self._update_ok()
        self.resize(560, 420)

    def _update_ok(self):
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            bool(self.list.selectedItems()))

    def _pick_one(self, item: QListWidgetItem):
        self.list.clearSelection()
        item.setSelected(True)
        self.accept()

    def selected_ids(self) -> list[str]:
        """按列表里的顺序（即资源包里的顺序）。"""
        return [self.list.item(row).data(Qt.ItemDataRole.UserRole)
                for row in range(self.list.count()) if self.list.item(row).isSelected()]
