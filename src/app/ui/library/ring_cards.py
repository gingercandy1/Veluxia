"""循环拼接的屏幕卡片条：卡片的顺序就是条带上从左到右的拼接顺序。

只负责展示和收集操作（添加 / 删除 / 拖动换位），不读取资源包、不发请求；
改完的顺序通过信号交给页面。同一屏可以有多张卡片。
"""
from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QDrag, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

CARD_THUMB = QSize(96, 54)
CARD_SIZE = QSize(112, 96)
CARD_GAP = 8
# 条带 + 横向滚动条（滚动条出现时不能把卡片挤掉）
BAR_HEIGHT = CARD_SIZE.height() + 24
CARD_MIME = "application/x-veluxia-ring-card"
# 拖到离可视区边缘这么近就自动滚动
EDGE_SCROLL_ZONE = 28
EDGE_SCROLL_STEP = 24
PICKER_THUMB = QSize(144, 81)


class RingCard(QFrame):
    """一张卡片 = 环上的一个位置。"""
    remove_clicked = Signal(object)
    drag_started = Signal(object)

    def __init__(self, item_id: str, tooltip: str, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_card")
        self.setFixedSize(CARD_SIZE)
        self.item_id = item_id
        self.setToolTip(tooltip)
        self._press_pos: QPoint | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)
        self.thumb = QLabel()
        self.thumb.setObjectName("ring_card_thumb")
        self.thumb.setFixedSize(CARD_THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.thumb)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(0)
        self.caption = QLabel()
        self.caption.setObjectName("detail_muted")
        footer.addWidget(self.caption, 1)
        self.remove_btn = QToolButton()
        self.remove_btn.setObjectName("ring_card_remove")
        self.remove_btn.setText("×")
        self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self))
        footer.addWidget(self.remove_btn)
        layout.addLayout(footer)

    def set_position(self, position: int):
        # 序号 = 在环上的第几屏；同一屏出现多次时靠它区分
        metrics = self.caption.fontMetrics()
        text = f"{position} · {self.item_id}"
        self.caption.setText(metrics.elidedText(
            text, Qt.TextElideMode.ElideRight, CARD_THUMB.width() - 14))

    def set_thumb(self, pixmap: QPixmap):
        self.thumb.setPixmap(pixmap.scaled(
            CARD_THUMB, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton and \
                (event.position().toPoint() - self._press_pos).manhattanLength() \
                >= QApplication.startDragDistance():
            self._press_pos = None
            self.drag_started.emit(self)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        super().mouseReleaseEvent(event)


class _CardHost(QWidget):
    """卡片所在的画布：把拖放事件交给条带处理（坐标是画布内的）。"""

    def __init__(self, bar: "RingCardBar"):
        super().__init__()
        self._bar = bar
        self.setObjectName("ring_cards_host")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(CARD_MIME) and self._bar.editable:
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(CARD_MIME) and self._bar.editable:
            self._bar.show_drop_marker(event.position().toPoint().x())
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._bar.hide_drop_marker()

    def dropEvent(self, event):
        self._bar.hide_drop_marker()
        if not (event.mimeData().hasFormat(CARD_MIME) and self._bar.editable):
            return
        source = int(bytes(event.mimeData().data(CARD_MIME)).decode())
        self._bar.move_card(source, self._bar.slot_at(event.position().toPoint().x()))
        event.acceptProposedAction()


class RingCardBar(QScrollArea):
    """横向滚动的卡片条，末尾是“添加”卡。"""
    add_requested = Signal()
    remove_requested = Signal(int)          # 卡片序号
    reordered = Signal(list)                # 换位后从左到右的条目 id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_cards")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedHeight(BAR_HEIGHT)
        self.editable = True
        self._removable = True
        self._key: list = []
        self._cards: list[RingCard] = []
        self._thumbs: dict[str, QPixmap] = {}
        self._host = _CardHost(self)
        self._layout = QHBoxLayout(self._host)
        self._layout.setContentsMargins(0, 4, 0, 4)
        self._layout.setSpacing(CARD_GAP)
        self.add_btn = QPushButton("+")
        self.add_btn.setObjectName("ring_card_add")
        self.add_btn.setFixedSize(CARD_SIZE)
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setToolTip(self.tr("Add a screen. The same screen can be added again."))
        self.add_btn.clicked.connect(self.add_requested)
        self._layout.addWidget(self.add_btn)
        self._layout.addStretch(1)
        self._marker = QFrame(self._host)
        self._marker.setObjectName("ring_card_drop_marker")
        self._marker.setFixedWidth(3)
        self._marker.hide()
        self.setWidget(self._host)

    # ---- 对外 ----
    def ids(self) -> list[str]:
        return [card.item_id for card in self._cards]

    def set_cards(self, entries: list[tuple[str, str]]):
        """entries：(条目 id, 提示文字)，从左到右。没变化时不重建，轮询刷新不会打断拖动。"""
        if entries == self._key:
            return
        self._key = list(entries)
        for card in self._cards:
            self._layout.removeWidget(card)
            card.deleteLater()
        self._cards = []
        for position, (item_id, tooltip) in enumerate(entries):
            card = RingCard(item_id, tooltip)
            card.set_position(position + 1)
            if (thumb := self._thumbs.get(item_id)) is not None:
                card.set_thumb(thumb)
            card.remove_clicked.connect(self._on_remove)
            card.drag_started.connect(self._start_drag)
            self._layout.insertWidget(position, card)
            self._cards.append(card)
        self._apply_state()

    def set_thumb(self, item_id: str, pixmap: QPixmap):
        self._thumbs[item_id] = pixmap
        for card in self._cards:
            if card.item_id == item_id:
                card.set_thumb(pixmap)

    def set_editable(self, editable: bool, can_add: bool = True, removable: bool = True):
        """editable：能否拖动 / 删除；can_add：没有可加的屏时添加卡置灰；
        removable：环里只剩两张时不能再删（后端要求至少两屏）。"""
        self.editable = editable
        self._removable = removable
        self.add_btn.setEnabled(editable and can_add)
        self._apply_state()

    def _apply_state(self):
        for card in self._cards:
            card.remove_btn.setEnabled(self.editable and self._removable)
            card.setCursor(Qt.CursorShape.OpenHandCursor if self.editable
                           else Qt.CursorShape.ArrowCursor)

    # ---- 换位 ----
    def slot_at(self, x: int) -> int:
        """画布内横坐标 x 对应的插入位置：落在第 k 张卡片之前（k == 张数表示末尾）。"""
        return sum(1 for card in self._cards if card.geometry().center().x() < x)

    def move_card(self, source: int, slot: int):
        if not 0 <= source < len(self._cards):
            return
        target = slot - 1 if slot > source else slot
        if target == source:
            return
        ids = self.ids()
        ids.insert(target, ids.pop(source))
        self.reordered.emit(ids)

    def show_drop_marker(self, x: int):
        slot = self.slot_at(x)
        if not self._cards:
            return
        if slot < len(self._cards):
            marker_x = self._cards[slot].x() - CARD_GAP // 2
        else:
            marker_x = self._cards[-1].geometry().right() + CARD_GAP // 2
        self._marker.setGeometry(marker_x - 1, 0, 3, self._host.height())
        self._marker.raise_()
        self._marker.show()
        # 卡片很多时目标可能在可视区外：拖到边缘自动滚动
        view_x = x - self.horizontalScrollBar().value()
        scroll = self.horizontalScrollBar()
        if view_x < EDGE_SCROLL_ZONE:
            scroll.setValue(scroll.value() - EDGE_SCROLL_STEP)
        elif view_x > self.viewport().width() - EDGE_SCROLL_ZONE:
            scroll.setValue(scroll.value() + EDGE_SCROLL_STEP)

    def hide_drop_marker(self):
        self._marker.hide()

    def _start_drag(self, card: RingCard):
        if not self.editable or card not in self._cards:
            return
        mime = QMimeData()
        mime.setData(CARD_MIME, str(self._cards.index(card)).encode())
        drag = QDrag(card)
        drag.setMimeData(mime)
        drag.setPixmap(card.grab())
        drag.setHotSpot(QPoint(card.width() // 2, card.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)
        self.hide_drop_marker()

    def _on_remove(self, card: RingCard):
        if card in self._cards:
            self.remove_requested.emit(self._cards.index(card))


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
            row = QListWidgetItem(QIcon(pixmap) if pixmap is not None else QIcon(),
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
