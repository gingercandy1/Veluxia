"""循环拼接的屏幕卡片条：卡片的顺序就是条带上从左到右的拼接顺序。

只负责展示和收集操作（添加 / 删除 / 拖动换位），不读取资源包、不发请求；
改完的顺序通过信号交给页面。同一屏可以有多张卡片。
"""
from enum import Enum

from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

CARD_THUMB = QSize(96, 54)
CARD_SIZE = QSize(112, 96)
CARD_GAP = 8
# 相邻两张卡片之间的接缝连接器占的宽度：它取代了卡片间的空隙，所以是 CARD_GAP 的两倍
LINK_WIDTH = CARD_GAP * 2
LINK_BAR_WIDTH = 6
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
    clicked = Signal(object)

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
        # 按下后没拖起来（拖动时 _press_pos 已清空）就算点击
        if event.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            self.clicked.emit(self)
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def set_active(self, active: bool):
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)


class SeamLinkState(Enum):
    """一条接缝（相邻两屏之间）当前的状态。"""
    NONE = "none"              # 还没有候选
    CANDIDATE = "candidate"    # 有候选，但还没采用
    ADOPTED = "adopted"
    STALE = "stale"            # 两侧的屏变了，要重做
    RUNNING = "running"        # 正在生成，带进度


class SeamLink(QWidget):
    """卡片之间的接缝连接器：一根竖条，颜色表示状态，生成中按进度填充。

    接缝属于相邻两屏之间，画在两张卡片之间，同一对相邻屏在环里出现几次就有几根。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(LINK_WIDTH)
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        # 竖条与缩略图上下对齐：顶部留出卡片的内边距
        layout.setContentsMargins(0, 6, 0, 0)
        self.bar = QProgressBar()
        self.bar.setObjectName("ring_seam_link")
        self.bar.setOrientation(Qt.Orientation.Vertical)
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedSize(LINK_BAR_WIDTH, CARD_THUMB.height())
        layout.addWidget(self.bar, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)
        self.clear()

    def clear(self):
        """没有接缝可画（环里不足两张卡片）：只占位，不画竖条。"""
        self.bar.hide()
        self.setToolTip("")

    def set_state(self, state: SeamLinkState, fraction: float = 0.0):
        """fraction 只在生成中有意义（0~1）。"""
        self.bar.show()
        self.bar.setProperty("state", state.value)
        # 非生成中的状态用满条表示“有内容”，NONE 只剩底槽
        if state is SeamLinkState.RUNNING:
            value = round(fraction * 100)
        else:
            value = 0 if state is SeamLinkState.NONE else 100
        self.bar.setValue(value)
        self.bar.style().unpolish(self.bar)
        self.bar.style().polish(self.bar)
        self.setToolTip(self._tooltip(state, value))

    def _tooltip(self, state: SeamLinkState, percent: int) -> str:
        if state is SeamLinkState.RUNNING:
            return self.tr("Generating seam… {0}%").format(percent)
        return {
            SeamLinkState.NONE: self.tr("Seam: not generated yet"),
            SeamLinkState.CANDIDATE: self.tr("Seam: has candidates, none adopted"),
            SeamLinkState.ADOPTED: self.tr("Seam: adopted"),
            SeamLinkState.STALE: self.tr("Seam: source image changed, redo it"),
        }[state]


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
    card_selected = Signal(int)             # 点中的卡片序号

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
        # 第 i 根连接 第 i 张 → 第 i+1 张（最后一根回到第一张），只在有两张以上时存在
        self._links: list[SeamLink] = []
        self._thumbs: dict[str, QPixmap] = {}
        self._active = -1
        self._host = _CardHost(self)
        self._layout = QHBoxLayout(self._host)
        self._layout.setContentsMargins(0, 4, 0, 4)
        # 卡片间的空隙由接缝连接器占掉，布局本身不留缝
        self._layout.setSpacing(0)
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
        for widget in (*self._cards, *self._links):
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._cards, self._links = [], []
        for position, (item_id, tooltip) in enumerate(entries):
            card = RingCard(item_id, tooltip)
            card.set_position(position + 1)
            if (thumb := self._thumbs.get(item_id)) is not None:
                card.set_thumb(thumb)
            card.remove_clicked.connect(self._on_remove)
            card.drag_started.connect(self._start_drag)
            card.clicked.connect(self._on_card_clicked)
            link = SeamLink()
            # 每张卡片后面跟着它右侧的接缝连接器，所以卡片在布局里的下标是 2 * 序号
            self._layout.insertWidget(2 * position, card)
            self._layout.insertWidget(2 * position + 1, link)
            self._cards.append(card)
            self._links.append(link)
        # 增删 / 换位后序号对应的卡片变了，旧的激活态不再有意义
        self.set_active(-1)
        self._apply_state()

    def set_active(self, index: int):
        """高亮第 index 张卡片（-1 取消）。"""
        self._active = index if 0 <= index < len(self._cards) else -1
        for position, card in enumerate(self._cards):
            card.set_active(position == self._active)

    def set_seam_links(self, states: list[tuple[SeamLinkState, float]]):
        """states[i] = (状态, 生成进度 0~1) 对应 第 i 张 → 第 i+1 张之间的接缝；
        比卡片少的部分（环里不足两张时为空）不画。"""
        for index, link in enumerate(self._links):
            if index < len(states):
                link.set_state(*states[index])
            else:
                link.clear()

    def active(self) -> int:
        return self._active

    def _on_card_clicked(self, card: RingCard):
        if card in self._cards:
            index = self._cards.index(card)
            self.set_active(index)
            self.card_selected.emit(index)

    def thumb(self, item_id: str) -> QPixmap | None:
        return self._thumbs.get(item_id)

    def clear_thumbs(self):
        self._thumbs.clear()
        for card in self._cards:
            card.thumb.clear()

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
            marker_x = self._cards[slot].x() - LINK_WIDTH // 2
        else:
            marker_x = self._cards[-1].geometry().right() + LINK_WIDTH // 2
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
