"""图层窗口：循环画布页右侧的节点树 + 顶部四个工具按钮 + 底部文字改图输入框。

只负责展示和收集操作，不碰后端：节点（屏 / 段 / 层）由外面按 plan.md 的目标键传进来，
点工具、切版本、改图、合并都只发信号，由页面去调接口。

沿用 layer_panel 的逐行控件写法（而不是 QTreeWidget）：子行缩进后眼睛列也能对齐，
重建时可以原样保留选中、折叠和滚动位置。"""
from dataclasses import dataclass, field

from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QDrag, QImage
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.layer_panel import ROW_HEIGHT, LayerRow

# 工具按钮从左到右的顺序；前四个是 plan.md 里"外扩 → 接缝 → 分段 → 分层"的流程，
# region 是框选局部改图（只动框内），不占模型，是画布上的选择模式而不是长任务
TOOLS = ("extend", "seams", "segment", "layer", "region")
INDENT = 18
FOLD_BUTTON = 16
PROMPT_LINES = 3
WINDOW_MIN_WIDTH = 360
BADGE_MAX_WIDTH = 110
VERSION_BUTTON = 20
LAYER_PREFIX = "layer:"
NODE_MIME = "application/x-veluxia-layer-node"
DROP_MARKER_HEIGHT = 3


@dataclass
class LayerNode:
    key: str
    title: str
    children: list["LayerNode"] = field(default_factory=list)
    badge: str = ""
    version_count: int = 0
    current_version: int = -1      # -1 = 原图
    visible: bool = True


class VersionControl(QFrame):
    """"◀ n/m ▶"：current 为 -1 时显示 orig（原图）。"""
    requested = Signal(int)

    def __init__(self, count: int, current: int, parent=None):
        super().__init__(parent)
        self.setObjectName("layer_version")
        self._current = current
        self.prev_btn = QToolButton()
        self.prev_btn.setObjectName("layer_version_btn")
        self.prev_btn.setText("◀")
        self.next_btn = QToolButton()
        self.next_btn.setObjectName("layer_version_btn")
        self.next_btn.setText("▶")
        for button in (self.prev_btn, self.next_btn):
            button.setFixedSize(VERSION_BUTTON, VERSION_BUTTON)
        self.label = QLabel("orig" if current < 0 else f"{current + 1}/{count}")
        self.label.setObjectName("layer_version_label")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 数字位数变化时标签不跳动
        self.label.setMinimumWidth(36)
        # 边界上不能再翻：上一个到原图（-1）为止，下一个到最新为止
        self.prev_btn.setEnabled(current >= 0)
        self.next_btn.setEnabled(current < count - 1)
        self.prev_btn.clicked.connect(lambda: self.requested.emit(self._current - 1))
        self.next_btn.clicked.connect(lambda: self.requested.emit(self._current + 1))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.label)
        layout.addWidget(self.next_btn)
        self.setToolTip(self.tr("Switch between the original and edited versions"))


class NodeRow(LayerRow):
    """树里的一行：在图层行的基础上加折叠箭头、badge 和版本控件。"""
    fold_toggled = Signal(str)
    version_requested = Signal(str, int)
    drag_started = Signal(str)

    def __init__(self, node: LayerNode, depth: int, folded: bool, parent=None):
        super().__init__(node.key, node.title, selectable=True, parent=parent)
        self.node = node
        self._press_pos = None
        layout = self.layout()
        layout.setContentsMargins(6 + depth * INDENT, 3, 8, 3)
        self.fold_btn = QToolButton()
        self.fold_btn.setObjectName("layer_fold")
        self.fold_btn.setFixedSize(FOLD_BUTTON, FOLD_BUTTON)
        self.fold_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.insertWidget(0, self.fold_btn)
        if node.children:
            self.set_folded(folded)
            self.fold_btn.clicked.connect(lambda: self.fold_toggled.emit(self.layer_id))
        else:
            # 叶子也占位，同级的眼睛和缩略图才对得齐
            self.fold_btn.setEnabled(False)
        # 名字列可以被压窄：缩进较深的子行加上 badge、版本控件时，整行不能撑出窗口右边
        self.label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.label.setToolTip(node.title)
        # badge 限宽：俄语等较长的状态文字省略显示，完整文字放提示里。
        # 先 polish 再量宽度，量的才是 QSS 里 badge 自己的字号
        self.badge = QLabel()
        self.badge.setObjectName("layer_row_badge")
        self.badge.ensurePolished()
        self.badge.setText(self.badge.fontMetrics().elidedText(
            node.badge, Qt.TextElideMode.ElideRight, BADGE_MAX_WIDTH - 10))
        self.badge.setToolTip(node.badge)
        self.badge.setMaximumWidth(BADGE_MAX_WIDTH)
        self.badge.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.badge.setVisible(bool(node.badge))
        layout.addWidget(self.badge, 0, Qt.AlignmentFlag.AlignVCenter)
        self.version = None
        if node.version_count > 0:
            self.version = VersionControl(node.version_count, node.current_version)
            self.version.requested.connect(
                lambda index: self.version_requested.emit(self.layer_id, index))
            layout.addWidget(self.version)

    def set_folded(self, folded: bool):
        self.fold_btn.setText("▸" if folded else "▾")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # 按住左键移动超过系统拖动距离才算拖动；单击仍然只是选中
        if (self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton
                and (event.position().toPoint() - self._press_pos).manhattanLength()
                >= QApplication.startDragDistance()):
            self._press_pos = None
            self.drag_started.emit(self.layer_id)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_pos = None
        super().mouseReleaseEvent(event)


class _TreeBody(QWidget):
    """树的内容区：接收顶层节点的拖放，落点之间画一条插入线。"""

    def __init__(self, window: "LayerWindow"):
        super().__init__()
        self._window = window
        self.setAcceptDrops(True)
        self.marker = QFrame(self)
        self.marker.setObjectName("layer_drop_marker")
        self.marker.hide()

    def _accepts(self, event) -> bool:
        return self._window.reorderable() and event.mimeData().hasFormat(NODE_MIME)

    def dragEnterEvent(self, event):
        if self._accepts(event):
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if not self._accepts(event):
            return
        event.acceptProposedAction()
        y = self._window.drop_line_y(self._window.slot_at(event.position().toPoint().y()))
        self.marker.setGeometry(0, y - DROP_MARKER_HEIGHT // 2, self.width(), DROP_MARKER_HEIGHT)
        self.marker.raise_()
        self.marker.show()

    def dragLeaveEvent(self, event):
        self.marker.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):
        self.marker.hide()
        if not self._accepts(event):
            return
        event.acceptProposedAction()
        key = bytes(event.mimeData().data(NODE_MIME)).decode()
        self._window.move_node(key, self._window.slot_at(event.position().toPoint().y()))


class _PromptEdit(QTextEdit):
    """Ctrl+Enter 发送；普通 Enter 仍是换行。"""
    submitted = Signal()

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class LayerWindow(BaseWidget):
    tool_clicked = Signal(str)             # 工具名；开始还是停止由外部按当前状态决定
    selection_changed = Signal(list)       # 选中的 key，按显示顺序
    visibility_changed = Signal(list)      # 所有可见节点的 key
    version_requested = Signal(str, int)   # key, 目标版本序号（-1 = 原图）
    node_opened = Signal(str)              # 双击行
    merge_requested = Signal(list)         # 层 key 列表
    delete_requested = Signal(str)         # 合并层 key
    edit_requested = Signal(str, str)      # key, 文字指令
    region_edit_requested = Signal(str, str, list)  # key, 文字指令, 归一化框选 [x,y,w,h]
    add_requested = Signal()               # 往列表末尾添加（由页面弹选择框）
    remove_requested = Signal(list)        # 要移除的顶层节点 key
    reorder_requested = Signal(list)       # 拖动换位后，顶层节点 key 的新顺序

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("layer_window")
        self.setMinimumWidth(WINDOW_MIN_WIDTH)
        # 要能拿到焦点，Ctrl+A 才只在本控件内生效；输入框自己处理自己的 Ctrl+A
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._nodes: list[LayerNode] = []
        self._images: dict[str, QImage] = {}
        self._hidden: set[str] = set()
        self._known: set[str] = set()
        self._selected: list[str] = []
        self._anchor = ""
        self._collapsed: set[str] = set()
        self._idle = True
        # 列表模式（未分段时的屏列表）：可以添加、移除、拖动换位；分段后换成合并 / 删除合并层
        self._list_mode = False
        self._list_editable = False
        self._can_remove = False
        self._body: _TreeBody | None = None
        self._rows: dict[str, NodeRow] = {}
        self._bodies: dict[str, QWidget] = {}   # 有子节点的行 -> 装子行的容器
        self._order: list[str] = []            # 当前可见（未被折叠藏起）的行，从上到下
        self._tool_labels = {
            "extend": (self.tr("Extend"), self.tr("Stop extend")),
            "seams": (self.tr("Seams"), self.tr("Stop seams")),
            "segment": (self.tr("Segment"), self.tr("Stop segment")),
            "layer": (self.tr("Layer"), self.tr("Stop layer")),
            "region": (self.tr("Region"), self.tr("Stop region")),
        }
        # 画布上的框选（归一化 [x,y,w,h]，相对当前选中目标的图）：有它时发送走框选改图
        self._region_bbox: list | None = None

        title = QLabel(self.tr("Layers"))
        title.setObjectName("layer_window_title")
        self.tool_buttons: dict[str, QPushButton] = {}
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.addStretch(1)
        for tool in TOOLS:
            button = QPushButton(self._tool_labels[tool][0])
            button.setObjectName("layer_tool_btn")
            button.clicked.connect(lambda _=False, t=tool: self.tool_clicked.emit(t))
            self.tool_buttons[tool] = button
            toolbar.addWidget(button)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("ring_layer_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.setMinimumHeight(ROW_HEIGHT * 5)

        self.merge_btn = QPushButton(self.tr("Merge"))
        self.merge_btn.setToolTip(self.tr("Merge the selected layers into a new layer"))
        self.delete_btn = QPushButton(self.tr("Delete merged layer"))
        self.merge_btn.clicked.connect(self._emit_merge)
        self.delete_btn.clicked.connect(self._emit_delete)
        self.add_btn = QPushButton(self.tr("Add screens"))
        self.add_btn.setToolTip(self.tr("Append screens to the end of the ring; "
                                        "a screen can be used more than once"))
        self.remove_btn = QPushButton(self.tr("Remove"))
        self.remove_btn.setToolTip(self.tr("Remove the selected screens from the ring (Delete)"))
        self.add_btn.clicked.connect(self._emit_add)
        self.remove_btn.clicked.connect(self._emit_remove)
        merge_row = QHBoxLayout()
        merge_row.addWidget(self.add_btn)
        merge_row.addWidget(self.remove_btn)
        merge_row.addWidget(self.merge_btn)
        merge_row.addWidget(self.delete_btn)
        merge_row.addStretch(1)

        self.prompt_edit = _PromptEdit()
        self.prompt_edit.setObjectName("layer_prompt")
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setPlaceholderText(
            self.tr("Describe the change for the selected layer, e.g. make the ground steeper"))
        # 限高约三行：再长的指令靠滚动，不挤占树的空间
        metrics = self.prompt_edit.fontMetrics()
        self.prompt_edit.setFixedHeight(metrics.lineSpacing() * PROMPT_LINES + 16)
        self.prompt_edit.submitted.connect(self._send)
        self.prompt_edit.textChanged.connect(self._update_buttons)
        self.send_btn = QPushButton(self.tr("Send"))
        self.send_btn.setToolTip(self.tr("Edit the selected item with this instruction (Ctrl+Enter)"))
        self.send_btn.clicked.connect(self._send)
        self.undo_btn = QPushButton(self.tr("Undo"))
        self.undo_btn.setToolTip(self.tr(
            "Undo the last change and go back to the previous version"))
        self.undo_btn.clicked.connect(self._undo)
        edit_row = QHBoxLayout()
        edit_row.setSpacing(6)
        edit_row.addStretch(1)
        edit_row.addWidget(self.undo_btn)
        edit_row.addWidget(self.send_btn)
        edit_title = QLabel(self.tr("Edit with Qwen-Image-2.1"))
        edit_title.setObjectName("layer_window_caption")
        separator = QFrame()
        separator.setObjectName("layer_window_separator")
        separator.setFixedHeight(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        # 标题单独一行：和四个工具按钮挤一行会把整个窗口撑宽，压缩左边画布
        layout.addWidget(title)
        layout.addLayout(toolbar)
        layout.addWidget(self.scroll, 1)
        layout.addLayout(merge_row)
        layout.addWidget(separator)
        layout.addWidget(edit_title)
        layout.addWidget(self.prompt_edit)
        layout.addLayout(edit_row)

        for tool in TOOLS:
            self.set_tool_state(tool, running=False, enabled=True)
        self.set_list_mode(False)

    # ---- 对外 ----
    def set_tool_state(self, tool: str, running: bool, enabled: bool, tooltip: str = ""):
        button = self.tool_buttons[tool]
        idle_text, running_text = self._tool_labels[tool]
        button.setText(running_text if running else idle_text)
        button.setEnabled(enabled)
        button.setToolTip(tooltip)
        if button.property("running") != running:
            button.setProperty("running", running)
            button.style().unpolish(button)
            button.style().polish(button)

    def set_list_mode(self, enabled: bool, editable: bool = False, can_add: bool = False,
                      can_remove: bool = False):
        """enabled：节点是可增删、可拖动换位的列表（未分段的屏）。
        editable：此刻允许改（后端空闲）；can_add / can_remove 由页面按规则算好
        （没有可加的屏、删了会不够两屏时为 False）。"""
        self._list_mode = enabled
        self._list_editable = enabled and editable
        self._can_remove = can_remove
        for button in (self.add_btn, self.remove_btn):
            button.setVisible(enabled)
        for button in (self.merge_btn, self.delete_btn):
            button.setVisible(not enabled)
        self.add_btn.setEnabled(self._list_editable and can_add)
        self._update_buttons()

    def reorderable(self) -> bool:
        return self._list_editable

    def node_keys(self) -> list[str]:
        """所有节点的 key（含被折叠藏起的），从上到下。"""
        return [key for key, _depth, _node in self._walk(self._nodes)]

    def set_idle(self, idle: bool):
        """后端在跑长任务时不能发改图、合并、删除，也不能切版本。"""
        self._idle = idle
        for row in self._rows.values():
            if row.version is not None:
                row.version.setEnabled(idle)
        self._update_buttons()

    def set_nodes(self, nodes: list[LayerNode]):
        self._nodes = list(nodes)
        keys = [key for key, _depth, _node in self._walk(self._nodes)]
        alive = set(keys)
        # 新出现的节点按 node.visible 初始化眼睛；已有的保持用户当前的开关
        for _key, _depth, node in self._walk(self._nodes):
            if node.key not in self._known and not node.visible:
                self._hidden.add(node.key)
        self._known = alive
        self._hidden &= alive
        self._collapsed &= alive
        before = list(self._selected)
        self._selected = [key for key in self._selected if key in alive]
        if self._anchor not in alive:
            self._anchor = self._selected[-1] if self._selected else ""
        self._rebuild()
        if self._selected != before:
            self.selection_changed.emit(self.selected_keys())

    def set_images(self, images: dict[str, QImage]):
        self._images = dict(images)
        for key, row in self._rows.items():
            row.set_image(self._images.get(key))

    def selected_keys(self) -> list[str]:
        """选中的 key，按显示顺序（被折叠藏起的选中项排在可见项之后）。"""
        shown = [key for key in self._order if key in self._selected]
        return shown + [key for key in self._selected if key not in shown]

    def select(self, keys: list[str]):
        known = self._known
        self._selected = [key for key in keys if key in known]
        self._anchor = self._selected[-1] if self._selected else ""
        self._apply_selection()

    def visible_keys(self) -> list[str]:
        return [key for key, _depth, _node in self._walk(self._nodes) if key not in self._hidden]

    def toggle_visible(self, key: str):
        if key not in self._known:
            return
        self._hidden ^= {key}
        row = self._rows.get(key)
        if row is not None:
            row.set_visible_eye(key not in self._hidden)
        self.visibility_changed.emit(self.visible_keys())

    def toggle_fold(self, key: str):
        row = self._rows.get(key)
        if row is None or not row.node.children:
            return
        self._collapsed ^= {key}
        # 折叠后藏起的行不再参与 Shift 连选，顺序要重算；子容器只改显隐，不整棵重建
        body = self._bodies.get(key)
        if body is not None:
            body.setVisible(key not in self._collapsed)
        row.set_folded(key in self._collapsed)
        self._order = [k for k, _d, _n in self._walk(self._nodes, skip_collapsed=True)]

    def is_collapsed(self, key: str) -> bool:
        return key in self._collapsed

    def row(self, key: str) -> NodeRow | None:
        return self._rows.get(key)

    # ---- 内部 ----
    def _walk(self, nodes: list[LayerNode], depth: int = 0, skip_collapsed: bool = False):
        for node in nodes:
            yield node.key, depth, node
            if node.children and not (skip_collapsed and node.key in self._collapsed):
                yield from self._walk(node.children, depth + 1, skip_collapsed)

    def _rebuild(self):
        scroll_value = self.scroll.verticalScrollBar().value()
        body = self._body = _TreeBody(self)
        column = QVBoxLayout(body)
        # 内边距放在这里而不是 QSS 的 padding：后者不计入内容宽度，行尾会被裁掉
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(2)
        self._rows, self._bodies = {}, {}
        self._add_nodes(column, self._nodes, 0)
        column.addStretch(1)
        self.scroll.setWidget(body)
        self.scroll.verticalScrollBar().setValue(scroll_value)
        self._order = [k for k, _d, _n in self._walk(self._nodes, skip_collapsed=True)]
        for key, row in self._rows.items():
            row.set_visible_eye(key not in self._hidden)
            row.set_image(self._images.get(key))
            if row.version is not None:
                row.version.setEnabled(self._idle)
        self._apply_selection(emit=False)

    def _add_nodes(self, column: QVBoxLayout, nodes: list[LayerNode], depth: int):
        for node in nodes:
            row = NodeRow(node, depth, node.key in self._collapsed)
            row.clicked.connect(self._on_row_clicked)
            row.eye_toggled.connect(self.toggle_visible)
            row.opened.connect(self.node_opened)
            row.fold_toggled.connect(self.toggle_fold)
            row.version_requested.connect(self.version_requested)
            if depth == 0:
                row.drag_started.connect(self._start_drag)
            self._rows[node.key] = row
            column.addWidget(row)
            if node.children:
                children = QWidget()
                child_column = QVBoxLayout(children)
                child_column.setContentsMargins(0, 0, 0, 0)
                child_column.setSpacing(2)
                self._add_nodes(child_column, node.children, depth + 1)
                children.setVisible(node.key not in self._collapsed)
                self._bodies[node.key] = children
                column.addWidget(children)

    def _on_row_clicked(self, key: str, modifiers):
        self.setFocus()
        before = list(self._selected)
        if modifiers & Qt.KeyboardModifier.ShiftModifier and self._anchor in self._order \
                and key in self._order:
            start, end = sorted((self._order.index(self._anchor), self._order.index(key)))
            self._selected = self._order[start:end + 1]
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            self._selected = ([k for k in self._selected if k != key]
                              if key in self._selected else [*self._selected, key])
            self._anchor = key
        else:
            self._selected, self._anchor = [key], key
        self._apply_selection(emit=self._selected != before)

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key.Key_A
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            before = list(self._selected)
            self._selected = [k for k, _d, _n in self._walk(self._nodes)]
            self._anchor = self._selected[-1] if self._selected else ""
            self._apply_selection(emit=self._selected != before)
            return
        if event.key() == Qt.Key.Key_Delete and self._list_mode:
            self._emit_remove()
            return
        super().keyPressEvent(event)

    # ---- 列表模式：添加 / 移除 / 拖动换位 ----
    def _top_rows(self) -> list[NodeRow]:
        return [self._rows[node.key] for node in self._nodes if node.key in self._rows]

    def slot_at(self, y: int) -> int:
        """树内容区纵坐标 y 对应的插入位置：落在第 k 个顶层节点之前（k == 个数表示末尾）。"""
        return sum(1 for row in self._top_rows() if row.geometry().center().y() < y)

    def drop_line_y(self, slot: int) -> int:
        rows = self._top_rows()
        if not rows:
            return 0
        if slot < len(rows):
            return rows[slot].y() - 1
        return rows[-1].geometry().bottom() + 1

    def move_node(self, key: str, slot: int):
        """把顶层节点 key 挪到 slot（插入位置）；放回原位不发信号。"""
        keys = [node.key for node in self._nodes]
        if not self._list_editable or key not in keys:
            return
        source = keys.index(key)
        target = slot - 1 if slot > source else slot
        if target == source:
            return
        keys.insert(target, keys.pop(source))
        self.reorder_requested.emit(keys)

    def _start_drag(self, key: str):
        row = self._rows.get(key)
        if not self._list_editable or row is None:
            return
        mime = QMimeData()
        mime.setData(NODE_MIME, key.encode())
        drag = QDrag(row)
        drag.setMimeData(mime)
        drag.setPixmap(row.grab())
        drag.exec(Qt.DropAction.MoveAction)
        if self._body is not None:
            self._body.marker.hide()

    def _selected_top(self) -> list[str]:
        top = {node.key for node in self._nodes}
        return [key for key in self.selected_keys() if key in top]

    def _emit_add(self):
        if self._list_editable:
            self.add_requested.emit()

    def _emit_remove(self):
        keys = self._selected_top()
        if self._list_editable and self._can_remove and keys:
            self.remove_requested.emit(keys)

    def _apply_selection(self, emit: bool = True):
        for key, row in self._rows.items():
            row.set_selected(key in self._selected)
        self._update_buttons()
        if emit:
            self.selection_changed.emit(self.selected_keys())

    def _selected_layers(self) -> list[str]:
        keys = self.selected_keys()
        return keys if keys and all(key.startswith(LAYER_PREFIX) for key in keys) else []

    @staticmethod
    def _segment_of(layer_key: str) -> str:
        return layer_key.rsplit(":", 1)[0]

    @staticmethod
    def _layer_id(layer_key: str) -> str:
        return layer_key.rsplit(":", 1)[1]

    def _can_merge(self) -> bool:
        layers = self._selected_layers()
        return (len(layers) >= 2
                and len({self._segment_of(key) for key in layers}) == 1)

    def _can_delete(self) -> bool:
        layers = self._selected_layers()
        return len(layers) == 1 and self._layer_id(layers[0]).startswith("M")

    def _emit_merge(self):
        if self._idle and self._can_merge():
            self.merge_requested.emit(self._selected_layers())

    def _emit_delete(self):
        if self._idle and self._can_delete():
            self.delete_requested.emit(self._selected_layers()[0])

    def set_region_bbox(self, bbox: list | None):
        """画布框选变化时由外面传入；合法才留，没有（None）就退回整图改。"""
        from src.app.ui.library.region_select import valid
        self._region_bbox = list(bbox) if valid(bbox) else None
        if self._region_bbox is not None:
            self.send_btn.setToolTip(self.tr(
                "Edit only the selected region with this instruction (Ctrl+Enter)"))
        else:
            self.send_btn.setToolTip(self.tr(
                "Edit the selected item with this instruction (Ctrl+Enter)"))

    def region_bbox(self) -> list | None:
        return list(self._region_bbox) if self._region_bbox is not None else None

    def _send(self):
        prompt = self.prompt_edit.toPlainText().strip()
        keys = self.selected_keys()
        if not (self._idle and len(keys) == 1 and prompt):
            return
        if self._region_bbox is not None:
            self.region_edit_requested.emit(keys[0], prompt, list(self._region_bbox))
        else:
            self.edit_requested.emit(keys[0], prompt)
        self.prompt_edit.clear()

    def _undoable(self) -> str:
        """可撤销的选中节点 key：只选中一个、且它不在原图上；否则为空串。
        和行内的版本控件一样，后端跑长任务时不能切版本。"""
        if not self._idle or len(self._selected) != 1:
            return ""
        row = self._rows.get(self._selected[0])
        if row is None or row.node.current_version < 0:
            return ""
        return self._selected[0]

    def _undo(self):
        key = self._undoable()
        if key:
            self.version_requested.emit(key, self._rows[key].node.current_version - 1)

    def _update_buttons(self):
        self.merge_btn.setEnabled(self._idle and self._can_merge())
        self.delete_btn.setEnabled(self._idle and self._can_delete())
        self.send_btn.setEnabled(self._idle and len(self._selected) == 1)
        self.undo_btn.setEnabled(bool(self._undoable()))
        self.remove_btn.setEnabled(
            self._list_editable and self._can_remove and bool(self._selected_top()))

    def sizeHint(self):
        return QSize(WINDOW_MIN_WIDTH + 40, 520)
