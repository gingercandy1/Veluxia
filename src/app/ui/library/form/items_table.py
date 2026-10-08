"""条目表格：一行一个素材，只放要打字的列和必填的下拉。

- 末尾始终留一个空行：在里面打字就再长出一行，最后一行按回车也会新增，不需要"加行"按钮；
- 鼠标悬停时行尾出现 ⋯（单独改这一行的整包设置）和 ×（删除），删除有收起动画；
- Ctrl+V 粘贴多行文本时一行一个条目、Tab 分列，从 Excel / 文本批量录入；
- AI 起草时先显示几行闪烁的占位行，结果回来后逐行浮现。
"""
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QEvent,
    QRect,
    QRectF,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemDelegate,
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.anim import animate
from src.app.ui.library.form.defaults_bar import short_label
from src.shared.schemas import IMAGE_FIELD, TemplateFieldInfo

PROMPT_COLUMN = "prompt"
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
# 表格高度随行数增长，超过这么多行才出现滚动条
MAX_VISIBLE_ROWS = 10
# 条目单独改过的整包设置存在第一列单元格上：随行移动，删行、插行都不用另外维护
OVERRIDES_ROLE = Qt.ItemDataRole.UserRole + 1
ROW_HEIGHT = 38
_ERROR_BRUSH = QBrush(QColor(248, 113, 113, 48))


@dataclass(frozen=True)
class Column:
    id: str
    label: str
    required: bool = False
    options: tuple[tuple[str, str], ...] = ()
    default: str = ""
    # 分段描述列：对应的屏数字段大于 1 时才显示
    segments_field: str = ""
    # 图片列：单元格里放本地图片路径，双击选文件，不手打
    kind: str = ""

    @classmethod
    def from_field(cls, spec: TemplateFieldInfo) -> "Column":
        return cls(spec.id, spec.label or spec.id, spec.required,
                   tuple((o.value, o.label or o.value) for o in spec.options),
                   spec.default, spec.segments_field, spec.kind)

    @property
    def is_image(self) -> bool:
        return self.kind == IMAGE_FIELD


def _dot_icon() -> QIcon:
    """改过整包设置的行在行首画一个小圆点。"""
    pixmap = QPixmap(10, 10)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#7B9DBC"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(2, 2, 6, 6)
    painter.end()
    return QIcon(pixmap)


class _Shimmer(QWidget):
    """AI 起草时的占位行：几条圆角灰条，一道高光从左扫到右。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.rows = 3
        self._phase = 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_phase)

    def start(self):
        self.show()
        self.raise_()
        self._anim.start()

    def stop(self):
        self._anim.stop()
        self.hide()

    def _on_phase(self, value: float):
        self._phase = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        width = self.width()
        for index in range(self.rows):
            # 每行长短不一，看起来更像文字
            bar = QRectF(12, index * ROW_HEIGHT + 12, width * (0.62 - 0.12 * (index % 3)), 14)
            center = -0.3 + 1.6 * self._phase
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
            gradient.setColorAt(0.0, QColor(255, 255, 255, 14))
            gradient.setColorAt(max(0.0, min(1.0, center)), QColor(255, 255, 255, 38))
            gradient.setColorAt(1.0, QColor(255, 255, 255, 14))
            painter.setBrush(gradient)
            painter.drawRoundedRect(bar, 6, 6)


class _PlaceholderDelegate(QStyledItemDelegate):
    """末尾空行的第一格画一行淡色提示：没有网格线时空行本来什么都看不见。"""

    def __init__(self, table: "ItemsTable"):
        super().__init__(table)
        self._table = table

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        table = self._table
        if (index.column() != 0 or index.row() != table.rowCount() - 1
                or not table.is_empty_row(index.row())):
            return
        painter.save()
        painter.setPen(QColor(255, 255, 255, 70))
        rect = option.rect.adjusted(10, 0, -10, 0)
        painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         table.placeholder)
        painter.restore()


class OverridePopup(QFrame):
    """单独改这一行的整包设置：选成和整包一样的值就等于没改。"""

    def __init__(self, fields: list[TemplateFieldInfo], defaults: dict[str, str],
                 overrides: dict[str, str], on_done: Callable[[dict[str, str]], None],
                 parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("library_popup")
        self._defaults = defaults
        self._on_done = on_done
        self._combos: dict[str, QComboBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        for spec in fields:
            row = QHBoxLayout()
            label = QLabel(short_label(spec.label or spec.id))
            label.setObjectName("library_form_label")
            row.addWidget(label)
            row.addStretch()
            combo = QComboBox()
            for option in spec.options:
                combo.addItem(option.label or option.value, option.value)
            value = overrides.get(spec.id, defaults.get(spec.id, spec.default))
            combo.setCurrentIndex(max(combo.findData(value), 0))
            row.addWidget(combo)
            layout.addLayout(row)
            self._combos[spec.id] = combo
        reset = QPushButton(self.tr("Use pack settings"))
        reset.setObjectName("library_ghost_btn")
        reset.clicked.connect(self._reset)
        layout.addWidget(reset)

    def values(self) -> dict[str, str]:
        return {field_id: combo.currentData() for field_id, combo in self._combos.items()
                if combo.currentData() != self._defaults.get(field_id)}

    def _reset(self):
        for field_id, combo in self._combos.items():
            combo.setCurrentIndex(max(combo.findData(self._defaults.get(field_id)), 0))
        self.close()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._on_done(self.values())


class ItemsTable(QTableWidget):
    rows_changed = Signal()   # 有效条目数可能变了：底栏按钮据此更新数量

    def __init__(self, parent=None):
        super().__init__(0, 1, parent)
        self.setObjectName("library_table")
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        header.setMinimumSectionSize(90)
        header.setHighlightSections(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setMouseTracking(True)
        self.viewport().installEventFilter(self)
        self.itemChanged.connect(self._on_item_changed)
        self.cellDoubleClicked.connect(self._choose_image)
        self.placeholder = ""
        self.setItemDelegate(_PlaceholderDelegate(self))
        self._column_anims: dict[int, QVariantAnimation] = {}
        self._height_anim = QVariantAnimation(self)
        self._height_anim.setDuration(160)
        self._height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._height_anim.valueChanged.connect(lambda h: self.setFixedHeight(round(h)))
        # 行数变了就调整高度：内容少时不留一大片空白，多了再滚动
        self.model().rowsInserted.connect(self._fit_height)
        self.model().rowsRemoved.connect(self._fit_height)

        self._columns: list[Column] = []
        self._override_fields: list[TemplateFieldInfo] = []
        self._defaults: Callable[[], dict[str, str]] = dict
        self._errors: set[tuple[int, int]] = set()
        self._building = False
        self._dot = _dot_icon()
        self._hover_row = -1

        # 行尾操作：⋯ 单独设置、× 删除；悬停哪行就挪到哪行
        self._actions = QWidget(self.viewport())
        self._actions.setObjectName("library_row_actions")
        actions = QHBoxLayout(self._actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(2)
        self.more_btn = QToolButton()
        self.more_btn.setText("⋯")
        self.more_btn.setObjectName("library_row_action")
        self.more_btn.setToolTip(self.tr("Settings for this item only"))
        self.more_btn.clicked.connect(lambda: self.open_overrides(self._hover_row))
        self.remove_btn = QToolButton()
        self.remove_btn.setText("×")
        self.remove_btn.setObjectName("library_row_action")
        self.remove_btn.setToolTip(self.tr("Remove"))
        self.remove_btn.clicked.connect(lambda: self.remove_row(self._hover_row))
        for button in (self.more_btn, self.remove_btn):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            actions.addWidget(button)
        self._actions.hide()

        self._shimmer = _Shimmer(self.viewport())
        self._shimmer.hide()

    # ---- 列 ----
    def set_columns(self, columns: list[Column], override_fields: list[TemplateFieldInfo],
                    defaults: Callable[[], dict[str, str]]):
        """按模板重建表格；清空所有行，只留一个空行。"""
        for anim in self._column_anims.values():
            anim.stop()
        self._column_anims.clear()
        self._columns = columns
        self._override_fields = override_fields
        self._defaults = defaults
        self._errors.clear()
        self._building = True
        self.clear()
        self.setRowCount(0)
        self.setColumnCount(len(columns))
        for col, column in enumerate(columns):
            label = short_label(column.label)
            item = QTableWidgetItem(label + (" *" if column.required else ""))
            if label != column.label:
                item.setToolTip(column.label)
            self.setHorizontalHeaderItem(col, item)
        # 只有一列时隐藏表头，看起来就是一张干净的列表
        self.horizontalHeader().setVisible(len(columns) > 1)
        if columns:
            self.setColumnWidth(0, 360 if len(columns) > 1 else 200)
            self.placeholder = short_label(columns[0].label) + "…"
        # 有条件的列先隐藏，由 update_conditional_columns 按当前设置决定要不要展开
        for col, column in enumerate(columns):
            self.setColumnHidden(col, bool(column.segments_field))
        self._building = False
        self.ensure_blank_row()
        self.update_conditional_columns()
        self.rows_changed.emit()

    def columns(self) -> list[Column]:
        return list(self._columns)

    def update_conditional_columns(self):
        """分段描述等列：对应的屏数（整包设置或任一行单独设置）大于 1 时才显示，宽度渐变展开。"""
        defaults = self._defaults()
        for col, column in enumerate(self._columns):
            if not column.segments_field:
                continue
            values = {defaults.get(column.segments_field, "1")}
            values.update(self.row_overrides(r).get(column.segments_field, "1")
                          for r in range(self.rowCount()))
            wanted = any(value.isdigit() and int(value) > 1 for value in values)
            self._set_column_shown(col, wanted)

    def _set_column_shown(self, col: int, shown: bool):
        running = self._column_anims.pop(col, None)
        if running is not None:
            # 快速来回切换长度时，上一次的展开 / 收起还没播完：停掉，从当前宽度接着变
            running.stop()
        elif shown == (not self.isColumnHidden(col)):
            return
        target = 220

        def done():
            self._column_anims.pop(col, None)
            if not shown:
                self.setColumnHidden(col, True)

        self.setColumnHidden(col, False)
        start = self.columnWidth(col) if running is not None or not shown else 0
        anim = animate(self, start, target if shown else 0,
                       lambda w: self.setColumnWidth(col, round(w)), on_done=done)
        if anim is not None:
            self._column_anims[col] = anim

    # ---- 行 ----
    def add_row(self, animate_in: bool = False) -> int:
        row = self.rowCount()
        self._building = True
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem())
        for col, column in enumerate(self._columns):
            if column.options:
                combo = QComboBox()
                for value, label in column.options:
                    combo.addItem(label, value)
                combo.setCurrentIndex(max(combo.findData(column.default), 0))
                combo.currentIndexChanged.connect(lambda _=0: self.rows_changed.emit())
                self._set_cell_combo(row, col, combo)
            elif column.is_image:
                cell = QTableWidgetItem()
                # 不可手打：路径写错只能到建包时才发现，选文件框能直接避免
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                cell.setToolTip(self.tr("Double-click to choose an image; Delete clears it"))
                self.setItem(row, col, cell)
            elif column.default and col > 0:
                cell = QTableWidgetItem()
                # 默认值只做提示：留空时后端用默认值，改模板默认值也能生效
                cell.setToolTip(self.tr("Default: {0}").format(column.default))
                self.setItem(row, col, cell)
        self._building = False
        if animate_in:
            self.setRowHeight(row, 0)
            animate(self, 0, ROW_HEIGHT, lambda h: self._set_height(row, h))
        return row

    def ensure_blank_row(self):
        """末尾保持一个空行，在里面打字就等于新增条目。"""
        if self.rowCount() == 0 or not self.is_empty_row(self.rowCount() - 1):
            self.add_row(animate_in=self.isVisible())

    def ensure_rows(self, count: int):
        while self.rowCount() < count:
            self.add_row()

    def remove_row(self, row: int):
        if not 0 <= row < self.rowCount():
            return
        self._actions.hide()
        self._hover_row = -1

        def finish():
            # 动画期间行号可能因别的删除变化：按单元格对象找回当前行
            current = self.row(marker) if marker is not None else row
            if current >= 0:
                self.removeRow(current)
            self._errors = {(r, c) for r, c in self._errors if r < self.rowCount()}
            self.ensure_blank_row()
            self.update_conditional_columns()
            self.rows_changed.emit()

        marker = self.item(row, 0)
        animate(self, self.rowHeight(row), 0, lambda h: self._set_height(row, h, marker),
                duration=160, on_done=finish)

    def remove_selected(self):
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        if not rows and self.currentRow() >= 0:
            rows = [self.currentRow()]
        for row in rows:
            # 删除选中是批量操作，直接删不逐行播动画
            self.removeRow(row)
        self.ensure_blank_row()
        self.update_conditional_columns()
        self.rows_changed.emit()

    def reset_rows(self):
        self._building = True
        self.setRowCount(0)
        self._building = False
        self._errors.clear()
        self.ensure_blank_row()
        self.update_conditional_columns()
        self.rows_changed.emit()

    def filled_rows(self) -> list[int]:
        return [row for row in range(self.rowCount()) if not self.is_empty_row(row)]

    def is_empty_row(self, row: int) -> bool:
        # 下拉列总有值，不算用户填写
        return not any(self.cell_text(row, col) for col, column in enumerate(self._columns)
                       if not column.options)

    def row_values(self, row: int) -> tuple[str, ...]:
        return tuple(self.cell_text(row, col) for col in range(self.columnCount()))

    def _fit_height(self, *_):
        """表格高度跟着行数走。只用一个动画、每次改终点：粘贴几十行时每插一行都会触发，
        各起一个动画会几十个同时抢着改高度，界面一顿一顿的。"""
        header = self.horizontalHeader().height() if self.horizontalHeader().isVisible() else 0
        rows = min(max(self.rowCount(), 1), MAX_VISIBLE_ROWS)
        target = header + rows * ROW_HEIGHT + 2 * self.frameWidth() + 2
        if target == self._height_anim.endValue() and \
                self._height_anim.state() == QAbstractAnimation.State.Running:
            return
        self._height_anim.stop()
        if not self.isVisible():
            self.setFixedHeight(target)
            return
        if target == self.height():
            return
        self._height_anim.setStartValue(float(self.height()))
        self._height_anim.setEndValue(float(target))
        self._height_anim.start()

    def _set_height(self, row: int, height: float, marker: QTableWidgetItem | None = None):
        current = self.row(marker) if marker is not None else row
        if 0 <= current < self.rowCount():
            self.setRowHeight(current, round(height))

    def reveal(self, rows: list[int]):
        """AI 起草的结果逐行浮现：每行错开一点时间从 0 高度长出来。"""
        if not self.isVisible():
            return
        for index, row in enumerate(rows):
            marker = self.item(row, 0)
            self.setRowHeight(row, 0)
            QTimer.singleShot(index * 70, lambda r=row, m=marker: animate(
                self, 0, ROW_HEIGHT, lambda h: self._set_height(r, h, m)))

    # ---- 单元格 ----
    def cell_text(self, row: int, col: int) -> str:
        combo = self.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            return combo.currentData() or ""
        cell = self.item(row, col)
        return cell.text().strip() if cell else ""

    def set_cell_text(self, row: int, col: int, value: str, quiet: bool = False):
        """quiet：整批填行时不触发 itemChanged 的补空行，否则每填一行就在它后面多出一个空行，
        下一行 add_row 只能接在空行之后，条目之间就隔着空行；批量填完统一补一次即可。"""
        if col >= self.columnCount():
            return
        if quiet:
            self._building = True
            try:
                self.set_cell_text(row, col, value)
            finally:
                self._building = False
            return
        combo = self.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            # 填入的可能是选项名也可能是实际值，两种都认
            index = combo.findText(value)
            combo.setCurrentIndex(index if index >= 0 else max(combo.findData(value), 0))
            return
        cell = self.item(row, col)
        if cell is None:
            self.setItem(row, col, QTableWidgetItem(value))
        else:
            # 保留单元格上的"默认值"提示
            cell.setText(value)

    def paste_rows(self, text: str):
        lines = [line for line in text.splitlines() if line.strip()]
        start_row = max(self.currentRow(), 0)
        start_col = max(self.currentColumn(), 0)
        self.ensure_rows(start_row + len(lines))
        for offset, line in enumerate(lines):
            for col_offset, value in enumerate(line.split("\t")):
                self.set_cell_text(start_row + offset, start_col + col_offset, value.strip())
        self.ensure_blank_row()
        self.rows_changed.emit()

    # ---- 单独设置 ----
    def row_overrides(self, row: int) -> dict[str, str]:
        cell = self.item(row, 0)
        data = cell.data(OVERRIDES_ROLE) if cell else None
        return dict(data) if data else {}

    def set_row_overrides(self, row: int, overrides: dict[str, str]):
        cell = self.item(row, 0)
        if cell is None:
            cell = QTableWidgetItem()
            self.setItem(row, 0, cell)
        self._building = True
        cell.setData(OVERRIDES_ROLE, dict(overrides) or None)
        cell.setIcon(self._dot if overrides else QIcon())
        labels = {spec.id: spec for spec in self._override_fields}
        lines = []
        for field_id, value in overrides.items():
            spec = labels.get(field_id)
            option = next((o.label for o in spec.options if o.value == value), value) if spec else value
            lines.append(f"{short_label(spec.label or spec.id) if spec else field_id}: {option}")
        cell.setToolTip("\n".join(lines))
        self._building = False
        self.update_conditional_columns()

    def open_overrides(self, row: int):
        if not self._override_fields or not 0 <= row < self.rowCount():
            return
        marker = self.item(row, 0)

        def done(values: dict[str, str]):
            current = self.row(marker) if marker is not None else row
            if current >= 0:
                self.set_row_overrides(current, values)

        popup = OverridePopup(self._override_fields, self._defaults(), self.row_overrides(row),
                              done, self)
        popup.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        anchor = self._actions.mapToGlobal(self._actions.rect().bottomRight())
        popup.adjustSize()
        popup.move(anchor.x() - popup.width(), anchor.y() + 4)
        popup.show()

    # ---- 校验提示 ----
    def mark_error(self, row: int, col: int):
        cell = self.item(row, col)
        if cell is None:
            cell = QTableWidgetItem()
            self.setItem(row, col, cell)
        self._building = True
        cell.setBackground(_ERROR_BRUSH)
        self._building = False
        self._errors.add((row, col))
        self.scrollToItem(cell)
        self.setCurrentCell(row, col)

    def clear_errors(self):
        self._building = True
        for row, col in self._errors:
            cell = self.item(row, col)
            if cell is not None:
                cell.setBackground(QBrush())
        self._building = False
        self._errors.clear()

    # ---- 起草占位 ----
    def show_placeholders(self, count: int):
        last = self.rowCount() - 1
        top = self.rowViewportPosition(last) if last >= 0 else 0
        self._shimmer.rows = min(count, 5)
        self._shimmer.setGeometry(QRect(0, top, self.viewport().width(),
                                        self._shimmer.rows * ROW_HEIGHT))
        self._shimmer.start()

    def hide_placeholders(self):
        self._shimmer.stop()

    # ---- 事件 ----
    def _on_item_changed(self, item: QTableWidgetItem):
        if self._building:
            return
        key = (item.row(), item.column())
        if key in self._errors:
            self._errors.discard(key)
            self._building = True
            item.setBackground(QBrush())
            self._building = False
        self.ensure_blank_row()
        self.rows_changed.emit()

    def keyPressEvent(self, event):
        editing = self.state() == QAbstractItemView.State.EditingState
        if event.matches(QKeySequence.StandardKey.Paste):
            text = QGuiApplication.clipboard().text()
            if "\n" in text.strip() or "\t" in text:
                self.paste_rows(text)
                return
        if not editing and event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._is_image_cell(self.currentRow(), self.currentColumn()):
                # 图片格上按删除只清掉图片，不删整行
                self.set_cell_text(self.currentRow(), self.currentColumn(), "")
                return
            self.remove_selected()
            return
        super().keyPressEvent(event)

    def closeEditor(self, editor, hint):
        """在单元格里按回车提交后跳到下一行同一列继续输入，像在列表里逐条往下写。"""
        super().closeEditor(editor, hint)
        if hint != QAbstractItemDelegate.EndEditHint.SubmitModelCache:
            return
        row, col = self.currentRow() + 1, self.currentColumn()
        self.ensure_rows(row + 1)
        self.setCurrentCell(row, col)
        self.editItem(self.item(row, col) or self._new_item(row, col))

    def _is_image_cell(self, row: int, col: int) -> bool:
        return 0 <= col < len(self._columns) and self._columns[col].is_image and row >= 0

    def _choose_image(self, row: int, col: int):
        if not self._is_image_cell(row, col):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Choose an image"), self.cell_text(row, col), IMAGE_FILTER)
        if path:
            self.set_cell_text(row, col, path)

    def _new_item(self, row: int, col: int) -> QTableWidgetItem:
        item = QTableWidgetItem()
        self.setItem(row, col, item)
        return item

    def eventFilter(self, watched, event):
        if watched is self.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self._hover(self.rowAt(event.position().toPoint().y()))
            elif event.type() == QEvent.Type.Leave and not self._actions.underMouse():
                self._hover(-1)
        return super().eventFilter(watched, event)

    def _hover(self, row: int):
        # 末尾的空行没有东西可删可改，不显示操作
        if row < 0 or (row == self.rowCount() - 1 and self.is_empty_row(row)):
            self._hover_row = -1
            self._actions.hide()
            return
        self._hover_row = row
        self.more_btn.setVisible(bool(self._override_fields))
        self._actions.adjustSize()
        x = self.viewport().width() - self._actions.width() - 6
        y = self.rowViewportPosition(row) + (self.rowHeight(row) - self._actions.height()) // 2
        self._actions.move(x, y)
        self._actions.show()
        self._actions.raise_()

    def _set_cell_combo(self, row: int, col: int, combo: QComboBox):
        """下拉框带内边距和箭头，比表格默认行高、最小列宽大，不撑开就会被裁掉文字。"""
        self.setCellWidget(row, col, combo)
        # 先套上样式表再量尺寸，否则拿到的是未加内边距的原生尺寸（表格未显示时也还没套）
        self.ensurePolished()
        combo.ensurePolished()
        # 单元格控件会被 ::item 的 padding 再缩一圈，从样式里量出来一并补上
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 100, 100)
        inner = self.style().subElementRect(QStyle.SubElement.SE_ItemViewItemText, option, self)
        width = combo.sizeHint().width() + 100 - inner.width()
        if self.columnWidth(col) < width:
            self.setColumnWidth(col, width)
