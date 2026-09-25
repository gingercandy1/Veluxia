from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.pack_status import category_text, ordered_categories
from src.shared.schemas import (
    CastMember,
    CollectionStyle,
    CreatePackRequest,
    NewCollectionItem,
    TemplateFieldInfo,
    TemplateInfo,
)

DIALOGUE_TYPE = "dialogue"


class PasteTable(QTableWidget):
    """Ctrl+V 粘贴多行文本时一行一个条目、Tab 分列，从 Excel / 文本批量填条目不用逐格输入。"""
    rows_needed = Signal(int)  # 需要的总行数，由表单负责补行（补行时要带上下拉控件）

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.StandardKey.Paste):
            text = QGuiApplication.clipboard().text()
            if "\n" in text.strip() or "\t" in text:
                self.paste_rows(text)
                return
        super().keyPressEvent(event)

    def paste_rows(self, text: str):
        lines = [line for line in text.splitlines() if line.strip()]
        start_row = max(self.currentRow(), 0)
        start_col = max(self.currentColumn(), 0)
        self.rows_needed.emit(start_row + len(lines))
        for offset, line in enumerate(lines):
            for col_offset, value in enumerate(line.split("\t")):
                self.set_cell_text(start_row + offset, start_col + col_offset, value.strip())

    def cell_text(self, row: int, col: int) -> str:
        combo = self.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            return combo.currentData() or ""
        cell = self.item(row, col)
        return cell.text().strip() if cell else ""

    def set_cell_text(self, row: int, col: int, value: str):
        if col >= self.columnCount():
            return
        combo = self.cellWidget(row, col)
        if isinstance(combo, QComboBox):
            # 粘贴的可能是选项名也可能是实际值，两种都认
            index = combo.findText(value)
            combo.setCurrentIndex(index if index >= 0 else max(combo.findData(value), 0))
            return
        self.setItem(row, col, QTableWidgetItem(value))


class PackForm(BaseWidget):
    """新建资源包：选分类和模板，填风格锁，条目用表格录入（模板字段各占一列）；
    对话包另填出场角色，可以绑定已有角色包的声线。"""
    create_requested = Signal(object)  # CreatePackRequest
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pack_form")
        self._templates: list[TemplateInfo] = []
        self._characters: list[tuple[str, str]] = []  # (显示名, "<资源包 id>/<条目 id>")
        self._fields: list[TemplateFieldInfo] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        body = QWidget()
        body.setObjectName("library_scroll_content")
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        back = QPushButton(QIcon(":/svg/back.svg"), self.tr("Library"))
        back.setObjectName("library_back_btn")
        back.setIconSize(QSize(16, 16))
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        header.addWidget(back)
        title = QLabel(self.tr("New pack"))
        title.setObjectName("library_title")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.category_combo = QComboBox()
        self.category_combo.currentIndexChanged.connect(self._fill_template_combo)
        form.addRow(self._label(self.tr("Category")), self.category_combo)
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        form.addRow(self._label(self.tr("Template")), self.template_combo)
        self.description_label = QLabel()
        self.description_label.setObjectName("detail_muted")
        self.description_label.setWordWrap(True)
        form.addRow(QLabel(), self.description_label)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(self.tr("e.g. Forest props"))
        form.addRow(self._label(self.tr("Name")), self.name_edit)
        self.style_edit = QPlainTextEdit()
        self.style_edit.setObjectName("library_text_edit")
        self.style_edit.setPlaceholderText(
            self.tr("Style lock, appended to every item, e.g. hand-painted, dark teal tones"))
        self.style_edit.setFixedHeight(64)
        form.addRow(self._label(self.tr("Style")), self.style_edit)
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText(self.tr("Optional"))
        form.addRow(self._label(self.tr("Negative")), self.negative_edit)
        layout.addLayout(form)

        self.cast_section = QWidget()
        cast_layout = QVBoxLayout(self.cast_section)
        cast_layout.setContentsMargins(0, 0, 0, 0)
        cast_layout.addWidget(self._heading(self.tr("Cast")))
        cast_layout.addWidget(self._hint(self.tr(
            "Link a character pack to reuse its voice, or describe the voice for a new role.")))
        self.cast_table = QTableWidget(0, 3)
        self.cast_table.setHorizontalHeaderLabels(
            [self.tr("Name"), self.tr("Character pack"), self.tr("Voice description")])
        self._setup_table(self.cast_table, stretch_column=2)
        self.cast_table.setMinimumHeight(140)
        cast_layout.addWidget(self.cast_table)
        cast_layout.addLayout(self._row_buttons(self._add_cast_row, self.cast_table))
        layout.addWidget(self.cast_section)

        layout.addWidget(self._heading(self.tr("Items")))
        layout.addWidget(self._hint(self.tr(
            "One row per asset. Paste multiple lines (tab-separated for columns) to add rows.")))
        self.items_table = PasteTable(0, 1)
        self._setup_table(self.items_table, stretch_column=0)
        self.items_table.setMinimumHeight(240)
        self.items_table.rows_needed.connect(self._ensure_item_rows)
        layout.addWidget(self.items_table, 1)
        layout.addLayout(self._row_buttons(lambda: self._add_item_row(), self.items_table))

        self.error_label = QLabel()
        self.error_label.setObjectName("library_error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch()
        self.create_btn = QPushButton(self.tr("Create"))
        self.create_btn.setObjectName("library_primary_btn")
        self.create_btn.clicked.connect(self._on_create)
        buttons.addWidget(self.create_btn)
        layout.addLayout(buttons)
        self.cast_section.hide()

    # ---- 对外 ----
    def set_templates(self, templates: list[TemplateInfo]):
        # 每次进入资料库都会重新拉模板；没变化时不重建，免得清掉用户填了一半的条目
        if templates == self._templates:
            return
        self._templates = templates
        current = self.category_combo.currentData()
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        types = {t.type for t in templates}
        for pack_type in ordered_categories({t: [] for t in types}):
            if pack_type in types:
                self.category_combo.addItem(category_text(pack_type), pack_type)
        self.category_combo.blockSignals(False)
        self.preselect_category(current or "")

    def set_characters(self, characters: list[tuple[str, str]]):
        self._characters = characters
        for row in range(self.cast_table.rowCount()):
            combo = self.cast_table.cellWidget(row, 1)
            value = combo.currentData()
            self._fill_character_combo(combo)
            combo.setCurrentIndex(max(combo.findData(value), 0))

    def preselect_category(self, pack_type: str):
        index = self.category_combo.findData(pack_type)
        self.category_combo.setCurrentIndex(max(index, 0))
        self._fill_template_combo()

    def reset(self):
        self.name_edit.clear()
        self.style_edit.clear()
        self.negative_edit.clear()
        self.cast_table.setRowCount(0)
        self._on_template_changed()
        self.set_error("")

    def set_busy(self, busy: bool):
        self.create_btn.setEnabled(not busy)

    def set_error(self, message: str):
        # 空标签也占一行布局高度，没有错误时隐藏
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def build_request(self) -> CreatePackRequest:
        """读表格生成请求；空行跳过。必填字段等规则由后端校验，错误原样提示。"""
        template = self.current_template()
        if template is None:
            raise ValueError(self.tr("No template available. Is the backend running?"))
        items = []
        for row in range(self.items_table.rowCount()):
            prompt = self.items_table.cell_text(row, 0)
            fields = {spec.id: self.items_table.cell_text(row, col)
                      for col, spec in enumerate(self._fields, start=1)}
            typed = [value for spec, value in zip(self._fields, fields.values(), strict=True)
                     if not spec.options]
            if not prompt and not any(typed):
                continue
            items.append(NewCollectionItem(prompt=prompt,
                                           fields={k: v for k, v in fields.items() if v}))
        if not items:
            raise ValueError(self.tr("Add at least one item."))
        return CreatePackRequest(
            name=self.name_edit.text().strip(),
            template=template.id,
            style=CollectionStyle(prompt=self.style_edit.toPlainText().strip(),
                                  negative=self.negative_edit.text().strip()),
            cast=self._cast() if template.type == DIALOGUE_TYPE else [],
            items=items,
        )

    def current_template(self) -> TemplateInfo | None:
        template_id = self.template_combo.currentData()
        return next((t for t in self._templates if t.id == template_id), None)

    # ---- 模板切换 ----
    def _fill_template_combo(self):
        pack_type = self.category_combo.currentData()
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        for template in self._templates:
            if template.type == pack_type:
                self.template_combo.addItem(template.name or template.id, template.id)
        self.template_combo.blockSignals(False)
        self._on_template_changed()

    def _on_template_changed(self):
        template = self.current_template()
        self._fields = list(template.fields) if template else []
        self.description_label.setText(template.description if template else "")
        prompt_label = (template.prompt_label if template else "") or self.tr("Prompt")
        self.items_table.clear()
        self.items_table.setRowCount(0)
        self.items_table.setColumnCount(1 + len(self._fields))
        headers = [prompt_label] + [
            (spec.label or spec.id) + (" *" if spec.required else "") for spec in self._fields]
        self.items_table.setHorizontalHeaderLabels(headers)
        for _ in range(3):
            self._add_item_row()
        is_dialogue = template is not None and template.type == DIALOGUE_TYPE
        self.cast_section.setVisible(is_dialogue)
        if is_dialogue and self.cast_table.rowCount() == 0:
            self._add_cast_row()

    # ---- 行 ----
    def _add_item_row(self):
        row = self.items_table.rowCount()
        self.items_table.insertRow(row)
        for col, spec in enumerate(self._fields, start=1):
            if spec.options:
                combo = QComboBox()
                for option in spec.options:
                    combo.addItem(option.label or option.value, option.value)
                combo.setCurrentIndex(max(combo.findData(spec.default), 0))
                self.items_table.setCellWidget(row, col, combo)
            elif spec.default:
                cell = QTableWidgetItem()
                # 默认值只做提示：留空时后端用默认值，改模板默认值也能生效
                cell.setToolTip(self.tr("Default: {0}").format(spec.default))
                self.items_table.setItem(row, col, cell)

    def _ensure_item_rows(self, count: int):
        while self.items_table.rowCount() < count:
            self._add_item_row()

    def _add_cast_row(self):
        row = self.cast_table.rowCount()
        self.cast_table.insertRow(row)
        combo = QComboBox()
        self._fill_character_combo(combo)
        self.cast_table.setCellWidget(row, 1, combo)

    def _fill_character_combo(self, combo: QComboBox):
        combo.clear()
        combo.addItem(self.tr("(New voice)"), "")
        for label, value in self._characters:
            combo.addItem(label, value)

    def _cast(self) -> list[CastMember]:
        cast = []
        for row in range(self.cast_table.rowCount()):
            name_cell = self.cast_table.item(row, 0)
            description_cell = self.cast_table.item(row, 2)
            name = name_cell.text().strip() if name_cell else ""
            description = description_cell.text().strip() if description_cell else ""
            character = self.cast_table.cellWidget(row, 1).currentData() or ""
            if name or description or character:
                cast.append(CastMember(name=name, character=character, description=description))
        return cast

    # ---- 工具 ----
    def _row_buttons(self, add, table: QTableWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        add_btn = QPushButton(self.tr("Add row"))
        remove_btn = QPushButton(self.tr("Remove row"))
        add_btn.clicked.connect(add)
        remove_btn.clicked.connect(lambda: self._remove_selected(table))
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch()
        return row

    @staticmethod
    def _remove_selected(table: QTableWidget):
        rows = sorted({index.row() for index in table.selectedIndexes()}, reverse=True)
        if not rows and table.currentRow() >= 0:
            rows = [table.currentRow()]
        for row in rows:
            table.removeRow(row)

    @staticmethod
    def _setup_table(table: QTableWidget, stretch_column: int):
        table.setObjectName("library_table")
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setMinimumSectionSize(90)
        table.setColumnWidth(stretch_column, 320)

    @staticmethod
    def _label(text: str) -> QLabel:
        # addRow 传字符串时 Qt 自建的标签没有 objectName，qss 挂不上
        label = QLabel(text)
        label.setObjectName("library_form_label")
        return label

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("library_section_title")
        return label

    @staticmethod
    def _hint(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("detail_muted")
        label.setWordWrap(True)
        return label

    def _on_create(self):
        try:
            request = self.build_request()
        except ValueError as exc:
            self.set_error(str(exc))
            return
        self.set_error("")
        self.create_requested.emit(request)
