from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QGuiApplication, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionViewItem,
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
    DraftItemsRequest,
    NewCollectionItem,
    StylePreset,
    TemplateFieldInfo,
    TemplateInfo,
)

DIALOGUE_TYPE = "dialogue"
# 动作包「添加常用动作」的固定主题：不用用户想主题，按绑定的角色起草一组基础动作
COMMON_MOTIONS_THEME = "游戏里最常用的基础动作：待机、移动、跳跃、受击、攻击、死亡"
COMMON_MOTIONS_COUNT = 6
# 与后端 drafts.MAX_DRAFT_COUNT 一致：再多 LLM 的 4096 上下文装不下
MAX_DRAFT_COUNT = 20


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
        cell = self.item(row, col)
        if cell is None:
            self.setItem(row, col, QTableWidgetItem(value))
        else:
            # 保留新行上的"默认值"提示
            cell.setText(value)


class PackForm(BaseWidget):
    """新建资源包：选分类和模板，填风格锁，条目用表格录入（模板字段各占一列）；
    对话包另填出场角色，可以绑定已有角色包的声线。"""
    create_requested = Signal(object)  # CreatePackRequest
    draft_requested = Signal(object)  # DraftItemsRequest
    style_save_requested = Signal(object)  # StylePreset
    style_delete_requested = Signal(str)  # 预设 id
    back_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pack_form")
        self._templates: list[TemplateInfo] = []
        self._characters: list[tuple[str, str]] = []  # (显示名, "<资源包 id>/<条目 id>")
        self._styles: list[StylePreset] = []
        self._fields: list[TemplateFieldInfo] = []
        # 上次 AI 起草填进去的行内容：再次起草时，没被用户改过的这些行会被换掉
        self._drafted_rows: set[tuple[str, ...]] = set()

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

        # 标签放在输入框上方、两列并排：标签列放左侧会让上面的表单和下面顶格的表格左边缘错开，
        # 标签全放上方又太高，把"创建"按钮挤出首屏
        form = QGridLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(12)
        form.setColumnStretch(0, 1)
        form.setColumnStretch(1, 1)
        self.category_combo = QComboBox()
        self.category_combo.currentIndexChanged.connect(self._fill_template_combo)
        form.addLayout(self._field(self.tr("Category"), self.category_combo), 0, 0)
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        form.addLayout(self._field(self.tr("Template"), self.template_combo), 0, 1)
        self.description_label = QLabel()
        self.description_label.setObjectName("detail_muted")
        self.description_label.setWordWrap(True)
        form.addWidget(self.description_label, 1, 0, 1, 2)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText(self.tr("e.g. Forest props"))
        form.addLayout(self._field(self.tr("Name"), self.name_edit), 2, 0)
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText(self.tr("Optional"))
        form.addLayout(self._field(self.tr("Negative"), self.negative_edit), 2, 1)
        self.style_edit = QPlainTextEdit()
        self.style_edit.setObjectName("library_text_edit")
        self.style_edit.setPlaceholderText(
            self.tr("Style lock, appended to every item, e.g. hand-painted, dark teal tones"))
        self.style_edit.setFixedHeight(64)
        # 项目级风格预设（ADR 0006）：选中后把内容填进风格锁，不同资源包的画风才统一
        preset_row = QWidget()
        preset_layout = QHBoxLayout(preset_row)
        preset_layout.setContentsMargins(0, 0, 0, 0)
        self.preset_combo = QComboBox()
        # 只响应用户手动选择；刷新列表时不要覆盖用户已经改过的风格锁
        self.preset_combo.activated.connect(self._on_preset_chosen)
        preset_layout.addWidget(self.preset_combo, 1)
        self.save_preset_btn = QPushButton(self.tr("Save as preset"))
        self.save_preset_btn.setToolTip(self.tr(
            "Save the style and negative above as a project preset. "
            "Using an existing name overwrites that preset."))
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        preset_layout.addWidget(self.save_preset_btn)
        self.delete_preset_btn = QPushButton(self.tr("Delete preset"))
        self.delete_preset_btn.clicked.connect(self._on_delete_preset)
        preset_layout.addWidget(self.delete_preset_btn)
        form.addLayout(self._field(self.tr("Style preset"), preset_row), 3, 0, 1, 2)
        form.addLayout(self._field(self.tr("Style"), self.style_edit), 4, 0, 1, 2)
        # 动作包等模板要绑定角色包里的一个角色（ADR 0006），用它的立绘做参考图
        self.source_section = QWidget()
        self.source_combo = QComboBox()
        self.source_section.setLayout(self._field(self.tr("Character"), self.source_combo))
        self.source_section.layout().setContentsMargins(0, 0, 0, 0)
        form.addWidget(self.source_section, 5, 0)
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
        draft_row = QHBoxLayout()
        self.draft_theme_edit = QLineEdit()
        self.draft_theme_edit.setPlaceholderText(
            self.tr("AI draft theme, e.g. ground details for a glowing forest"))
        self.draft_theme_edit.returnPressed.connect(self._on_draft)
        draft_row.addWidget(self.draft_theme_edit, 1)
        self.draft_count_spin = QSpinBox()
        self.draft_count_spin.setRange(1, MAX_DRAFT_COUNT)
        self.draft_count_spin.setValue(10)
        self.draft_count_spin.setPrefix(self.tr("Count "))
        draft_row.addWidget(self.draft_count_spin)
        self.draft_btn = QPushButton(self.tr("AI draft"))
        self.draft_btn.setToolTip(self.tr(
            "Replaces empty rows and untouched AI rows; rows you typed or edited are kept. "
            "The first run loads the text model and takes about a minute."))
        self.draft_btn.clicked.connect(self._on_draft)
        draft_row.addWidget(self.draft_btn)
        self.common_motions_btn = QPushButton(self.tr("Add common motions"))
        self.common_motions_btn.setToolTip(self.tr(
            "Drafts idle, move, jump, hit, attack and death for the chosen character. "
            "The AI writes each motion to fit what the character is, so check and edit the rows "
            "before creating the pack."))
        self.common_motions_btn.clicked.connect(self._on_common_motions)
        draft_row.addWidget(self.common_motions_btn)
        layout.addLayout(draft_row)
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
        self.source_section.hide()
        self.common_motions_btn.hide()
        self.set_styles([])

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
        value = self.source_combo.currentData()
        self.source_combo.clear()
        for label, reference in characters:
            self.source_combo.addItem(label, reference)
        self.source_combo.setCurrentIndex(max(self.source_combo.findData(value), 0))

    def set_styles(self, styles: list[StylePreset], select_name: str = ""):
        """select_name：刚保存的预设，列表刷新后选中它。"""
        self._styles = styles
        value = self.preset_combo.currentData()
        self.preset_combo.clear()
        self.preset_combo.addItem(self.tr("(No preset)"), "")
        for style in styles:
            self.preset_combo.addItem(style.name, style.id)
        selected = next((s.id for s in styles if s.name == select_name), value)
        self.preset_combo.setCurrentIndex(max(self.preset_combo.findData(selected), 0))
        self._update_preset_buttons()

    def preselect_category(self, pack_type: str):
        index = self.category_combo.findData(pack_type)
        self.category_combo.setCurrentIndex(max(index, 0))
        self._fill_template_combo()

    def reset(self):
        self.name_edit.clear()
        self.style_edit.clear()
        self.negative_edit.clear()
        self.preset_combo.setCurrentIndex(0)
        self._update_preset_buttons()
        self.cast_table.setRowCount(0)
        self._on_template_changed()
        self.set_error("")

    def set_busy(self, busy: bool):
        self.create_btn.setEnabled(not busy)

    def set_drafting(self, drafting: bool):
        self.draft_btn.setEnabled(not drafting)
        self.common_motions_btn.setEnabled(not drafting)
        self.draft_btn.setText(self.tr("Drafting...") if drafting else self.tr("AI draft"))

    def apply_drafts(self, template_id: str, items: list[NewCollectionItem]):
        """把 AI 起草的条目填进表格：空行和上次起草后没动过的行换成新条目，用户写过的行保留。"""
        template = self.current_template()
        if template is None or template.id != template_id:
            # 起草期间切换了模板，字段列已经对不上
            return
        for row in reversed(range(self.items_table.rowCount())):
            if self._is_empty_row(row) or self._row_values(row) in self._drafted_rows:
                self.items_table.removeRow(row)
        self._drafted_rows = set()
        for item in items:
            row = self.items_table.rowCount()
            self._add_item_row()
            self.items_table.set_cell_text(row, 0, item.prompt)
            for col, spec in enumerate(self._fields, start=1):
                if spec.id in item.fields:
                    self.items_table.set_cell_text(row, col, item.fields[spec.id])
            self._drafted_rows.add(self._row_values(row))

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
        source = (self.source_combo.currentData() or "") if template.source else ""
        if template.source and not source:
            raise ValueError(self.tr("Create a character pack first, then pick a character here."))
        style = CollectionStyle(prompt=self.style_edit.toPlainText().strip(),
                                negative=self.negative_edit.text().strip())
        preset = self._selected_preset()
        return CreatePackRequest(
            name=self.name_edit.text().strip(),
            template=template.id,
            style=style,
            # 选了预设后又手动改过风格锁，就不再算用了这个预设，否则详情页会一直提示"预设已更新"
            style_preset=preset.id if preset is not None and preset.style() == style else "",
            cast=self._cast() if template.type == DIALOGUE_TYPE else [],
            source=source,
            items=items,
        )

    def current_template(self) -> TemplateInfo | None:
        template_id = self.template_combo.currentData()
        return next((t for t in self._templates if t.id == template_id), None)

    # ---- 风格预设 ----
    def _selected_preset(self) -> StylePreset | None:
        style_id = self.preset_combo.currentData()
        return next((s for s in self._styles if s.id == style_id), None)

    def _update_preset_buttons(self):
        self.delete_preset_btn.setEnabled(self._selected_preset() is not None)

    def _on_preset_chosen(self):
        preset = self._selected_preset()
        if preset is not None:
            self.style_edit.setPlainText(preset.prompt)
            self.negative_edit.setText(preset.negative)
        self._update_preset_buttons()

    def _on_save_preset(self):
        prompt = self.style_edit.toPlainText().strip()
        negative = self.negative_edit.text().strip()
        if not prompt and not negative:
            self.set_error(self.tr("Fill in the style or negative before saving a preset."))
            return
        current = self._selected_preset()
        name, ok = QInputDialog.getText(self, self.tr("Save as preset"), self.tr("Preset name"),
                                        text=current.name if current else "")
        name = name.strip()
        if not ok or not name:
            return
        # 同名即覆盖：带上已有预设的 id，后端按 id 修改而不是报重名
        existing = next((s for s in self._styles if s.name == name), None)
        self.set_error("")
        self.style_save_requested.emit(StylePreset(
            id=existing.id if existing else "", name=name, prompt=prompt, negative=negative))

    def _on_delete_preset(self):
        preset = self._selected_preset()
        if preset is not None:
            self.style_delete_requested.emit(preset.id)

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
        self._drafted_rows = set()
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
        has_source = template is not None and bool(template.source)
        self.source_section.setVisible(has_source)
        self.common_motions_btn.setVisible(has_source)

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
                self._set_cell_combo(self.items_table, row, col, combo)
            elif spec.default:
                cell = QTableWidgetItem()
                # 默认值只做提示：留空时后端用默认值，改模板默认值也能生效
                cell.setToolTip(self.tr("Default: {0}").format(spec.default))
                self.items_table.setItem(row, col, cell)

    def _row_values(self, row: int) -> tuple[str, ...]:
        return tuple(self.items_table.cell_text(row, col)
                     for col in range(self.items_table.columnCount()))

    def _is_empty_row(self, row: int) -> bool:
        # 下拉列总有值，不算用户填写；与 build_request 跳过空行的规则一致
        return not self.items_table.cell_text(row, 0) and not any(
            self.items_table.cell_text(row, col)
            for col, spec in enumerate(self._fields, start=1) if not spec.options)

    def _ensure_item_rows(self, count: int):
        while self.items_table.rowCount() < count:
            self._add_item_row()

    def _add_cast_row(self):
        row = self.cast_table.rowCount()
        self.cast_table.insertRow(row)
        combo = QComboBox()
        self._fill_character_combo(combo)
        self._set_cell_combo(self.cast_table, row, 1, combo)

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
    def _set_cell_combo(table: QTableWidget, row: int, col: int, combo: QComboBox):
        """下拉框带内边距和箭头，比表格默认行高、最小列宽大，不撑开就会被裁掉文字。"""
        table.setCellWidget(row, col, combo)
        # 先套上样式表再量尺寸，否则拿到的是未加内边距的原生尺寸（表格未显示时也还没套）
        table.ensurePolished()
        combo.ensurePolished()
        # 单元格控件会被 ::item 的 padding 和网格线再缩一圈，从样式里量出来一并补上
        option = QStyleOptionViewItem()
        option.rect = QRect(0, 0, 100, 100)
        inner = table.style().subElementRect(QStyle.SubElement.SE_ItemViewItemText, option, table)
        grid = 1 if table.showGrid() else 0
        height = combo.sizeHint().height() + 100 - inner.height() + grid
        width = combo.sizeHint().width() + 100 - inner.width() + grid
        if table.rowHeight(row) < height:
            table.setRowHeight(row, height)
        if table.columnWidth(col) < width:
            table.setColumnWidth(col, width)

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
    def _field(text: str, widget: QWidget) -> QVBoxLayout:
        label = QLabel(text)
        label.setObjectName("library_form_label")
        column = QVBoxLayout()
        column.setSpacing(6)
        column.addWidget(label)
        column.addWidget(widget)
        return column

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

    def _on_draft(self):
        theme = self.draft_theme_edit.text().strip()
        if self.current_template() is not None and not theme:
            self.set_error(self.tr("Enter a theme for the AI to draft items from."))
            return
        self._request_draft(theme, self.draft_count_spin.value())

    def _on_common_motions(self):
        self._request_draft(COMMON_MOTIONS_THEME, COMMON_MOTIONS_COUNT)

    def _request_draft(self, theme: str, count: int):
        template = self.current_template()
        if template is None:
            self.set_error(self.tr("No template available. Is the backend running?"))
            return
        # 动作包要按绑定角色的样子写动作，没选角色就没法起草
        source = (self.source_combo.currentData() or "") if template.source else ""
        if template.source and not source:
            self.set_error(self.tr("Create a character pack first, then pick a character here."))
            return
        # 表格里已有的条目（含上次起草的）都告诉模型，再点一次拿到的是新条目
        existing = [self.items_table.cell_text(row, 0) for row in range(self.items_table.rowCount())]
        self.set_error("")
        self.draft_requested.emit(DraftItemsRequest(
            template=template.id,
            theme=theme,
            count=count,
            style=self.style_edit.toPlainText().strip(),
            cast=self._cast() if template.type == DIALOGUE_TYPE else [],
            exclude=[prompt for prompt in existing if prompt],
            source=source,
        ))

    def _on_create(self):
        try:
            request = self.build_request()
        except ValueError as exc:
            self.set_error(str(exc))
            return
        self.set_error("")
        self.create_requested.emit(request)
