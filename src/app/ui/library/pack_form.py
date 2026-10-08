"""新建资源包：选分类和模板 → 整包设置 → 条目 → 风格（默认折叠）→ 创建并生成。

界面原则：常驻文字尽量少（说明放进悬停提示、输入框只用占位文字），操作少（空行自动长出、
可选下拉整包设一次、记住上次的选择），关键操作有动画反馈。
"""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.anim import (
    WINDOW_BACKGROUND,
    Collapsible,
    bounce,
    cover_fade_in,
    shake,
)
from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.form.cast_chips import CastChips
from src.app.ui.library.form.defaults_bar import DefaultsBar, pack_level, short_label
from src.app.ui.library.form.items_table import PROMPT_COLUMN, Column, ItemsTable
from src.app.ui.library.form.template_picker import TemplatePicker
from src.shared.schemas import (
    AddItemsRequest,
    CollectionStyle,
    CreatePackRequest,
    DraftItemsRequest,
    Manifest,
    NewCollectionItem,
    StylePreset,
    TemplateInfo,
)
from src.shared.settings import ConfigManager

DIALOGUE_TYPE = "dialogue"
COMMON_MOTIONS_THEME = "游戏里最常用的基础动作：待机、移动、跳跃、受击、攻击、死亡"
COMMON_MOTIONS_COUNT = 6
MAX_DRAFT_COUNT = 20                 # 与后端 drafts.MAX_DRAFT_COUNT 一致：再多 LLM 的 4096 上下文装不下
AUTO_NAME_CHARS = 12                 # 名称留空时自动命名，取第一条描述的前几个字
_STYLE_SUMMARY_CHARS = 24
PAGE_BACKGROUND = WINDOW_BACKGROUND  # 与 library.qss 里 #pack_form 的底色一致：淡入盖层用它


class FormMemory:
    """按模板记住上次的整包设置和风格预设，下次打开同一个模板时自动带出。"""
    SECTION, KEY = "library", "form_memory"

    def load(self, template_id: str) -> dict:
        return dict(ConfigManager().get(self.SECTION, self.KEY, {}).get(template_id, {}))

    def save(self, template_id: str, defaults: dict[str, str], preset: str):
        config = ConfigManager()
        memory = dict(config.get(self.SECTION, self.KEY, {}))
        memory[template_id] = {"defaults": defaults, "preset": preset}
        config.set(self.SECTION, self.KEY, memory)
        config.save()


def count_segments(value: str) -> int:
    """与后端 template.count_segments 同样的切分规则：全角竖线也算分隔，空段不计。"""
    return sum(1 for part in value.replace("｜", "|").split("|") if part.strip())


class PackForm(BaseWidget):
    """append=True 时是"往已有资源包追加条目"：模板、名称、风格锁、出场角色和来源角色都沿用
    包里的，界面只留条目表格、整包下拉和 AI 起草。"""
    create_requested = Signal(object)  # CreatePackRequest
    add_requested = Signal(str, object)  # 资源包 id, AddItemsRequest
    draft_requested = Signal(object)  # DraftItemsRequest
    style_save_requested = Signal(object)  # StylePreset
    style_delete_requested = Signal(str)  # 预设 id
    back_requested = Signal()

    def __init__(self, parent=None, memory: FormMemory | None = None, append: bool = False):
        super().__init__(parent)
        self.setObjectName("pack_form")
        self._memory = memory or FormMemory()
        self._append = append
        # 追加模式下的目标资源包
        self._target: Manifest | None = None
        self._templates: list[TemplateInfo] = []
        self._styles: list[StylePreset] = []
        self._busy = False
        # 上次 AI 起草填进去的行内容：再次起草时，没被用户改过的这些行会被换掉
        self._drafted_rows: set[tuple[str, ...]] = set()
        # 换了分类就清空条目；同一分类内换模板保留描述列
        self._category_switched = False
        # 记住的风格预设在预设列表到来之前就要用到时，先记下，列表到了再套用
        self._pending_preset = ""
        # 点创建时的整包设置：后端确认创建成功（reset）后才记住，被拒绝的那次不算
        self._pending_memory: tuple[str, dict[str, str], str] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)
        body = QWidget()
        body.setObjectName("library_scroll_content")
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 6, 28, 20)
        layout.setSpacing(18)

        self.picker = TemplatePicker()
        self.picker.category_changed.connect(self._on_category_changed)
        self.picker.template_changed.connect(self._on_template_changed)
        layout.addWidget(self.picker)

        # 模板相关的部分放在一个容器里，换模板时整体淡入
        self.content = QWidget()
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(12)
        self.defaults = DefaultsBar()
        self.defaults.changed.connect(self._on_defaults_changed)
        self.source_section = self.defaults.source_section
        self.source_combo = self.defaults.source_combo
        content.addWidget(self.defaults)

        self.cast_section = QWidget()
        cast_row = QHBoxLayout(self.cast_section)
        cast_row.setContentsMargins(0, 0, 0, 0)
        cast_row.setSpacing(10)
        cast_label = QLabel(self.tr("Cast"))
        cast_label.setObjectName("library_form_label")
        cast_label.setToolTip(self.tr(
            "Link a character pack to reuse its voice, or describe the voice for a new role."))
        cast_row.addWidget(cast_label, 0, Qt.AlignmentFlag.AlignTop)
        self.cast_chips = CastChips()
        cast_row.addWidget(self.cast_chips, 1)
        content.addWidget(self.cast_section)

        content.addLayout(self._build_items_header())
        self.draft_panel = Collapsible(self._build_draft_row())
        content.addWidget(self.draft_panel)
        self.items_table = ItemsTable()
        self.items_table.rows_changed.connect(self._update_create_button)
        content.addWidget(self.items_table)
        layout.addWidget(self.content)

        self.style_toggle = QPushButton()
        self.style_toggle.setObjectName("library_style_toggle")
        self.style_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.style_toggle.clicked.connect(self._toggle_style)
        layout.addWidget(self.style_toggle, 0, Qt.AlignmentFlag.AlignLeft)
        self.style_panel = Collapsible(self._build_style_panel())
        layout.addWidget(self.style_panel)
        # 多出来的高度全给末尾：stretch 为 0 时 Qt 会把它分给上面各块，块与块之间出现空洞
        layout.addStretch(1)

        outer.addWidget(self._build_footer())

        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            QShortcut(QKeySequence(keys), self, self._on_create,
                      context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.cast_section.hide()
        self.set_styles([])
        self._update_style_toggle()
        self._update_create_button()
        if append:
            for widget in (self.picker, self.name_edit, self.style_toggle, self.style_panel):
                widget.hide()

    # ---- 构建 ----
    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("library_form_header")
        header = QHBoxLayout(bar)
        header.setContentsMargins(24, 14, 28, 10)
        back = QPushButton(QIcon(":/svg/back.svg"), self.tr("Library"))
        back.setObjectName("library_back_btn")
        back.setIconSize(QSize(16, 16))
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        header.addWidget(back)
        self.title_label = QLabel(self.tr("New pack"))
        self.title_label.setObjectName("library_title")
        header.addWidget(self.title_label)
        header.addStretch()
        # 名称可以不填：标题栏里点一下就能改，留空时自动命名
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("library_name_edit")
        self.name_edit.setPlaceholderText(self.tr("Untitled pack"))
        self.name_edit.setToolTip(self.tr("Leave empty to name it after the template and first item"))
        self.name_edit.addAction(QIcon(":/svg/edit.svg"), QLineEdit.ActionPosition.TrailingPosition)
        self.name_edit.setFixedWidth(240)
        header.addWidget(self.name_edit)
        return bar

    def _build_items_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        title = QLabel(self.tr("Items"))
        title.setObjectName("library_section_title")
        title.setToolTip(self.tr(
            "One row per asset. Paste rows from Excel or text, or let AI draft them."))
        row.addWidget(title)
        row.addStretch()
        # 模板带了常用条目才显示（如角色动作：待机、走、跑、攻击、死亡……）
        self.preset_items_btn = QPushButton(self.tr("Add common items"))
        self.preset_items_btn.setObjectName("library_ghost_btn")
        self.preset_items_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preset_items_btn.setToolTip(self.tr(
            "Fill in the template's common items. Rows you already wrote are kept, "
            "and you can edit or delete any row afterwards."))
        self.preset_items_btn.clicked.connect(self.add_presets)
        self.preset_items_btn.hide()
        row.addWidget(self.preset_items_btn)
        self.draft_toggle_btn = QPushButton("✨ " + self.tr("AI draft"))
        self.draft_toggle_btn.setObjectName("library_ghost_btn")
        self.draft_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.draft_toggle_btn.clicked.connect(self._toggle_draft)
        row.addWidget(self.draft_toggle_btn)
        return row

    def _build_draft_row(self) -> QWidget:
        panel = QWidget()
        row = QHBoxLayout(panel)
        row.setContentsMargins(0, 0, 0, 4)
        row.setSpacing(8)
        self.draft_theme_edit = QLineEdit()
        self.draft_theme_edit.setPlaceholderText(
            self.tr("AI draft theme, e.g. ground details for a glowing forest"))
        self.draft_theme_edit.returnPressed.connect(self._on_draft)
        row.addWidget(self.draft_theme_edit, 1)
        self.draft_count_spin = QSpinBox()
        self.draft_count_spin.setRange(1, MAX_DRAFT_COUNT)
        self.draft_count_spin.setValue(10)
        self.draft_count_spin.setPrefix(self.tr("Count "))
        row.addWidget(self.draft_count_spin)
        self.draft_btn = QPushButton(self.tr("Draft"))
        self.draft_btn.setObjectName("library_primary_btn")
        self.draft_btn.setToolTip(self.tr(
            "Replaces empty rows and untouched AI rows; rows you typed or edited are kept. "
            "The first run loads the text model and takes about a minute."))
        self.draft_btn.clicked.connect(self._on_draft)
        row.addWidget(self.draft_btn)
        self.common_motions_btn = QPushButton(self.tr("Add common motions"))
        self.common_motions_btn.setObjectName("library_ghost_btn")
        self.common_motions_btn.setToolTip(self.tr(
            "Drafts idle, move, jump, hit, attack and death for the chosen character. "
            "The AI writes each motion to fit what the character is, so check and edit the rows "
            "before creating the pack."))
        self.common_motions_btn.clicked.connect(self._on_common_motions)
        self.common_motions_btn.hide()
        row.addWidget(self.common_motions_btn)
        return panel

    def _build_style_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(10)
        # 项目级风格预设（ADR 0006）：选中后把内容填进风格锁，不同资源包的画风才统一
        presets = QHBoxLayout()
        presets.setSpacing(8)
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(200)
        # 只响应用户手动选择；刷新列表时不要覆盖用户已经改过的风格锁
        self.preset_combo.activated.connect(self._on_preset_chosen)
        presets.addWidget(self.preset_combo)
        self.save_preset_btn = QPushButton(self.tr("Save as preset"))
        self.save_preset_btn.setObjectName("library_ghost_btn")
        self.save_preset_btn.setToolTip(self.tr(
            "Save the style and negative above as a project preset. "
            "Using an existing name overwrites that preset."))
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        presets.addWidget(self.save_preset_btn)
        self.delete_preset_btn = QPushButton(self.tr("Delete preset"))
        self.delete_preset_btn.setObjectName("library_ghost_btn")
        self.delete_preset_btn.clicked.connect(self._on_delete_preset)
        presets.addWidget(self.delete_preset_btn)
        presets.addStretch()
        layout.addLayout(presets)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        self.style_edit = QPlainTextEdit()
        self.style_edit.setObjectName("library_text_edit")
        self.style_edit.setPlaceholderText(
            self.tr("Style lock, appended to every item, e.g. hand-painted, dark teal tones"))
        self.style_edit.setFixedHeight(64)
        self.style_edit.textChanged.connect(self._update_style_toggle)
        grid.addWidget(self.style_edit, 0, 0)
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText(self.tr("Negative (optional)"))
        self.negative_edit.textChanged.connect(self._update_style_toggle)
        grid.addWidget(self.negative_edit, 0, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(grid)
        return panel

    def _build_footer(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("library_form_footer")
        footer = QHBoxLayout(bar)
        footer.setContentsMargins(28, 10, 28, 12)
        footer.setSpacing(16)
        self.error_label = QLabel()
        self.error_label.setObjectName("library_error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        footer.addWidget(self.error_label, 1)
        footer.addStretch()
        self.create_btn = QPushButton()
        self.create_btn.setObjectName("library_create_btn")
        self.create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.create_btn.setToolTip(self.tr("Ctrl+Enter"))
        self.create_btn.clicked.connect(self._on_create)
        footer.addWidget(self.create_btn)
        return bar

    # ---- 对外 ----
    def set_templates(self, templates: list[TemplateInfo]):
        # 每次进入资料库都会重新拉模板；没变化时不重建，免得清掉用户填了一半的条目
        if templates == self._templates:
            return
        current = self.picker.current_category()
        self._templates = templates
        self.picker.set_templates(templates)
        self._category_switched = True
        self.picker.select_category(current)

    def set_characters(self, characters: list[tuple[str, str]]):
        self.defaults.set_characters(characters)
        self.cast_chips.set_characters(characters)

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
        if self._pending_preset:
            self._apply_preset_memory(self._pending_preset)
        self._update_preset_buttons()
        self._update_style_toggle()

    def set_target(self, manifest: Manifest, template: TemplateInfo):
        """追加模式：指定往哪个资源包里加。换了资源包才清空表格，回到同一个包时保留没提交的行。"""
        same = self._target is not None and self._target.id == manifest.id
        self._target = manifest
        self.title_label.setText(self.tr("Add to \"{0}\"").format(
            manifest.name or template.name or manifest.id))
        if same and self._templates == [template]:
            return
        self._templates = [template]
        self.picker.set_templates([template])
        self._category_switched = True
        self.picker.select_category(template.type)
        self.draft_theme_edit.clear()
        self.draft_panel.set_expanded(False)
        self.set_error("")

    def target_id(self) -> str:
        return self._target.id if self._target is not None else ""

    def preselect_category(self, pack_type: str):
        """从资料库某个分类的"新建"进来：分类没变就保留已填内容。"""
        if pack_type and pack_type != self.picker.current_category():
            self._category_switched = True
            self.picker.select_category(pack_type)

    def reset(self):
        """创建成功后调用：先记住这次创建用的整包设置和风格预设，再清空并重新带出。"""
        if self._pending_memory is not None:
            self._memory.save(*self._pending_memory)
            self._pending_memory = None
        self.name_edit.clear()
        self.style_edit.clear()
        self.negative_edit.clear()
        self.preset_combo.setCurrentIndex(0)
        self.cast_chips.clear()
        self.draft_theme_edit.clear()
        self.draft_panel.set_expanded(False)
        self._drafted_rows = set()
        self.items_table.reset_rows()
        template = self.current_template()
        if template is not None:
            self._apply_memory(template.id)
        self._update_preset_buttons()
        self.set_error("")

    def set_busy(self, busy: bool):
        self._busy = busy
        self._update_create_button()

    def set_drafting(self, drafting: bool):
        self.draft_btn.setEnabled(not drafting)
        self.common_motions_btn.setEnabled(not drafting)
        self.draft_btn.setText(self.tr("Drafting...") if drafting else self.tr("Draft"))
        if drafting:
            self.items_table.show_placeholders(self.draft_count_spin.value())
        else:
            self.items_table.hide_placeholders()

    def apply_drafts(self, template_id: str, items: list[NewCollectionItem]):
        """把 AI 起草的条目填进表格：空行和上次起草后没动过的行换成新条目，用户写过的行保留。"""
        template = self.current_template()
        if template is None or template.id != template_id:
            # 起草期间切换了模板，字段列已经对不上
            return
        table = self.items_table
        for row in reversed(range(table.rowCount())):
            if table.is_empty_row(row) or table.row_values(row) in self._drafted_rows:
                table.removeRow(row)
        self._drafted_rows = set()
        self._fill_rows(items, track_drafted=True)

    def add_presets(self):
        """把模板自带的常用条目（如角色动作的待机、走、跑……）一键填进表格。

        只追加：用户写过的行、AI 起草的行都保留；表格里已经有同名预设（按条目的字段和描述
        判断）就不再重复添加，所以连点几次也不会越加越多。填进来的行和手写的一样可以改、删。
        预设不记进 _drafted_rows：之后再 AI 起草不应该把它们清掉。
        """
        template = self.current_template()
        if template is None or not template.presets:
            return
        table = self.items_table
        present = {table.cell_text(row, 0) for row in table.filled_rows()}
        fresh = [NewCollectionItem(prompt=preset.prompt, fields=dict(preset.fields))
                 for preset in template.presets if preset.prompt not in present]
        for row in reversed(range(table.rowCount())):
            if table.is_empty_row(row):
                table.removeRow(row)
        self._fill_rows(fresh, track_drafted=False)

    def _fill_rows(self, items: list[NewCollectionItem], track_drafted: bool):
        table = self.items_table
        columns = table.columns()
        defaults = self.defaults.values()
        added = []
        for item in items:
            row = table.add_row()
            table.set_cell_text(row, 0, item.prompt, quiet=True)
            for col, column in enumerate(columns[1:], start=1):
                if column.id in item.fields:
                    table.set_cell_text(row, col, item.fields[column.id], quiet=True)
            # 模型给某条选了和整包不同的值（如某个场景要多屏），记成这一行的单独设置
            overrides = {field_id: value for field_id, value in item.fields.items()
                         if field_id in defaults and value and value != defaults[field_id]}
            if overrides:
                table.set_row_overrides(row, overrides)
            if track_drafted:
                self._drafted_rows.add(table.row_values(row))
            added.append(row)
        table.ensure_blank_row()
        table.update_conditional_columns()
        table.reveal(added)
        self._update_create_button()

    def set_error(self, message: str):
        # 空标签也占一行布局高度，没有错误时隐藏
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        if message:
            # 创建被拒绝：这次的设置不记，免得之后别处调 reset 时误存
            self._pending_memory = None
            shake(self.error_label)

    def build_request(self) -> CreatePackRequest:
        """读表格生成请求；空行跳过。能定位到格子的错误先在界面上标出来，其余规则由后端校验。"""
        template = self.current_template()
        if template is None:
            raise ValueError(self.tr("No template available. Is the backend running?"))
        items = self._build_items()
        source = (self.source_combo.currentData() or "") if template.source else ""
        if template.source and not source:
            raise ValueError(self.tr("Create a character pack first, then pick a character here."))
        style = CollectionStyle(prompt=self.style_edit.toPlainText().strip(),
                                negative=self.negative_edit.text().strip())
        preset = self._selected_preset()
        return CreatePackRequest(
            name=self.name_edit.text().strip() or self._auto_name(template, items),
            template=template.id,
            style=style,
            # 选了预设后又手动改过风格锁，就不再算用了这个预设，否则详情页会一直提示"预设已更新"
            style_preset=preset.id if preset is not None and preset.style() == style else "",
            cast=self.cast_chips.members() if template.type == DIALOGUE_TYPE else [],
            source=source,
            items=items,
        )

    def build_add_request(self) -> AddItemsRequest:
        if self.current_template() is None:
            raise ValueError(self.tr("No template available. Is the backend running?"))
        return AddItemsRequest(items=self._build_items())

    def _build_items(self) -> list[NewCollectionItem]:
        table = self.items_table
        table.clear_errors()
        columns = table.columns()
        defaults = self.defaults.values()
        items = []
        for row in table.filled_rows():
            values = {**defaults, **table.row_overrides(row)}
            fields = {}
            for col, column in enumerate(columns[1:], start=1):
                text = table.cell_text(row, col)
                if column.required and not text:
                    table.mark_error(row, col)
                    raise ValueError(self.tr("Fill in {0}").format(short_label(column.label)))
                if column.segments_field and text:
                    screens = values.get(column.segments_field, "1")
                    count = count_segments(text)
                    if screens.isdigit() and count != int(screens):
                        table.mark_error(row, col)
                        raise ValueError(self.tr(
                            "{0} has {1} segments but the length is {2} screens").format(
                            short_label(column.label), count, screens))
                if text:
                    fields[column.id] = text
            fields.update({k: v for k, v in values.items() if v})
            items.append(NewCollectionItem(prompt=table.cell_text(row, 0), fields=fields))
        if not items:
            raise ValueError(self.tr("Add at least one item."))
        return items

    def current_template(self) -> TemplateInfo | None:
        return self.picker.current_template()

    # ---- 模板切换 ----
    def _on_category_changed(self, _pack_type: str):
        self._category_switched = True

    def _on_template_changed(self, _template_id: str):
        template = self.current_template()
        table = self.items_table
        # 同一分类内换模板保留描述列；换了分类清空
        kept = [] if self._category_switched else [
            table.cell_text(row, 0) for row in table.filled_rows() if table.cell_text(row, 0)]
        self._category_switched = False
        self._drafted_rows = set()
        fields = list(template.fields) if template else []
        self.preset_items_btn.setVisible(bool(template and template.presets))
        self.common_motions_btn.setVisible(bool(template and template.source))
        # 追加时来源角色沿用包里绑定的，不再选
        self.defaults.set_fields(
            fields, needs_source=bool(template and template.source) and not self._append)
        if template is not None:
            self._apply_memory(template.id)
        prompt_label = (template.prompt_label if template else "") or self.tr("Prompt")
        columns = [Column(PROMPT_COLUMN, prompt_label)] + [
            Column.from_field(spec) for spec in fields if not pack_level(spec)]
        table.set_columns(columns, self.defaults.fields(), self.defaults.values)
        for prompt in kept:
            row = table.rowCount() - 1
            table.set_cell_text(row, 0, prompt)
            table.ensure_blank_row()
        is_dialogue = template is not None and template.type == DIALOGUE_TYPE
        self.cast_section.setVisible(is_dialogue and not self._append)
        if not is_dialogue:
            self.cast_chips.clear()
        self._update_create_button()
        # 只淡入不滑动：这块高度随内容变化，布局一直在重排，横向滑动会和布局抢位置；
        # 里面有表格和下拉框，用盖层淡入而不是透明度特效，不卡也不会画错位
        cover_fade_in(self.content, PAGE_BACKGROUND)

    def _on_defaults_changed(self):
        self.items_table.update_conditional_columns()

    # ---- 记住上次的选择 ----
    def _apply_memory(self, template_id: str):
        memory = self._memory.load(template_id)
        self.defaults.set_values(memory.get("defaults", {}))
        preset = memory.get("preset", "")
        # 用户已经写了风格就不覆盖
        if preset and not self.style_edit.toPlainText().strip() and not self.negative_edit.text():
            self._apply_preset_memory(preset)

    def _apply_preset_memory(self, preset_id: str):
        index = self.preset_combo.findData(preset_id)
        if index < 0:
            # 预设列表还没到：等 set_styles 再套用
            self._pending_preset = preset_id
            return
        self._pending_preset = ""
        self.preset_combo.setCurrentIndex(index)
        self._on_preset_chosen()

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
        self._update_style_toggle()

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

    def _toggle_style(self):
        self.style_panel.toggle()
        self._update_style_toggle()

    def _update_style_toggle(self):
        """折叠时的一行摘要：用了哪个预设，或者风格锁的开头几个词。"""
        preset = self._selected_preset()
        style = self.style_edit.toPlainText().strip()
        if preset is not None and preset.style() == CollectionStyle(
                prompt=style, negative=self.negative_edit.text().strip()):
            summary = preset.name
        elif style:
            summary = style if len(style) <= _STYLE_SUMMARY_CHARS else \
                style[:_STYLE_SUMMARY_CHARS] + "…"
        else:
            summary = self.tr("None")
        arrow = "▾" if self.style_panel.is_expanded() else "▸"
        # tr 不能写在 f-string 里：lupdate 提取不到，会被当成废弃字符串删掉
        label = self.tr("Style")
        self.style_toggle.setText(f"{label}  ·  {summary}  {arrow}")

    # ---- 起草 ----
    def _toggle_draft(self):
        self.draft_panel.toggle()
        if self.draft_panel.is_expanded():
            self.draft_theme_edit.setFocus()

    def _on_draft(self):
        theme = self.draft_theme_edit.text().strip()
        if not theme:
            self.draft_panel.set_expanded(True)
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
        source = (self._target.source if self._target is not None else
                  self.source_combo.currentData() or "") if template.source else ""
        if template.source and not source:
            self.set_error(self.tr("Create a character pack first, then pick a character here."))
            return
        # 表格里已有的条目（含上次起草的）都告诉模型，再点一次拿到的是新条目；
        # 追加时包里已有的条目也算，避免起草出和已生成素材重复的东西
        table = self.items_table
        existing = [table.cell_text(row, 0) for row in table.filled_rows()]
        if self._target is not None:
            existing += [item.prompt for item in self._target.items]
            style = self._target.style.prompt
            cast = self._target.cast
        else:
            style = self.style_edit.toPlainText().strip()
            cast = self.cast_chips.members() if template.type == DIALOGUE_TYPE else []
        self.set_error("")
        self.draft_requested.emit(DraftItemsRequest(
            template=template.id,
            theme=theme,
            count=count,
            style=style,
            cast=cast,
            exclude=[prompt for prompt in existing if prompt],
            source=source,
        ))

    # ---- 创建 ----
    def _update_create_button(self):
        count = len(self.items_table.filled_rows())
        text = self.tr("Add and generate ({0})") if self._append else \
            self.tr("Create and generate ({0})")
        self.create_btn.setText(text.format(count))
        self.create_btn.setEnabled(count > 0 and not self._busy)

    def _auto_name(self, template: TemplateInfo, items: list[NewCollectionItem]) -> str:
        first = next((item.prompt for item in items if item.prompt), "")
        name = template.name or template.id
        return f"{name} · {first[:AUTO_NAME_CHARS]}" if first else name

    def _on_create(self):
        if self._busy:
            return
        if self._append:
            self._on_add()
            return
        try:
            request = self.build_request()
        except ValueError as exc:
            self.set_error(str(exc))
            return
        self.set_error("")
        self._pending_memory = (request.template, self.defaults.values(), request.style_preset)
        bounce(self.create_btn)
        self.create_requested.emit(request)

    def _on_add(self):
        if self._target is None:
            return
        try:
            request = self.build_add_request()
        except ValueError as exc:
            self.set_error(str(exc))
            return
        self.set_error("")
        bounce(self.create_btn)
        self.add_requested.emit(self._target.id, request)

    def keyPressEvent(self, event):
        """Esc：正在输入框里编辑时只退出编辑，否则返回资料库。"""
        if event.key() == Qt.Key.Key_Escape:
            focus = QApplication.focusWidget()
            if isinstance(focus, (QLineEdit, QPlainTextEdit, QSpinBox)) and self.isAncestorOf(focus):
                focus.clearFocus()
            else:
                self.back_requested.emit()
            return
        super().keyPressEvent(event)
