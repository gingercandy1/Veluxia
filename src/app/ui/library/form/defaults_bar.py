"""整包设置：模板里可选的下拉字段（视角、无缝、时长……）统一设一次，对所有条目生效。

必填的下拉（如角色性别）因人而异，留在条目表格里每行选；这里只放可选的。
需要绑定角色的模板（动作、转面），角色选择也放在这一排。
"""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

from src.app.ui.base.anim import FlowHost, discard
from src.app.ui.library.flow_layout import FlowLayout
from src.shared.schemas import TemplateFieldInfo


def short_label(label: str) -> str:
    """只取括号前的短名，括号里的补充说明放进悬停提示：界面上的字越少越好。"""
    short = label
    for mark in ("（", "("):
        head = short.split(mark, 1)[0].strip()
        if head:
            short = head
    return short


def pack_level(spec: TemplateFieldInfo) -> bool:
    """可选的下拉放到整包设置；必填的下拉和要打字的字段留在每一行。"""
    return bool(spec.options) and not spec.required


class DefaultsBar(FlowHost):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library_defaults_bar")
        self._flow = FlowLayout(self, spacing=18)
        self._combos: dict[str, QComboBox] = {}
        self._fields: list[TemplateFieldInfo] = []
        # 绑定角色：模板声明了 source 时才显示；刷新角色列表时要保留当前选择，所以控件常驻
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(160)
        self.source_combo.currentIndexChanged.connect(self.changed)
        self.source_section = self._entry(self.tr("Character"), self.source_combo)
        self.source_section.setParent(self)
        self.source_section.hide()

    # ---- 对外 ----
    def set_fields(self, fields: list[TemplateFieldInfo], needs_source: bool):
        """按模板重建下拉；保留 source 选择。"""
        self._fields = [spec for spec in fields if pack_level(spec)]
        while self._flow.count():
            item = self._flow.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None and widget is not self.source_section:
                discard(widget)
        self._combos = {}
        if needs_source:
            self._flow.addWidget(self.source_section)
        self.source_section.setVisible(needs_source)
        for spec in self._fields:
            combo = QComboBox()
            for option in spec.options:
                combo.addItem(option.label or option.value, option.value)
            combo.setCurrentIndex(max(combo.findData(spec.default), 0))
            combo.currentIndexChanged.connect(self.changed)
            entry = self._entry(short_label(spec.label or spec.id), combo)
            if short_label(spec.label or spec.id) != (spec.label or spec.id):
                entry.setToolTip(spec.label)
            self._flow.addWidget(entry)
            self._combos[spec.id] = combo
        self.setVisible(bool(self._combos) or needs_source)

    def fields(self) -> list[TemplateFieldInfo]:
        return list(self._fields)

    def values(self) -> dict[str, str]:
        return {field_id: combo.currentData() or "" for field_id, combo in self._combos.items()}

    def set_values(self, values: dict[str, str]):
        """恢复上次的选择；不认识的字段、已不在选项里的值忽略。"""
        for field_id, value in values.items():
            combo = self._combos.get(field_id)
            index = combo.findData(value) if combo is not None else -1
            if index >= 0:
                combo.setCurrentIndex(index)

    def set_characters(self, characters: list[tuple[str, str]]):
        value = self.source_combo.currentData()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for label, reference in characters:
            self.source_combo.addItem(label, reference)
        self.source_combo.setCurrentIndex(max(self.source_combo.findData(value), 0))
        self.source_combo.blockSignals(False)

    @staticmethod
    def _entry(text: str, combo: QComboBox) -> QWidget:
        entry = QWidget()
        layout = QHBoxLayout(entry)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(text)
        label.setObjectName("library_form_label")
        layout.addWidget(label)
        layout.addWidget(combo)
        return entry
