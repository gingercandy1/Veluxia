"""出场角色（对话模板）：一排角色小块，点开浮层编辑名字、角色包、音色描述，最后一块是 +。

对话通常只有两三个角色，小块比三列表格省地方；声线来自角色包的小块带高亮边框，
从描述现场设计声线的是普通边框，悬停提示里写明。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.anim import FlowHost, discard, fade_slide_in
from src.app.ui.library.flow_layout import FlowLayout
from src.shared.schemas import CastMember


class _CastEditor(QFrame):
    """编辑一个出场角色的浮层；关闭时把结果交回去。"""

    def __init__(self, member: CastMember, characters: list[tuple[str, str]], on_done, on_remove,
                 parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("library_popup")
        self._on_done = on_done
        self._removed = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self.name_edit = QLineEdit(member.name)
        self.name_edit.setPlaceholderText(self.tr("Name"))
        layout.addWidget(self.name_edit)
        self.character_combo = QComboBox()
        self.character_combo.addItem(self.tr("(New voice)"), "")
        for label, value in characters:
            self.character_combo.addItem(label, value)
        self.character_combo.setCurrentIndex(max(self.character_combo.findData(member.character), 0))
        layout.addWidget(self.character_combo)
        self.voice_edit = QLineEdit(member.description)
        self.voice_edit.setPlaceholderText(self.tr("Voice description"))
        layout.addWidget(self.voice_edit)
        remove = QPushButton(self.tr("Remove"))
        remove.setObjectName("library_ghost_btn")
        remove.clicked.connect(lambda: (on_remove(), self._mark_removed()))
        layout.addWidget(remove)
        self.setMinimumWidth(260)

    def _mark_removed(self):
        self._removed = True
        self.close()

    def hideEvent(self, event):
        super().hideEvent(event)
        if not self._removed:
            self._on_done(CastMember(name=self.name_edit.text().strip(),
                                     character=self.character_combo.currentData() or "",
                                     description=self.voice_edit.text().strip()))


class CastChips(FlowHost):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._members: list[CastMember] = []
        self._characters: list[tuple[str, str]] = []
        self._flow = FlowLayout(self, spacing=8)
        self.add_btn = QPushButton("+")
        self.add_btn.setObjectName("library_chip_add")
        self.add_btn.setToolTip(self.tr("Add a role"))
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.clicked.connect(self._on_add)
        self._rebuild()

    # ---- 对外 ----
    def members(self) -> list[CastMember]:
        return [m for m in self._members if m.name or m.character or m.description]

    def add_member(self, member: CastMember):
        self._members.append(member)
        self._rebuild()
        self.changed.emit()

    def clear(self):
        self._members = []
        self._rebuild()

    def set_characters(self, characters: list[tuple[str, str]]):
        self._characters = characters
        self._rebuild()

    # ---- 内部 ----
    def _rebuild(self):
        while self._flow.count():
            item = self._flow.takeAt(0)
            widget = item.widget() if item else None
            if widget is not None and widget is not self.add_btn:
                discard(widget)
        linked = {value: label for label, value in self._characters}
        for index, member in enumerate(self._members):
            chip = QPushButton(member.name or self.tr("Unnamed"))
            chip.setObjectName("library_chip")
            chip.setProperty("linked", bool(member.character))
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            if member.character:
                chip.setToolTip(self.tr("Voice from {0}").format(
                    linked.get(member.character, member.character)))
            else:
                chip.setToolTip(member.description or self.tr("Voice designed from description"))
            chip.clicked.connect(lambda _=False, i=index, c=chip: self._edit(i, c))
            self._flow.addWidget(chip)
        self._flow.addWidget(self.add_btn)

    def _on_add(self):
        self._members.append(CastMember(name=""))
        self._rebuild()
        chip = self._flow.itemAt(len(self._members) - 1).widget()
        fade_slide_in(chip)
        self._edit(len(self._members) - 1, chip)

    def _edit(self, index: int, anchor: QWidget):
        def done(member: CastMember):
            if index < len(self._members):
                self._members[index] = member
                # 什么都没填的空角色不留
                if not (member.name or member.character or member.description):
                    del self._members[index]
                self._rebuild()
                self.changed.emit()

        def remove():
            if index < len(self._members):
                del self._members[index]
                self._rebuild()
                self.changed.emit()

        editor = _CastEditor(self._members[index], self._characters, done, remove, self)
        editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        editor.move(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        editor.show()
        editor.name_edit.setFocus()
