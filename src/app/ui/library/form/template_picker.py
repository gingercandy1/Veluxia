"""选分类和模板：带图标的分类标签（底部指示条滑动过去）+ 模板小块 + 一句话简介。

分类下只有一个模板时模板小块整排隐藏；完整说明不常驻，只放在模板小块和简介的悬停提示里。
"""
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.anim import DURATION, discard, fade_slide_in
from src.app.ui.library.pack_status import category_icon, category_text, ordered_categories
from src.shared.schemas import TemplateInfo

_INDICATOR_HEIGHT = 2


class CategoryTabs(QWidget):
    """一排分类标签，选中项下方的指示条会滑到新位置。"""
    changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, _INDICATOR_HEIGHT + 4)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._group.buttonClicked.connect(self._on_clicked)
        self._buttons: dict[str, QPushButton] = {}
        self.indicator = QFrame(self)
        self.indicator.setObjectName("library_tab_indicator")
        self._slide = QPropertyAnimation(self.indicator, b"geometry", self)
        self._slide.setDuration(DURATION + 40)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)

    def set_categories(self, pack_types: list[str]):
        for button in self._buttons.values():
            self._group.removeButton(button)
            discard(button)
        self._buttons = {}
        for pack_type in pack_types:
            button = QPushButton(QIcon(category_icon(pack_type)), category_text(pack_type))
            button.setObjectName("library_tab")
            button.setCheckable(True)
            button.setIconSize(QSize(16, 16))
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setProperty("pack_type", pack_type)
            self._group.addButton(button)
            self._layout.insertWidget(self._layout.count() - 1, button)
            self._buttons[pack_type] = button
        self.indicator.setVisible(bool(pack_types))

    def select(self, pack_type: str, animate: bool = False):
        button = self._buttons.get(pack_type)
        if button is None:
            return
        button.setChecked(True)
        self._move_indicator(animate)

    def current(self) -> str:
        button = self._group.checkedButton()
        return button.property("pack_type") if button else ""

    def _on_clicked(self, button: QPushButton):
        self._move_indicator(animate=True)
        self.changed.emit(button.property("pack_type"))

    def _target(self) -> QRect | None:
        button = self._group.checkedButton()
        if button is None:
            return None
        geometry = button.geometry()
        inset = 10
        return QRect(geometry.x() + inset, geometry.bottom() + 3,
                     max(geometry.width() - inset * 2, 8), _INDICATOR_HEIGHT)

    def _move_indicator(self, animate: bool):
        target = self._target()
        if target is None:
            return
        self._slide.stop()
        if animate and self.isVisible() and self.indicator.width() > 0:
            self._slide.setStartValue(self.indicator.geometry())
            self._slide.setEndValue(target)
            self._slide.start()
        else:
            self.indicator.setGeometry(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._move_indicator(animate=False)

    def showEvent(self, event):
        super().showEvent(event)
        self._move_indicator(animate=False)


class TemplatePicker(QWidget):
    """分类标签 + 模板小块 + 简介。换分类或模板时发出 template_changed。"""
    template_changed = Signal(str)
    category_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._templates: list[TemplateInfo] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.tabs = CategoryTabs()
        self.tabs.changed.connect(self._on_category)
        layout.addWidget(self.tabs)

        self.chips_row = QWidget()
        self._chips_layout = QHBoxLayout(self.chips_row)
        self._chips_layout.setContentsMargins(0, 0, 0, 0)
        self._chips_layout.setSpacing(6)
        self._chips_layout.addStretch()
        self._chips = QButtonGroup(self)
        self._chips.setExclusive(True)
        self._chips.buttonClicked.connect(self._on_chip)
        layout.addWidget(self.chips_row)

        self.summary_label = QLabel()
        self.summary_label.setObjectName("library_form_summary")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

    # ---- 对外 ----
    def set_templates(self, templates: list[TemplateInfo]):
        self._templates = templates
        types = {t.type for t in templates}
        self.tabs.set_categories([t for t in ordered_categories({t: [] for t in types})
                                  if t in types])

    def select_category(self, pack_type: str):
        """选中分类（不存在时选第一个）并选中它的第一个模板。"""
        types = [t.type for t in self._templates]
        if pack_type not in types:
            pack_type = self._first_category()
        self.tabs.select(pack_type)
        self._fill_chips(pack_type)

    def select_template(self, template_id: str):
        template = self._find(template_id)
        if template is None:
            return
        if self.tabs.current() != template.type:
            self.tabs.select(template.type)
            self._fill_chips(template.type, template_id)
            return
        for button in self._chips.buttons():
            if button.property("template_id") == template_id:
                button.setChecked(True)
        self._show_summary(template)
        self.template_changed.emit(template_id)

    def current_category(self) -> str:
        return self.tabs.current()

    def current_template(self) -> TemplateInfo | None:
        button = self._chips.checkedButton()
        return self._find(button.property("template_id")) if button else None

    # ---- 内部 ----
    def _first_category(self) -> str:
        types = {t.type for t in self._templates}
        ordered = [t for t in ordered_categories({t: [] for t in types}) if t in types]
        return ordered[0] if ordered else ""

    def _find(self, template_id: str) -> TemplateInfo | None:
        return next((t for t in self._templates if t.id == template_id), None)

    def _on_category(self, pack_type: str):
        self.category_changed.emit(pack_type)
        self._fill_chips(pack_type)
        fade_slide_in(self.chips_row)
        fade_slide_in(self.summary_label)

    def _on_chip(self, button: QPushButton):
        template = self._find(button.property("template_id"))
        if template is not None:
            self._show_summary(template)
            fade_slide_in(self.summary_label)
            self.template_changed.emit(template.id)

    def _fill_chips(self, pack_type: str, select_id: str = ""):
        for button in self._chips.buttons():
            self._chips.removeButton(button)
            discard(button)
        templates = [t for t in self._templates if t.type == pack_type]
        for template in templates:
            button = QPushButton(template.name or template.id)
            button.setObjectName("library_chip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(template.description)
            button.setProperty("template_id", template.id)
            self._chips.addButton(button)
            self._chips_layout.insertWidget(self._chips_layout.count() - 1, button)
        # 只有一个模板时不必选，整排隐藏
        self.chips_row.setVisible(len(templates) > 1)
        chosen = self._find(select_id) if select_id else (templates[0] if templates else None)
        if chosen is None:
            self.summary_label.clear()
            self.template_changed.emit("")
            return
        for button in self._chips.buttons():
            button.setChecked(button.property("template_id") == chosen.id)
        self._show_summary(chosen)
        self.template_changed.emit(chosen.id)

    def _show_summary(self, template: TemplateInfo):
        self.summary_label.setText(template.summary or template.description)
        self.summary_label.setToolTip(template.description if template.summary else "")
