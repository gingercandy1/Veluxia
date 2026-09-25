from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.cards import PACK_THUMB, AddCard, Card
from src.app.ui.library.flow_layout import FlowLayout
from src.app.ui.library.pack_status import (
    category_text,
    ordered_categories,
    pack_progress,
    step_fraction,
)
from src.shared.schemas import PackResponse, TemplateInfo


class GardenView(BaseWidget):
    """资料库首页：按分类分块的卡片墙，每块自动换行铺满，块尾是新建这一类的占位卡。"""
    pack_opened = Signal(str)       # pack id
    new_requested = Signal(str)     # 分类；空字符串表示不预选

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library_garden")
        self._cards: dict[str, Card] = {}
        self._layout_key: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(self.tr("Library"))
        title.setObjectName("library_title")
        header.addWidget(title)
        header.addStretch()
        self.new_btn = QPushButton(self.tr("New pack"))
        self.new_btn.setObjectName("library_primary_btn")
        self.new_btn.clicked.connect(lambda: self.new_requested.emit(""))
        header.addWidget(self.new_btn)
        layout.addLayout(header)

        self.error_label = QLabel()
        self.error_label.setObjectName("library_error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._content.setObjectName("library_scroll_content")
        self._sections = QVBoxLayout(self._content)
        self._sections.setContentsMargins(0, 8, 0, 24)
        self._sections.setSpacing(20)
        scroll.setWidget(self._content)
        layout.addWidget(scroll, 1)

    def set_error(self, message: str):
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def set_packs(self, packs: list[PackResponse], templates: dict[str, TemplateInfo]):
        """资源包没增删时只原地刷新进度（执行中每两秒轮询一次，重建会闪、会丢缩略图）；
        增删了才整体重建。"""
        by_type: dict[str, list[PackResponse]] = {}
        for pack in packs:
            by_type.setdefault(pack.manifest.type, []).append(pack)
        layout_key = [(t, [p.manifest.id for p in by_type.get(t, [])])
                      for t in ordered_categories(by_type)]
        if layout_key == self._layout_key:
            for pack in packs:
                self._update_card(self._cards[pack.manifest.id], pack,
                                  templates[pack.manifest.template])
            return
        self._layout_key = layout_key
        self._clear_sections()

        for pack_type in ordered_categories(by_type):
            group = by_type.get(pack_type, [])
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(10)
            heading = QLabel(f"{category_text(pack_type)}  ·  {len(group)}")
            heading.setObjectName("library_section_title")
            section_layout.addWidget(heading)

            flow = FlowLayout(spacing=14)
            for pack in group:
                card = self._pack_card(pack, templates[pack.manifest.template])
                flow.addWidget(card)
            add = AddCard(self.tr("New {0} pack").format(category_text(pack_type)))
            add.clicked.connect(lambda t=pack_type: self.new_requested.emit(t))
            flow.addWidget(add)
            section_layout.addLayout(flow)
            self._sections.addWidget(section)
        self._sections.addStretch()

    def set_thumbnail(self, pack_id: str, pixmap: QPixmap):
        card = self._cards.get(pack_id)
        if card is not None:
            card.set_thumbnail(pixmap)

    def _pack_card(self, pack: PackResponse, template: TemplateInfo) -> Card:
        manifest = pack.manifest
        card = Card(PACK_THUMB, category_text(manifest.type), manifest.type)
        self._update_card(card, pack, template)
        card.clicked.connect(lambda pack_id=manifest.id: self.pack_opened.emit(pack_id))
        self._cards[manifest.id] = card
        return card

    @staticmethod
    def _update_card(card: Card, pack: PackResponse, template: TemplateInfo):
        manifest = pack.manifest
        card.set_name(manifest.name or template.name or manifest.id)
        progress = pack_progress(manifest, template, running=pack.running)
        card.set_progress(progress, f"{progress.done}/{progress.total}",
                          percent=step_fraction(manifest, template))

    def _clear_sections(self):
        self._cards.clear()
        while self._sections.count():
            entry = self._sections.takeAt(0)
            if entry.widget() is not None:
                entry.widget().hide()
                entry.widget().deleteLater()
