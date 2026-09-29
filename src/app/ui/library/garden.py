from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import LoadingStrip
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
        # 每个资源包可被搜到的文字：包名、模板名、分类，以及各条目的描述和字段
        self._search_text: dict[str, str] = {}
        # 分类 → (分块, 标题, 新建占位卡, 该分类下的资源包 id)，过滤时原地显隐，不重建
        self._section_parts: dict[str, tuple[QWidget, QLabel, QWidget, list[str]]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(self.tr("Library"))
        title.setObjectName("library_title")
        header.addWidget(title)
        header.addStretch()
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("library_search")
        self.search_edit.setPlaceholderText(self.tr("Search packs and items"))
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedWidth(240)
        self.search_edit.textChanged.connect(self._apply_filter)
        header.addWidget(self.search_edit)
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

        # 首次进入要先后取模板和资源包列表，远程后端时会空白一两秒
        self.loading_strip = LoadingStrip(height=3)
        self.loading_strip.setMaximumWidth(360)
        self.loading_label = QLabel(self.tr("Loading packs…"))
        self.loading_label.setObjectName("detail_muted")
        layout.addWidget(self.loading_strip)
        layout.addWidget(self.loading_label)
        self.loading_strip.set_loading(False)
        self.loading_label.hide()

        self.empty_label = QLabel(self.tr("No packs match your search."))
        self.empty_label.setObjectName("detail_muted")
        self.empty_label.hide()
        layout.addWidget(self.empty_label)

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

    def set_loading(self, loading: bool):
        """已经显示着卡片时（再次进入页面刷新）不打扰，只在还是空白时显示加载条。"""
        show = loading and not self._cards
        self.loading_strip.set_loading(show)
        self.loading_label.setVisible(show)

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
            self._apply_filter()
            return
        self._layout_key = layout_key
        self._clear_sections()

        for pack_type in ordered_categories(by_type):
            group = by_type.get(pack_type, [])
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(10)
            heading = QLabel()
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
            self._section_parts[pack_type] = (section, heading, add,
                                              [p.manifest.id for p in group])
        self._sections.addStretch()
        self._apply_filter()

    def _apply_filter(self):
        """按空格分词，每个词都要命中（不分大小写）。搜索时隐藏新建占位卡和没有命中的分类。"""
        words = self.search_edit.text().lower().split()
        any_match = False
        for pack_type, (section, heading, add, ids) in self._section_parts.items():
            matched = [pack_id for pack_id in ids
                       if all(word in self._search_text.get(pack_id, "") for word in words)]
            for pack_id in ids:
                self._cards[pack_id].setVisible(pack_id in matched)
            add.setVisible(not words)
            section.setVisible(not words or bool(matched))
            count = f"{len(matched)}/{len(ids)}" if words else f"{len(ids)}"
            heading.setText(f"{category_text(pack_type)}  ·  {count}")
            any_match = any_match or bool(matched)
        self.empty_label.setVisible(bool(words) and not any_match)

    def card(self, pack_id: str) -> Card | None:
        return self._cards.get(pack_id)

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

    def _update_card(self, card: Card, pack: PackResponse, template: TemplateInfo):
        manifest = pack.manifest
        card.set_name(manifest.name or template.name or manifest.id)
        parts = [manifest.name, template.name, manifest.id, category_text(manifest.type),
                 manifest.style.prompt]
        for item in manifest.items:
            parts.extend([item.id, item.prompt, *item.fields.values()])
        self._search_text[manifest.id] = "\n".join(parts).lower()
        progress = pack_progress(manifest, template, running=pack.running)
        card.set_progress(progress, f"{progress.done}/{progress.total}",
                          percent=step_fraction(manifest, template), active=pack.running)

    def _clear_sections(self):
        self._cards.clear()
        self._search_text.clear()
        self._section_parts.clear()
        while self._sections.count():
            entry = self._sections.takeAt(0)
            if entry.widget() is not None:
                entry.widget().hide()
                entry.widget().deleteLater()
