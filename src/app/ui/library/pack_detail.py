from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.cards import ASSET_THUMB, Card
from src.app.ui.library.detail_panel import DetailPanel
from src.app.ui.library.flow_layout import FlowLayout
from src.app.ui.library.pack_status import (
    MEDIA_ORDER,
    asset_progress,
    category_text,
    deliverables,
    item_title,
    media_kind,
    media_text,
    ordered_categories,
    pack_progress,
    status_text,
    step_chain,
)
from src.app.ui.message.image_preview import ImagePreviewOverlay
from src.shared.schemas import PackResponse, TemplateInfo, TemplateStepInfo

TREE_WIDTH = 220


class PackDetail(BaseWidget):
    """打开的资源包：左侧是缩成树状图的花园，中间按媒体分块的素材卡片，右侧是素材详情。
    双击放大只盖住中间区域，详情栏保持可见。"""
    back_requested = Signal()
    pack_selected = Signal(str)                    # 树里点了别的资源包
    run_requested = Signal()
    stop_requested = Signal()
    delete_requested = Signal()
    asset_selected = Signal(str, str)              # item id, step id
    asset_opened = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library_detail_page")
        self.pack: PackResponse | None = None
        self.template: TemplateInfo | None = None
        self.selected: tuple[str, str] | None = None
        self._cards: dict[tuple[str, str], Card] = {}
        self._layout_key: list = []
        self._tree_items: dict[str, QTreeWidgetItem] = {}
        self._tree_key: list = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_tree())
        layout.addWidget(self._divider())
        layout.addWidget(self._build_center(), 1)
        layout.addWidget(self._divider())
        self.detail = DetailPanel()
        layout.addWidget(self.detail)

    # ---- 构建 ----
    def _build_tree(self) -> QWidget:
        side = QWidget()
        side.setObjectName("library_tree_side")
        side.setFixedWidth(TREE_WIDTH)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(8, 12, 8, 12)
        layout.setSpacing(8)
        back = QPushButton(self.tr("← Library"))
        back.setObjectName("library_back_btn")
        back.clicked.connect(self.back_requested)
        layout.addWidget(back)
        self.tree = QTreeWidget()
        self.tree.setObjectName("library_tree")
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(12)
        self.tree.itemClicked.connect(self._on_tree_clicked)
        layout.addWidget(self.tree, 1)
        return side

    def _build_center(self) -> QWidget:
        self.center = QWidget()
        self.center.setObjectName("library_center")
        layout = QVBoxLayout(self.center)
        layout.setContentsMargins(24, 16, 24, 0)
        layout.setSpacing(8)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.title_label = QLabel()
        self.title_label.setObjectName("library_title")
        self.summary_label = QLabel()
        self.summary_label.setObjectName("detail_muted")
        titles.addWidget(self.title_label)
        titles.addWidget(self.summary_label)
        header.addLayout(titles, 1)
        self.run_btn = QPushButton(self.tr("Run"))
        self.run_btn.setObjectName("library_primary_btn")
        self.stop_btn = QPushButton(self.tr("Stop"))
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setObjectName("library_danger_btn")
        self.run_btn.clicked.connect(self.run_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.delete_btn.clicked.connect(self.delete_requested)
        for button in (self.run_btn, self.stop_btn, self.delete_btn):
            header.addWidget(button)
        layout.addLayout(header)

        self.error_label = QLabel()
        self.error_label.setObjectName("library_error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("library_scroll_content")
        self._sections = QVBoxLayout(content)
        self._sections.setContentsMargins(0, 8, 0, 24)
        self._sections.setSpacing(20)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        return self.center

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setObjectName("library_divider")
        line.setFixedWidth(1)
        return line

    # ---- 对外 ----
    def set_error(self, message: str):
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def set_tree(self, packs: list[PackResponse], templates: dict[str, TemplateInfo]):
        by_type: dict[str, list[PackResponse]] = {}
        for pack in packs:
            by_type.setdefault(pack.manifest.type, []).append(pack)
        key = [(t, [p.manifest.id for p in by_type.get(t, [])]) for t in ordered_categories(by_type)]
        if key != self._tree_key:
            # 只有增删资源包才重建，轮询时原地改文字，树的展开和滚动位置不会跳
            self._tree_key = key
            self.tree.clear()
            self._tree_items.clear()
            for pack_type, ids in key:
                if not ids:
                    continue
                root = QTreeWidgetItem([f"{category_text(pack_type)}  ·  {len(ids)}"])
                root.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tree.addTopLevelItem(root)
                for pack_id in ids:
                    entry = QTreeWidgetItem()
                    entry.setData(0, Qt.ItemDataRole.UserRole, pack_id)
                    root.addChild(entry)
                    self._tree_items[pack_id] = entry
                root.setExpanded(True)
        for pack in packs:
            template = templates[pack.manifest.template]
            progress = pack_progress(pack.manifest, template, running=pack.running)
            name = pack.manifest.name or template.name or pack.manifest.id
            text = f"{name}  {progress.done}/{progress.total}"
            if progress.status in ("review", "running", "error"):
                text += f"  · {status_text(progress.status)}"
            self._tree_items[pack.manifest.id].setText(0, text)
        self._select_tree_item()

    def show_pack(self, pack: PackResponse, template: TemplateInfo, busy: bool):
        """刷新中间和右侧；换了资源包时清空选中。返回值同 DetailPanel.show_asset。"""
        if self.pack is None or self.pack.manifest.id != pack.manifest.id:
            self.selected = None
            self.detail.clear()
            self._layout_key = []
            self.set_error("")
        self.pack, self.template = pack, template
        manifest = pack.manifest
        self.title_label.setText(manifest.name or template.name or manifest.id)
        progress = pack_progress(manifest, template, running=pack.running)
        summary = [category_text(manifest.type), template.name or template.id,
                   self.tr("{0}/{1} done").format(progress.done, progress.total)]
        if progress.status in ("review", "running", "error"):
            summary.append(status_text(progress.status))
        self.summary_label.setText("  ·  ".join(summary))
        self._select_tree_item()
        self._show_assets()
        return self.refresh_detail(busy)

    def refresh_detail(self, busy: bool) -> bool:
        if self.selected is None or self.pack is None:
            return False
        item_id, step_id = self.selected
        item = next((i for i in self.pack.manifest.items if i.id == item_id), None)
        step = next((s for s in self.template.step_details if s.id == step_id), None)
        if item is None or step is None:
            self.selected = None
            self.detail.clear()
            return False
        return self.detail.show_asset(self.pack, self.template, item, step, busy)

    def set_running(self, running: bool, other_running: bool):
        """other_running：别的资源包正在执行。单驻留下同时只跑一个包，这时不让再点执行。"""
        self.run_btn.setVisible(not running)
        self.run_btn.setEnabled(not other_running)
        self.stop_btn.setVisible(running)
        self.stop_btn.setEnabled(running)
        self.delete_btn.setEnabled(not running)
        # 确认剧本、重做只改本包的 manifest，别的包在跑也能改，改完等那边结束再执行
        self.detail.set_busy(running)

    def set_asset_thumbnail(self, item_id: str, step_id: str, pixmap: QPixmap):
        card = self._cards.get((item_id, step_id))
        if card is not None:
            card.set_thumbnail(pixmap)

    def show_image(self, path: str) -> ImagePreviewOverlay:
        # 宿主是中间区域而不是整页：放大时右侧详情仍然可见
        return ImagePreviewOverlay(path, self.center)

    def select_asset(self, item_id: str, step_id: str):
        previous = self._cards.get(self.selected) if self.selected else None
        if previous is not None:
            previous.set_selected(False)
        self.selected = (item_id, step_id)
        card = self._cards.get(self.selected)
        if card is not None:
            card.set_selected(True)

    # ---- 素材卡片 ----
    def _asset_groups(self) -> list[tuple[str, list[TemplateStepInfo]]]:
        steps = deliverables(self.template)
        return [(kind, [s for s in steps if media_kind(s.type) == kind]) for kind in MEDIA_ORDER
                if any(media_kind(s.type) == kind for s in steps)]

    def _show_assets(self):
        groups = self._asset_groups()
        items = self.pack.manifest.items
        layout_key = [(kind, [s.id for s in steps], [i.id for i in items]) for kind, steps in groups]
        if layout_key != self._layout_key:
            self._layout_key = layout_key
            self._rebuild_assets(groups)
        for (item_id, step_id), card in self._cards.items():
            item = next(i for i in items if i.id == item_id)
            progress = asset_progress(item, step_chain(self.template, step_id))
            card.set_progress(progress, f"{progress.done}/{progress.total}")

    def _rebuild_assets(self, groups: list[tuple[str, list[TemplateStepInfo]]]):
        self._cards.clear()
        while self._sections.count():
            widget = self._sections.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        items = self.pack.manifest.items
        for kind, steps in groups:
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(10)
            heading = QLabel(f"{media_text(kind)}  ·  {len(steps) * len(items)}")
            heading.setObjectName("library_section_title")
            section_layout.addWidget(heading)
            flow = FlowLayout(spacing=14)
            for item in items:
                for step in steps:
                    flow.addWidget(self._asset_card(kind, item.id, item_title(item), step, len(steps)))
            section_layout.addLayout(flow)
            self._sections.addWidget(section)
        self._sections.addStretch()
        if self.selected in self._cards:
            self._cards[self.selected].set_selected(True)

    def _asset_card(self, kind: str, item_id: str, title: str, step: TemplateStepInfo,
                    steps_in_group: int) -> Card:
        # 没有图片的素材（音频、剧本）占位处显示分类和媒体类型，一眼能认出是什么
        placeholder = f"{category_text(self.pack.manifest.type)}\n{media_text(kind)}"
        card = Card(ASSET_THUMB, placeholder, self.pack.manifest.type)
        # 同一块里一个条目有多个交付物时，名称带上步骤名区分
        card.set_name(f"{title} · {step.label or step.id}" if steps_in_group > 1 else title)
        card.clicked.connect(lambda: self._on_card_clicked(item_id, step.id))
        card.double_clicked.connect(lambda: self.asset_opened.emit(item_id, step.id))
        self._cards[(item_id, step.id)] = card
        return card

    def _on_card_clicked(self, item_id: str, step_id: str):
        self.select_asset(item_id, step_id)
        self.asset_selected.emit(item_id, step_id)

    # ---- 树 ----
    def _select_tree_item(self):
        if self.pack is None:
            return
        entry = self._tree_items.get(self.pack.manifest.id)
        if entry is not None:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(entry)
            self.tree.blockSignals(False)

    def _on_tree_clicked(self, entry: QTreeWidgetItem, _column: int):
        pack_id = entry.data(0, Qt.ItemDataRole.UserRole)
        if pack_id and (self.pack is None or pack_id != self.pack.manifest.id):
            self.pack_selected.emit(pack_id)
