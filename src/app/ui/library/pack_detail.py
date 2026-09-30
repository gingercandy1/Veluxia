from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import DOTS_WIDTH, FrameTicker, ProgressStrip, paint_dots
from src.app.ui.library.cards import ASSET_THUMB, Card
from src.app.ui.library.detail_panel import DetailPanel
from src.app.ui.library.flow_layout import FlowLayout
from src.app.ui.library.pack_status import (
    MEDIA_ORDER,
    asset_progress,
    category_icon,
    category_text,
    deliverables,
    item_title,
    media_kind,
    media_text,
    ordered_categories,
    pack_progress,
    status_text,
    step_chain,
    step_fraction,
    step_skipped,
)
from src.app.ui.message.image_preview import ImagePreviewOverlay
from src.shared.schemas import (
    CollectionItem,
    PackResponse,
    StylePreset,
    TemplateInfo,
    TemplateStepInfo,
)

TREE_WIDTH = 220
# 树节点上标记"正在执行"：由委托在名字后面画跳动的三个点
RUNNING_ROLE = Qt.ItemDataRole.UserRole + 1


class RunningDotsDelegate(QStyledItemDelegate):
    """执行中的资源包名字后面画跳动的三个点；名字太长时先省略，给点留出位置。"""
    GAP = 6

    def paint(self, painter, option, index):
        if not index.data(RUNNING_ROLE):
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QStyle()
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        opt.text = opt.fontMetrics.elidedText(
            opt.text, Qt.TextElideMode.ElideRight, int(text_rect.width() - DOTS_WIDTH - self.GAP))
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        left = text_rect.left() + opt.fontMetrics.horizontalAdvance(opt.text) + self.GAP
        paint_dots(painter, left, text_rect.center().y() + 1)


class PackDetail(BaseWidget):
    """打开的资源包：左侧是缩成树状图的花园，中间按媒体分块的素材卡片，右侧是素材详情。
    双击放大只盖住中间区域，详情栏保持可见。"""
    back_requested = Signal()
    pack_selected = Signal(str)                    # 树里点了别的资源包
    run_requested = Signal()
    stop_requested = Signal()
    delete_requested = Signal()
    export_requested = Signal()
    style_sync_requested = Signal()
    source_refresh_requested = Signal()
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
        self._styles: list[StylePreset] = []
        self._has_output = False
        self._exporting = False

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
        back = QPushButton(QIcon(":/svg/back.svg"), self.tr("Library"))
        back.setObjectName("library_back_btn")
        back.setIconSize(QSize(16, 16))
        back.setCursor(Qt.CursorShape.PointingHandCursor)
        back.clicked.connect(self.back_requested)
        layout.addWidget(back)
        self.tree = QTreeWidget()
        self.tree.setObjectName("library_tree")
        self.tree.setHeaderHidden(True)
        # 分类始终展开，不需要展开箭头；子项缩进到分类图标之后，和分类名对齐
        self.tree.setRootIsDecorated(False)
        self.tree.setItemsExpandable(False)
        self.tree.setIndentation(22)
        self.tree.setIconSize(QSize(16, 16))
        self.tree.setItemDelegate(RunningDotsDelegate(self.tree))
        # 只有树里有执行中的包时才逐帧刷新
        self._tree_ticker = FrameTicker(self.tree.viewport())
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
        # 整包进度：按步骤完成比例，执行中带流光，停在某一步较久时也能看出没卡死
        self.progress_strip = ProgressStrip(height=4)
        self.progress_strip.setMaximumWidth(360)
        titles.addWidget(self.title_label)
        titles.addWidget(self.summary_label)
        titles.addSpacing(4)
        titles.addWidget(self.progress_strip)
        header.addLayout(titles, 1)
        self.run_btn = QPushButton(self.tr("Run"))
        self.run_btn.setObjectName("library_primary_btn")
        self.stop_btn = QPushButton(self.tr("Stop"))
        self.export_btn = QPushButton(self.tr("Export"))
        self.export_btn.setToolTip(self.tr(
            "Save the finished assets as a zip, one folder per item, "
            "without drafts and intermediate files."))
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setObjectName("library_danger_btn")
        self.run_btn.clicked.connect(self.run_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.export_btn.clicked.connect(self.export_requested)
        self.delete_btn.clicked.connect(self.delete_requested)
        for button in (self.run_btn, self.stop_btn, self.export_btn, self.delete_btn):
            header.addWidget(button)
        layout.addLayout(header)

        # 建包时选的风格预设之后被改过：提示一下，是否同步由用户决定（ADR 0006）
        self.style_notice = QWidget()
        notice = QHBoxLayout(self.style_notice)
        notice.setContentsMargins(0, 0, 0, 0)
        self.style_notice_label = QLabel()
        self.style_notice_label.setObjectName("detail_muted")
        self.style_notice_label.setWordWrap(True)
        notice.addWidget(self.style_notice_label, 1)
        self.sync_style_btn = QPushButton(self.tr("Use new style"))
        self.sync_style_btn.setToolTip(self.tr(
            "Replace this pack's style lock with the preset. "
            "Existing outputs are kept; redo the steps you want in the new style."))
        self.sync_style_btn.clicked.connect(self.style_sync_requested)
        notice.addWidget(self.sync_style_btn)
        self.style_notice.hide()
        layout.addWidget(self.style_notice)

        # 来源角色的立绘重做过：已做的动作 / 视角还是旧立绘的样子，是否重做由用户决定（ADR 0006）
        self.source_notice = QWidget()
        notice = QHBoxLayout(self.source_notice)
        notice.setContentsMargins(0, 0, 0, 0)
        self.source_notice_label = QLabel(self.tr(
            "The source character's portrait has changed since these assets were made."))
        self.source_notice_label.setObjectName("detail_muted")
        self.source_notice_label.setWordWrap(True)
        notice.addWidget(self.source_notice_label, 1)
        self.refresh_source_btn = QPushButton(self.tr("Redo with new portrait"))
        self.refresh_source_btn.setToolTip(self.tr(
            "Redo the steps made from the old portrait, and everything after them."))
        self.refresh_source_btn.clicked.connect(self.source_refresh_requested)
        notice.addWidget(self.refresh_source_btn)
        self.source_notice.hide()
        layout.addWidget(self.source_notice)

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
                root = QTreeWidgetItem([category_text(pack_type)])
                root.setIcon(0, QIcon(category_icon(pack_type)))
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
            # 树只管导航，进度数字在中间标题下已有；只保留需要关注的状态，执行中改用跳动的点
            text = name
            if progress.status in ("review", "error"):
                text += f"  · {status_text(progress.status)}"
            entry = self._tree_items[pack.manifest.id]
            entry.setText(0, text)
            entry.setData(0, RUNNING_ROLE, pack.running)
            entry.setToolTip(0, f"{name}  · {status_text(progress.status)}")
        self._tree_ticker.set_running(any(pack.running for pack in packs))
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
        self.progress_strip.set_progress(step_fraction(manifest, template), progress.status,
                                         active=pack.running)
        self._update_style_notice()
        self.source_notice.setVisible(pack.source_changed)
        self._has_output = any(
            (state := item.steps.get(step.id)) is not None and state.status == "done"
            for item in manifest.items for step in deliverables(template))
        self._update_export_btn()
        self._select_tree_item()
        self._show_assets()
        return self.refresh_detail(busy)

    def set_exporting(self, exporting: bool):
        self._exporting = exporting
        self._update_export_btn()

    def _update_export_btn(self):
        self.export_btn.setEnabled(self._has_output and not self._exporting)
        self.export_btn.setText(self.tr("Exporting…") if self._exporting else self.tr("Export"))

    def set_styles(self, styles: list[StylePreset]):
        self._styles = styles
        self._update_style_notice()

    def _update_style_notice(self):
        manifest = self.pack.manifest if self.pack is not None else None
        preset = next((s for s in self._styles
                       if manifest is not None and s.id == manifest.style_preset), None)
        # 预设被删了就不提示：包里存的是副本，照常可用
        changed = preset is not None and preset.style() != manifest.style
        if changed:
            self.style_notice_label.setText(
                self.tr("Style preset \"{0}\" has changed since this pack was created.")
                .format(preset.name))
        self.style_notice.setVisible(changed)

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
        # 执行中改风格锁会让同一个包前后两半风格不一致，后端也会因包被占用而拒绝
        self.sync_style_btn.setEnabled(not running)
        self.refresh_source_btn.setEnabled(not running)
        # 确认剧本、重做只改本包的 manifest，别的包在跑也能改，改完等那边结束再执行
        self.detail.set_busy(running)

    def asset_card(self, item_id: str, step_id: str) -> Card | None:
        return self._cards.get((item_id, step_id))

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
    def _asset_groups(self) -> list[tuple[str, list[tuple[CollectionItem, TemplateStepInfo]]]]:
        """按媒体分块的 (条目, 交付步骤)；跳过的步骤（如"不分层"条目的图层）不出卡片。"""
        steps = deliverables(self.template)
        groups = []
        for kind in MEDIA_ORDER:
            cards = [(item, step) for item in self.pack.manifest.items
                     for step in steps if media_kind(step.type) == kind
                     and not step_skipped(item, step, self.template)]
            if cards:
                groups.append((kind, cards))
        return groups

    def _show_assets(self):
        groups = self._asset_groups()
        items = self.pack.manifest.items
        layout_key = [(kind, [(i.id, s.id) for i, s in cards]) for kind, cards in groups]
        if layout_key != self._layout_key:
            self._layout_key = layout_key
            self._rebuild_assets(groups)
        for (item_id, step_id), card in self._cards.items():
            item = next(i for i in items if i.id == item_id)
            progress = asset_progress(item, step_chain(self.template, step_id))
            # manifest 里残留的 running 不算：包真的在跑时才让卡片动起来
            card.set_progress(progress, f"{progress.done}/{progress.total}",
                              active=self.pack.running and progress.status == "running")

    def _rebuild_assets(
            self, groups: list[tuple[str, list[tuple[CollectionItem, TemplateStepInfo]]]]):
        self._cards.clear()
        while self._sections.count():
            widget = self._sections.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        for kind, cards in groups:
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(10)
            heading = QLabel(f"{media_text(kind)}  ·  {len(cards)}")
            heading.setObjectName("library_section_title")
            section_layout.addWidget(heading)
            flow = FlowLayout(spacing=14)
            steps_in_group = len({step.id for _, step in cards})
            for item, step in cards:
                flow.addWidget(self._asset_card(kind, item.id, item_title(item), step,
                                                steps_in_group))
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
