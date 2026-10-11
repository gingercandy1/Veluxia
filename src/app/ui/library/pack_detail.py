import math
import time

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import ACCENT, FrameTicker, ProgressStrip
from src.app.ui.library.cards import ASSET_THUMB, Card
from src.app.ui.library.detail_panel import PANEL_WIDTH, DetailPanel
from src.app.ui.library.flow_layout import FlowLayout
from src.app.ui.library.layer_panel import LayerPanel
from src.app.ui.library.pack_status import (
    MEDIA_ORDER,
    asset_progress,
    card_step_id,
    card_steps,
    category_icon,
    category_text,
    deliverables,
    item_title,
    media_kind,
    media_text,
    ordered_categories,
    overlay_step,
    pack_progress,
    status_text,
    step_chain,
    step_fraction,
    step_skipped,
)
from src.app.ui.library.ring_cards import RingCardBar, RingItemPicker, SeamLinkState
from src.app.ui.library.ring_strip import RingStrip
from src.app.ui.library.segment_progress import SegmentProgressBar, SegmentState
from src.app.ui.message.image_preview import ImagePreviewOverlay
from src.shared.schemas import (
    CollectionItem,
    MergedLayer,
    PackResponse,
    RingLayerProgress,
    RingStatusResponse,
    StylePreset,
    TemplateInfo,
    TemplateStepInfo,
)

TREE_WIDTH = 220
# 接缝补丁是窄长条（304×768），预览按高度取，宽度随比例
SEAM_PREVIEW_SIZE = QSize(80, 200)
# 循环拼接用的源图步骤，与后端 ring.GENERATE_STEP 一致
RING_SOURCE_STEP = "generate"
# 手动调整后每段的宽度范围，与后端 ring_cuts.MIN_MANUAL_SEGMENT / MAX_SEGMENT 一致
MIN_SEGMENT_WIDTH = 256
MAX_SEGMENT_WIDTH = 2536


def segment_widths(cuts: list[int], total: int) -> list[int]:
    """第 i 段从第 i 个切点到下一个，最后一段绕回第一个；只有一个切点时整条环是一段。"""
    if len(cuts) == 1:
        return [total]
    return [(cuts[(i + 1) % len(cuts)] - cuts[i]) % total for i in range(len(cuts))]


# 每段的图层数，与后端 ring.LAYER_COUNT 一致；一段分层约 18 分钟（640 档 4 层 20 步实测）
LAYER_COUNT = 4
LAYER_MINUTES = 18


# 树节点上标记"正在执行"：由委托在名字后面画跳动的三个点
RUNNING_ROLE = Qt.ItemDataRole.UserRole + 1


class RunningDotsDelegate(QStyledItemDelegate):
    """执行中的资源包名字后面画跳动的三个点；名字太长时先省略，给点留出位置。"""
    GAP = 6
    _DOT_PERIOD = 1.2
    _DOT_HOP = 3.0
    DOT_RADIUS = 2.0
    DOT_GAP = 5.0
    DOTS_WIDTH = DOT_RADIUS * 2 * 3 + DOT_GAP * 2

    @classmethod
    def paint_dots(cls, painter: QPainter, left: float, center_y: float, color: QColor = ACCENT):
        """三个点依次跳起再落下，像聊天里的"正在输入"。"""
        now = time.monotonic()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for index in range(3):
            # 每个点错开一点相位；周期前 40% 跳一下，其余时间停在原位
            phase = (now / cls._DOT_PERIOD - index * 0.15) % 1.0
            hop = math.sin(math.pi * phase / 0.4) * cls._DOT_HOP if phase < 0.4 else 0.0
            x = left + cls.DOT_RADIUS + index * (cls.DOT_RADIUS * 2 + cls.DOT_GAP)
            painter.drawEllipse(QPointF(x, center_y - hop), cls.DOT_RADIUS, cls.DOT_RADIUS)
        painter.restore()


    def paint(self, painter, option, index):
        if not index.data(RUNNING_ROLE):
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QStyle()
        text_rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        opt.text = opt.fontMetrics.elidedText(
            opt.text, Qt.TextElideMode.ElideRight, int(text_rect.width() - self.DOTS_WIDTH - self.GAP))
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)
        left = text_rect.left() + opt.fontMetrics.horizontalAdvance(opt.text) + self.GAP
        self.paint_dots(painter, left, text_rect.center().y() + 1)


class PackDetail(BaseWidget):
    """打开的资源包：左侧是缩成树状图的花园，中间按媒体分块的素材卡片，右侧是素材详情。
    双击放大只盖住中间区域，详情栏保持可见。"""
    back_requested = Signal()
    pack_selected = Signal(str)                    # 树里点了别的资源包
    run_requested = Signal()
    stop_requested = Signal()
    add_items_requested = Signal()
    delete_requested = Signal()
    export_requested = Signal()
    style_sync_requested = Signal()
    source_refresh_requested = Signal()
    asset_selected = Signal(str, str)              # item id, step id
    asset_opened = Signal(str, str)
    asset_export_requested = Signal(str, str)      # item id, step id
    asset_delete_requested = Signal(str)           # item id：删的是整个条目
    ring_view_requested = Signal()
    ring_plan_requested = Signal(list)             # 自动排序：参与的条目 id（去重）
    ring_order_requested = Signal(list)            # 手动排的顺序：环上从左到右的条目 id，可重复
    ring_generate_requested = Signal()
    ring_candidates_requested = Signal(str)        # seam key：要预览它的各个候选
    ring_adopt_requested = Signal(str, str)        # seam key, candidate id
    ring_redo_requested = Signal(str, str)         # seam key, 追加的提示词
    ring_cuts_auto_requested = Signal()
    ring_cuts_confirm_requested = Signal(list)     # 切点（长图原图像素）
    ring_layers_requested = Signal(list)           # 要重做的段序号；空列表 = 做所有没分层的段
    ring_segment_requested = Signal(int)           # 要看这一段的段图和图层
    ring_layers_progressed = Signal()              # 分层中已完成的段数变了：该刷新图层了
    ring_layers_merge_requested = Signal(int, list)    # 段序号, 要合并的层编号
    ring_merged_delete_requested = Signal(int, str)    # 段序号, 要删除的合并层编号
    ring_layer_opened = Signal(str)                # 双击放大：段图或图层的相对路径

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
        self._busy = False
        self._ring_running = False
        self._ring_images_before: list = []
        self._ring_images_after: list = []
        self._ring_operation: str | None = None
        self._ring_order: list[str] = []
        self._ring_seams: dict = {}
        # 卡片条上从左到右的条目 id（同一屏可重复）。后端已有环时等于 _ring_order；
        # 还没凑够两屏时只存在这里，等第二张卡片加进来才发给后端
        self._ring_cards: list[str] = []
        # 后端存的切点；条带上的切点被拖过后和它不同，就是"还没确认"
        self._saved_cuts: list[int] = []
        self._cuts_stale = False
        self._segment_count = 0
        # 段图和每段已完成的图层（相对路径），来自 ring_status；分层进行中进度另从 manifest 算
        self._segments: list[str] = []
        self._segment_layers: list[list[str]] = []
        self._segment_merged: list[list[MergedLayer]] = []
        self._segment_images: dict[str, QImage] = {}
        self._current_segment = -1
        self._layers_done_seen = 0
        self._ring_layering = False
        # 正在生成的接缝 (key, 0~1) 和分层任务回传的最新进度；任务结束即清空
        self._seam_running: tuple[str, float] | None = None
        self._layer_live: RingLayerProgress | None = None
        self._seam_key = ""
        self._candidate_images: dict[str, dict[str, QImage]] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # 三栏宽度不固定（树的包名长短、详情的字段多少都不一样），交给用户拖分隔条调
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("library_splitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)
        self.detail = DetailPanel()
        splitter.addWidget(self._build_tree())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([TREE_WIDTH, 600, PANEL_WIDTH])
        layout.addWidget(splitter)

    # ---- 构建 ----
    def _build_tree(self) -> QWidget:
        side = QWidget()
        side.setObjectName("library_tree_side")
        side.setMinimumWidth(140)
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
        self.add_btn = QPushButton(self.tr("Add items"))
        self.add_btn.setToolTip(self.tr(
            "Add more items to this pack with the same template, style and characters. "
            "Finished assets are kept; only the new items are generated."))
        self.export_btn = QPushButton(self.tr("Export"))
        self.export_btn.setToolTip(self.tr(
            "Save the finished assets as a zip, one folder per item, "
            "without drafts and intermediate files."))
        self.delete_btn = QPushButton(self.tr("Delete"))
        self.delete_btn.setObjectName("library_danger_btn")
        self.run_btn.clicked.connect(self.run_requested)
        self.stop_btn.clicked.connect(self.stop_requested)
        self.add_btn.clicked.connect(self.add_items_requested)
        self.export_btn.clicked.connect(self.export_requested)
        self.delete_btn.clicked.connect(self.delete_requested)

        for button in (self.run_btn, self.stop_btn, self.add_btn, self.export_btn,
                       self.delete_btn):
            header.addWidget(button)
        layout.addLayout(header)

        self.view_switch = QWidget()
        switch_layout = QHBoxLayout(self.view_switch)
        switch_layout.setContentsMargins(0, 0, 0, 0)
        switch_layout.setSpacing(6)
        self.resources_view_btn = QPushButton(self.tr("Resources"))
        self.ring_view_btn = QPushButton(self.tr("Ring"))
        # 切段确认后才出现：一个场景 = 资源 / 循环 / 分层三页
        self.layer_view_btn = QPushButton(self.tr("Layers"))
        for button in (self.resources_view_btn, self.ring_view_btn, self.layer_view_btn):
            button.setCheckable(True)
            button.setObjectName("library_view_toggle")
            switch_layout.addWidget(button)
        self.layer_view_btn.hide()
        switch_layout.addStretch(1)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addButton(self.resources_view_btn, 0)
        self.view_group.addButton(self.ring_view_btn, 1)
        self.view_group.addButton(self.layer_view_btn, 2)
        self.resources_view_btn.setChecked(True)
        layout.addWidget(self.view_switch)
        self.view_switch.hide()

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

        self.center_stack = QStackedWidget()
        self.center_stack.setObjectName("library_center_stack")
        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setObjectName("library_scroll_content")
        self._sections = QVBoxLayout(content)
        self._sections.setContentsMargins(0, 8, 0, 24)
        self._sections.setSpacing(20)
        scroll.setWidget(content)
        self.center_stack.addWidget(scroll)
        self.scroll = scroll

        ring_page = QWidget()
        ring_page.setObjectName("library_ring_page")
        ring_layout = QVBoxLayout(ring_page)
        ring_layout.setContentsMargins(0, 4, 0, 12)
        ring_layout.setSpacing(10)
        self.ring_strip = RingStrip()
        ring_layout.addWidget(self.ring_strip, 1)
        # 环里的屏：卡片从左到右就是条带上的拼接顺序，可拖动换位；同一屏能加多次。
        # 每次改动立刻交给后端（纯 CPU），条带随之更新


        # 自动排序留作一键起步：按边缘相似度重排，重复的屏会合并成一张
        cards_row = QHBoxLayout()
        self.ring_plan_btn = QPushButton(self.tr("Auto arrange"))
        self.ring_plan_btn.setToolTip(self.tr(
            "Order the screens by how well their edges match. "
            "Screens added more than once are merged into one."))
        self.ring_plan_btn.clicked.connect(
            lambda: self.ring_plan_requested.emit(self.ring_plan_ids()))
        cards_row.addStretch()
        cards_row.addWidget(self.ring_plan_btn, 0, Qt.AlignmentFlag.AlignTop)
        ring_layout.addLayout(cards_row)

        self.ring_cards = RingCardBar()
        self.ring_cards.add_requested.connect(self._on_add_card)
        self.ring_cards.remove_requested.connect(self._on_remove_card)
        self.ring_cards.reordered.connect(self._set_ring_cards)
        self.ring_cards.card_selected.connect(self._on_card_selected)
        ring_layout.addWidget(self.ring_cards)

        # 拼接前/后：浮在条带右上角，点哪个切哪张图，切换时保持画面位置不动
        self.ring_before_btn = QPushButton(self.tr("Before"))
        self.ring_after_btn = QPushButton(self.tr("After"))
        self.ring_before_btn.setCheckable(True)
        self.ring_after_btn.setCheckable(True)
        self.ring_compare_group = QButtonGroup(self)
        self.ring_compare_group.setExclusive(True)
        self.ring_compare_group.addButton(self.ring_before_btn, 0)
        self.ring_compare_group.addButton(self.ring_after_btn, 1)
        self.ring_before_btn.setChecked(True)
        self.ring_before_btn.clicked.connect(lambda: self._show_ring_images(False))
        self.ring_after_btn.clicked.connect(lambda: self._show_ring_images(True))
        self.ring_compare_bar = QFrame()
        self.ring_compare_bar.setObjectName("ring_compare_bar")
        compare_layout = QHBoxLayout(self.ring_compare_bar)
        compare_layout.setContentsMargins(3, 3, 3, 3)
        compare_layout.setSpacing(2)
        for button in (self.ring_before_btn, self.ring_after_btn):
            button.setObjectName("library_view_toggle")
            compare_layout.addWidget(button)
        self.ring_strip.set_overlay(self.ring_compare_bar)
        ring_controls = QHBoxLayout()
        self.ring_generate_btn = QPushButton(self.tr("Generate seams"))
        self.ring_generate_btn.setObjectName("library_primary_btn")
        self.ring_generate_btn.clicked.connect(self.ring_generate_requested)

        self.ring_auto_cut_btn = QPushButton(self.tr("Auto cut"))
        self.ring_auto_cut_btn.clicked.connect(self.ring_cuts_auto_requested)
        self.ring_confirm_cut_btn = QPushButton(self.tr("Confirm cuts"))
        self.ring_confirm_cut_btn.clicked.connect(
            lambda: self.ring_cuts_confirm_requested.emit(self.ring_strip.cuts()))
        
        for button in (self.ring_generate_btn, self.ring_auto_cut_btn,
                       self.ring_confirm_cut_btn):
            ring_controls.addWidget(button)
        ring_layout.addLayout(ring_controls)

        # 切段：合成后的长环太长，分层前要切成一段段；红色切线可以在条带上拖
        cut_row = QHBoxLayout()
        self.ring_cut_info = QLabel()
        self.ring_cut_info.setObjectName("detail_muted")
        self.ring_cut_info.setWordWrap(True)

        cut_row.addWidget(self.ring_cut_info, 1)
        ring_layout.addLayout(cut_row)
        self.ring_strip.cuts_edited.connect(self._on_ring_cuts_edited)
        self.ring_strip.setToolTip(self._cut_hint())

        # 分层：确认切段后每段交给 Qwen-Layered 分 4 层；这里逐段看结果、不满意单段重做
        self.layer_panel = QWidget()
        self.layer_panel.setObjectName("ring_layer_panel")
        layer_layout = QVBoxLayout(self.layer_panel)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        layer_layout.setSpacing(6)
        layer_head = QHBoxLayout()
        self.layer_info = QLabel()
        self.layer_info.setObjectName("detail_muted")
        self.layer_info.setWordWrap(True)
        self.layer_redo_btn = QPushButton(self.tr("Redo this segment"))
        self.layer_run_btn = QPushButton(self.tr("Layer segments"))
        self.layer_run_btn.setObjectName("library_primary_btn")
        self.layer_stack = LayerPanel()
        layer_head.addWidget(self.layer_info, 1)
        layer_head.addWidget(self.layer_redo_btn)
        layer_head.addWidget(self.layer_run_btn)
        layer_head.addWidget(self.layer_stack.merge_btn)
        layer_head.addWidget(self.layer_stack.delete_btn)
        layer_layout.addLayout(layer_head)
        self.layer_bar = SegmentProgressBar()
        layer_layout.addWidget(self.layer_bar)
        # 一段一个小按钮，点哪个看哪段；已分层的带 ✓
        self.layer_chip_box = QWidget()
        self.layer_chips = QButtonGroup(self)
        self.layer_chips.setExclusive(True)
        self.layer_chip_layout = FlowLayout(self.layer_chip_box, spacing=4)
        layer_layout.addWidget(self.layer_chip_box)
        layer_layout.addWidget(self.layer_stack)
        # 分层单独一页：循环页只管拼接和切段，不再挤在一起
        self.layer_page = QWidget()
        self.layer_page.setObjectName("library_layer_page")
        layer_page_layout = QVBoxLayout(self.layer_page)
        layer_page_layout.setContentsMargins(0, 4, 0, 12)
        layer_page_layout.addWidget(self.layer_panel, 1)
        self.layer_run_btn.clicked.connect(lambda: self.ring_layers_requested.emit([]))
        self.layer_redo_btn.clicked.connect(self._emit_layer_redo)
        self.layer_chips.idClicked.connect(self._on_layer_segment_changed)
        self.layer_stack.layer_opened.connect(self.ring_layer_opened)
        self.layer_stack.image_opened.connect(self.show_image)
        self.layer_stack.merge_requested.connect(
            lambda ids: self.ring_layers_merge_requested.emit(self.selected_segment(), ids))
        self.layer_stack.delete_requested.connect(
            lambda layer_id: self.ring_merged_delete_requested.emit(
                self.selected_segment(), layer_id))

        # 选中接缝后才出现：挑候选、追加提示词重做。重做的是这一个接缝，别处不动
        self.seam_panel = QWidget()
        self.seam_panel.setObjectName("ring_seam_panel")
        seam_layout = QVBoxLayout(self.seam_panel)
        seam_layout.setContentsMargins(0, 0, 0, 0)
        seam_layout.setSpacing(6)
        seam_head = QHBoxLayout()
        self.seam_title = QLabel()
        self.seam_title.setObjectName("detail_muted")
        self.seam_candidate_box = QComboBox()
        # 条目文字形如 "c1 · seed 76525359 ✓"，默认宽度会把它截断
        self.seam_candidate_box.setMinimumContentsLength(22)
        self.seam_candidate_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.seam_adopt_btn = QPushButton(self.tr("Use this one"))
        seam_head.addWidget(self.seam_title, 1)
        seam_head.addWidget(self.seam_candidate_box)
        seam_head.addWidget(self.seam_adopt_btn)
        seam_layout.addLayout(seam_head)
        seam_body = QHBoxLayout()
        self.seam_preview = QLabel()
        self.seam_preview.setObjectName("ring_seam_preview")
        self.seam_preview.setFixedSize(SEAM_PREVIEW_SIZE)
        self.seam_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.seam_prompt = QLineEdit()
        self.seam_prompt.setPlaceholderText(self.tr(
            "Add to the prompt for this seam (optional), e.g. a stream, fewer trees"))
        self.seam_redo_btn = QPushButton(self.tr("Redo this seam"))
        self.seam_redo_btn.setObjectName("library_primary_btn")
        seam_body.addWidget(self.seam_prompt, 1, Qt.AlignmentFlag.AlignVCenter)
        seam_body.addWidget(self.seam_redo_btn, 0, Qt.AlignmentFlag.AlignVCenter)
        seam_layout.addLayout(seam_body)
        self.seam_panel.hide()
        ring_layout.addWidget(self.seam_panel)
        self.ring_strip.seam_selected.connect(self._on_ring_seam_selected)
        self.seam_candidate_box.currentIndexChanged.connect(self._on_seam_candidate_changed)
        self.seam_adopt_btn.clicked.connect(self._emit_seam_adopt)
        self.seam_redo_btn.clicked.connect(self._emit_seam_redo)

        self.ring_status_label = QLabel()
        self.ring_status_label.setObjectName("detail_muted")
        self.ring_status_label.setWordWrap(True)
        ring_layout.addWidget(self.ring_status_label)
        self.center_stack.addWidget(ring_page)
        self.center_stack.addWidget(self.layer_page)
        layout.addWidget(self.center_stack, 1)
        self.resources_view_btn.clicked.connect(lambda: self._set_center_view(0))
        self.ring_view_btn.clicked.connect(self._show_ring_view)
        self.layer_view_btn.clicked.connect(lambda: self._set_center_view(2))
        return self.center

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
        pack_changed = self.pack is None or self.pack.manifest.id != pack.manifest.id
        if pack_changed:
            self.selected = None
            self.detail.clear()
            self._layout_key = []
            self.set_error("")
            self.resources_view_btn.setChecked(True)
            self.center_stack.setCurrentIndex(0)
            # 接缝状态属于上一个包，换包后全部作废
            self._ring_order, self._ring_seams, self._ring_cards = [], {}, []
            # 缩略图按条目 id 记，不同包的 id 会重名；也避免旧包的图一直占着内存
            self.ring_cards.clear_thumbs()
            self._seam_key, self._candidate_images = "", {}
            self._saved_cuts, self._cuts_stale, self._segment_count = [], False, 0
            self._segments, self._segment_layers, self._segment_images = [], [], {}
            self._segment_merged, self._current_segment = [], -1
            self._layers_done_seen = 0
            self.ring_strip.set_cuts([])
            self.ring_strip.reset_view()
            self.seam_panel.hide()
        self.pack, self.template, self._busy = pack, template, busy
        manifest = pack.manifest
        self._refresh_ring_cards()
        # 分层要跑几个小时：轮询刷新 manifest 时顺带更新"已分层 x/y 段"
        self._update_layer_controls()
        ring_available = manifest.type == "scene" and len(manifest.items) >= 2
        self.view_switch.setVisible(ring_available)
        self.ring_view_btn.setVisible(ring_available)
        if not ring_available:
            self.layer_view_btn.hide()
            self.resources_view_btn.setChecked(True)
            self.center_stack.setCurrentIndex(0)
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
        self.stop_btn.setVisible(running or self._ring_running)
        self.stop_btn.setEnabled((running or self._ring_running)
                                 and not (self._ring_running and self._busy))
        self.delete_btn.setEnabled(not running)
        # 执行器持有开始时读到的 manifest，执行中追加的条目会在它存盘时被覆盖，后端也会拒绝
        self.add_btn.setEnabled(not running)
        # 执行中改风格锁会让同一个包前后两半风格不一致，后端也会因包被占用而拒绝
        self.sync_style_btn.setEnabled(not running)
        self.refresh_source_btn.setEnabled(not running)
        # 确认剧本、重做只改本包的 manifest，别的包在跑也能改，改完等那边结束再执行
        self.detail.set_busy(running)
        self._busy = running
        for button in (self.ring_plan_btn, self.ring_generate_btn):
            button.setEnabled(not running and not self._ring_running)
        self._update_ring_cards_state()
        self._update_cut_controls()
        self._update_seam_buttons()
        self._update_card_actions()

    def set_ring(self, images_before: list, images_after: list, status_text: str,
                 status: RingStatusResponse | None = None):
        self._ring_images_before = images_before
        self._ring_images_after = images_after
        self.ring_status_label.setText(status_text)
        if status is not None:
            self._ring_order = list(status.order)
            self._ring_seams = dict(status.seams)
            if status.order:
                # 后端的环是准的；还没有环时保留卡片条上用户已经加的那一张
                self._ring_cards = list(status.order)
            self._refresh_ring_cards()
            self._saved_cuts = list(status.cuts)
            self._cuts_stale = status.cuts_stale
            self._segment_count = len(status.segments)
            self.ring_strip.set_cuts(status.cuts)
            self._set_segments(list(status.segments), [list(x) for x in status.segment_layers],
                               [list(x) for x in status.segment_merged])
        self._show_ring_images(self.ring_after_btn.isChecked())
        self._update_cut_controls()
        self._refresh_seam_panel()
        if self._seam_key and self._ring_seams.get(self._seam_key) is not None:
            # 候选可能刚变了（重做出了新的一个），预览图要跟着重取
            self.ring_candidates_requested.emit(self._seam_key)

    def ring_seam(self, key: str):
        return self._ring_seams.get(key)

    def ring_card_ids(self) -> list[str]:
        return list(self._ring_cards)

    def ring_plan_ids(self) -> list[str]:
        """自动排序用的屏：卡片条上的屏去重；条上还没有卡片时用全部可用的屏。"""
        chosen = list(dict.fromkeys(self._ring_cards))
        return chosen or [item.id for item in self.ready_items()]

    def ready_items(self) -> list[CollectionItem]:
        """generate 已完成的条目：只有它们能进环。"""
        items = self.pack.manifest.items if self.pack is not None else []
        return [item for item in items
                if (state := item.steps.get(RING_SOURCE_STEP)) is not None
                and state.status == "done"]

    def _on_card_selected(self, index: int):
        self.ring_strip.set_active_image(index)

    def _refresh_ring_cards(self):
        ready = {item.id: item for item in self.ready_items()}
        # 条目被删或重做后不再可用：从卡片里拿掉（后端那边的环由下一次状态刷新纠正）
        self._ring_cards = [item_id for item_id in self._ring_cards if item_id in ready]
        self.ring_cards.set_cards([
            (item_id, f"{item_title(ready[item_id])}\n{ready[item_id].prompt}")
            for item_id in self._ring_cards])
        # 卡片重建会清掉激活态，条带上的黄框跟着同步
        self.ring_strip.set_active_image(self.ring_cards.active(), focus=False)
        self._refresh_seam_links()
        self._update_ring_cards_state()

    def _seam_link_state(self, key: str) -> tuple[SeamLinkState, float]:
        """key（左>右）这条接缝画成什么样：正在生成的优先，其次过期，再按候选 / 采用。"""
        if self._seam_running is not None and self._seam_running[0] == key:
            return SeamLinkState.RUNNING, self._seam_running[1]
        seam = self._ring_seams.get(key)
        if seam is None or not seam.candidates:
            return SeamLinkState.NONE, 0.0
        if seam.stale:
            return SeamLinkState.STALE, 0.0
        return (SeamLinkState.ADOPTED if seam.adopted else SeamLinkState.CANDIDATE), 0.0

    def _refresh_seam_links(self):
        """按卡片条上的相邻关系给每对相邻屏（含末尾回到开头）画接缝状态。"""
        cards = self._ring_cards
        if len(cards) < 2:
            self.ring_cards.set_seam_links([])
            return
        self.ring_cards.set_seam_links([
            self._seam_link_state(f"{cards[index]}>{cards[(index + 1) % len(cards)]}")
            for index in range(len(cards))])

    def _set_ring_cards(self, order: list[str]):
        """卡片条上的改动（加、删、换位）：更新卡片，凑够两屏就立刻交给后端。"""
        self._ring_cards = list(order)
        self._refresh_ring_cards()
        if len(order) >= 2 and order != self._ring_order:
            self.ring_order_requested.emit(list(order))

    def add_ring_cards(self, item_ids: list[str]):
        """追加到环的末尾；已经在环里的屏也可以再加。"""
        if item_ids:
            self._set_ring_cards([*self._ring_cards, *item_ids])

    def _on_add_card(self):
        entries = [(item.id, item_title(item), f"{item_title(item)}\n{item.prompt}",
                    self.ring_cards.thumb(item.id)) for item in self.ready_items()]
        picker = RingItemPicker(entries, self)
        if picker.exec():
            self.add_ring_cards(picker.selected_ids())

    def _on_remove_card(self, index: int):
        if 0 <= index < len(self._ring_cards):
            self._set_ring_cards(self._ring_cards[:index] + self._ring_cards[index + 1:])

    def _on_ring_cuts_edited(self, _cuts: list):
        self._update_cut_controls()

    def _set_segments(self, segments: list[str], layers: list[list[str]],
                      merged: list[list[MergedLayer]], force: bool = False):
        # 后端的三个列表本应一一对应；少了的按"没分层"补齐，不让下标越界
        layers = layers + [[] for _ in range(len(segments) - len(layers))]
        merged = merged + [[] for _ in range(len(segments) - len(merged))]
        changed = force or (segments != self._segments or layers != self._segment_layers
                            or merged != self._segment_merged)
        self._segments, self._segment_layers, self._segment_merged = segments, layers, merged
        if not changed:
            return
        current = self._current_segment
        self._current_segment = current if 0 <= current < len(segments) else (
            0 if segments else -1)
        self._rebuild_segment_chips()
        self._show_segment()

    def _rebuild_segment_chips(self):
        for button in self.layer_chips.buttons():
            self.layer_chips.removeButton(button)
        self.layer_chip_layout.clear()
        for index, done in enumerate(self._segment_layers):
            chip = QPushButton(f"{index + 1} ✓" if done else str(index + 1))
            chip.setObjectName("ring_segment_chip")
            chip.setCheckable(True)
            chip.setChecked(index == self._current_segment)
            chip.setToolTip(self.tr("Segment {0}").format(index + 1))
            self.layer_chips.addButton(chip, index)
            self.layer_chip_layout.addWidget(chip)

    def apply_layer_status(self, status: RingStatusResponse):
        """分层进行中取回的最新状态：只更新选段按钮和图层列表，不动接缝和条带。
        重做的段沿用原文件名，缓存的图要作废，当前段的图随后重新取。"""
        self._segment_images = {}
        self._set_segments(list(status.segments), [list(x) for x in status.segment_layers],
                           [list(x) for x in status.segment_merged], force=True)

    def clear_segment_images(self):
        """作废缓存的图并重新取当前段的。不能清空段列表：那会让"分层"页签闪退，
        正看着的页被切走（合并 / 删除合并层完成后就是这样跳走的）。"""
        self._segment_images = {}
        if self._segments:
            self._show_segment()

    def set_segment_images(self, images: dict[str, QImage]):
        """页面取回的段图 / 图层，键是相对路径；只更新当前选中那段用得上的。"""
        self._segment_images.update(images)
        self.layer_stack.set_images(self._segment_images)

    def _on_layer_segment_changed(self, index: int):
        self._current_segment = index
        self._show_segment()

    def _show_segment(self):
        """当前段的段图、图层和合并层交给图层面板；还没取回的图先去取。"""
        index = self._current_segment
        if 0 <= index < len(self._segments):
            self.layer_stack.set_segment(
                index, self._segments[index], self._segment_layers[index],
                self._segment_merged[index])
            if any(path not in self._segment_images for path in self.segment_paths(index)):
                self.ring_segment_requested.emit(index)
        else:
            self.layer_stack.set_segment(-1, "", [], [])
        self.layer_stack.set_images(self._segment_images)
        self._update_layer_controls()

    def selected_segment(self) -> int:
        return self._current_segment

    def segment_paths(self, index: int) -> list[str]:
        """段图 + 原层 + 合并层：这一段要显示的所有图。"""
        if not 0 <= index < len(self._segments):
            return []
        return [self._segments[index], *self._segment_layers[index],
                *(merged.path for merged in self._segment_merged[index])]

    def _emit_layer_redo(self):
        if self._current_segment >= 0:
            self.ring_layers_requested.emit([self._current_segment])

    def _layer_done_flags(self) -> list[bool]:
        """每段是否已分好层。分层要跑几个小时，直接从轮询刷新的 manifest 里算，
        不必每次都重取图。"""
        ring = self.pack.manifest.ring if self.pack is not None else None
        total = self.ring_strip.source_width()
        if ring is None or not ring.segments or not ring.cuts or not total:
            # manifest 还没带上切段（或条带还没有图）：用上次取到的状态
            return [bool(layers) for layers in self._segment_layers]
        flags = []
        for start, width in zip(ring.cuts, segment_widths(ring.cuts, total)):
            state = ring.segment_layers.get(f"{start}+{width}")
            flags.append(state is not None and state.source_sig == ring.cuts_sig
                         and len(state.layers) == LAYER_COUNT)
        return flags

    def layers_pending(self) -> bool:
        """还有没分层的段：确认切段后据此决定要不要自动开始分层。"""
        done, total = self._layer_progress()
        return done < total

    def _layer_progress(self) -> tuple[int, int]:
        """(已完成段数, 总段数)。"""
        flags = self._layer_done_flags()
        return sum(flags), len(flags)

    def _segment_states(self) -> list[SegmentState]:
        """分层进行中以任务回传的为准（当前段、剩余段），否则只看哪些段已经有图层。"""
        live = self._layer_live
        states = []
        for index, done in enumerate(self._layer_done_flags()):
            if live is not None and index == live.index:
                states.append(SegmentState.RUNNING)
            elif (live is not None and index in live.remaining) or not done:
                states.append(SegmentState.PENDING)
            else:
                states.append(SegmentState.DONE)
        return states

    def _segment_bar_widths(self) -> list[int]:
        ring = self.pack.manifest.ring if self.pack is not None else None
        total = self.ring_strip.source_width()
        if ring is None or not ring.cuts or not total:
            return [1] * len(self._segments)
        return segment_widths(ring.cuts, total)

    def _update_layer_controls(self):
        has_segments = bool(self._segments) and not self._cuts_stale
        self.layer_view_btn.setVisible(has_segments)
        if not has_segments:
            # 正看着分层页时切点失效了：页签没了，退回循环页
            if self.center_stack.currentIndex() == 2:
                self.ring_view_btn.setChecked(True)
                self.center_stack.setCurrentIndex(1)
            return
        done, total = self._layer_progress()
        if done != self._layers_done_seen:
            self._layers_done_seen = done
            if self._ring_layering:
                # 分层要跑几个小时：每做完一段就去取新的图层，不必等整个任务结束
                self.ring_layers_progressed.emit()
        idle = not self._busy and not self._ring_running and self._ring_operation is None
        parts = [self.tr("Layered {0}/{1} segments").format(done, total)]
        if done < total:
            parts.append(self.tr("about {0} min left").format((total - done) * LAYER_MINUTES))
        if self._ring_layering:
            parts.append(self.tr("layering… you can stop and continue later"))
            if self._layer_live is not None:
                stage = (self.tr("encoding") if self._layer_live.stage == "encode"
                         else self.tr("denoising"))
                parts.append(self.tr("segment {0}: {1}").format(self._layer_live.index + 1, stage))
        self.layer_info.setText("  ·  ".join(parts))
        self.layer_bar.set_segments(self._segment_bar_widths())
        self.layer_bar.set_states(self._segment_states())
        self.layer_run_btn.setEnabled(idle and done < total)
        self.layer_redo_btn.setEnabled(idle and self._current_segment >= 0)
        self.layer_stack.set_idle(idle)

    def _cut_hint(self) -> str:
        return self.tr("drag the red lines to adjust · Shift+click adds one · "
                       "Shift+right-click removes one · then confirm")

    def _update_cut_controls(self):
        cuts = self.ring_strip.cuts()
        idle = not self._busy and not self._ring_running and self._ring_operation is None
        self.ring_auto_cut_btn.setEnabled(idle and len(self._ring_order) >= 2)
        if not cuts:
            self.ring_cut_info.setText(self.tr(
                "Before layering, cut the ring into segments: render first, then Auto cut."))
            self.ring_confirm_cut_btn.setEnabled(False)
            self._update_layer_controls()
            return
        widths = segment_widths(cuts, self.ring_strip.source_width())
        parts = [self.tr("{0} segments · {1}–{2} px wide").format(
            len(widths), min(widths), max(widths))]
        bad = next(((index, width) for index, width in enumerate(widths, start=1)
                    if not MIN_SEGMENT_WIDTH <= width <= MAX_SEGMENT_WIDTH), None)
        edited = cuts != self._saved_cuts
        if bad is not None:
            parts.append(self.tr("segment {0} is {1} px, keep each within {2}–{3} px").format(
                bad[0], bad[1], MIN_SEGMENT_WIDTH, MAX_SEGMENT_WIDTH))
        elif self._cuts_stale and not edited:
            parts.append(self.tr("screens changed since cutting, auto cut again"))
        elif edited or not self._segment_count:
            parts.append(self._cut_hint())
        else:
            parts.append(self.tr("{0} segment images saved").format(self._segment_count))
        self.ring_cut_info.setText("  ·  ".join(parts))
        self.ring_confirm_cut_btn.setEnabled(
            idle and bad is None
            and (edited or self._cuts_stale or not self._segment_count))
        self._update_layer_controls()

    def _update_ring_cards_state(self):
        idle = not self._busy and not self._ring_running and self._ring_operation is None
        ready = len(self.ready_items())
        self.ring_plan_btn.setEnabled(idle and ready >= 2)
        # 环里只剩两屏时不能再删：后端要求至少两屏
        self.ring_cards.set_editable(
            idle, can_add=ready > 0, removable=len(self._ring_cards) != 2)

    def set_ring_candidate_images(self, key: str, images: dict[str, QImage]):
        self._candidate_images[key] = images
        if key == self._seam_key:
            self._update_seam_preview()

    def set_ring_progress(self, key: str, fraction: float):
        """生成接缝时，key 这条接缝的连接器按进度填充；同一对相邻屏出现几次就都画。"""
        self._seam_running = (key, fraction)
        self._refresh_seam_links()

    def set_layer_progress(self, progress: RingLayerProgress):
        """分层任务回传的进度：给每段上色，并在说明里写当前段和阶段。"""
        self._layer_live = progress
        self._update_layer_controls()

    def set_ring_running(self, running: bool, layering: bool = False):
        """layering：这次长任务是分层（不是生成接缝），状态文字和进度显示不同。"""
        if not running:
            self._seam_running = None
            self._layer_live = None
            self._refresh_seam_links()
        self._ring_running = running
        self._ring_layering = running and layering
        for button in (self.ring_plan_btn, self.ring_generate_btn):
            button.setEnabled(not running and not self._busy)
        if running:
            self.ring_status_label.setText(
                self.tr("Layering segments…") if layering else self.tr("Generating seams…"))
        self.stop_btn.setVisible(self._busy or running)
        self.stop_btn.setEnabled(self._busy or running)
        self._update_ring_cards_state()
        self._update_cut_controls()
        self._update_seam_buttons()

    def set_ring_operation(self, operation: str | None):
        self._ring_operation = operation
        active = operation is not None
        for button in (self.ring_plan_btn, self.ring_generate_btn):
            button.setEnabled(not active and not self._busy)
        if operation == "plan":
            self.ring_status_label.setText(self.tr("Planning order…"))
        elif operation == "order":
            self.ring_status_label.setText(self.tr("Updating the ring order…"))
        elif operation == "adopt":
            self.ring_status_label.setText(self.tr("Applying the chosen seam…"))
        elif operation == "cuts":
            # 首次要下载深度模型（约 100MB），会比平时久
            self.ring_status_label.setText(self.tr("Finding cut points…"))
        elif operation == "confirm":
            self.ring_status_label.setText(self.tr("Saving segments…"))
        elif operation == "merge":
            self.ring_status_label.setText(self.tr("Merging layers…"))
        elif operation == "delete_merged":
            self.ring_status_label.setText(self.tr("Deleting the merged layer…"))
        self._update_ring_cards_state()
        self._update_cut_controls()
        self._update_seam_buttons()

    def _on_ring_seam_selected(self, index: int):
        if not self._ring_order:
            return
        left = self._ring_order[index]
        right = self._ring_order[(index + 1) % len(self._ring_order)]
        self._seam_key = f"{left}>{right}"
        self.ring_candidates_requested.emit(self._seam_key)
        self._refresh_seam_panel()

    def _refresh_seam_panel(self):
        seam = self._ring_seams.get(self._seam_key)
        self.seam_panel.setVisible(seam is not None)
        if seam is None:
            self._seam_key = ""
            self._update_seam_buttons()
            return
        left, right = self._seam_key.split(">", 1)
        title = self.tr("Seam {0} → {1}").format(left, right)
        if seam.stale:
            title += " · " + self.tr("source image changed, redo this seam")
        elif not seam.candidates:
            title += " · " + self.tr("no candidate yet")
        self.seam_title.setText(title)
        self.seam_candidate_box.blockSignals(True)
        self.seam_candidate_box.clear()
        for candidate in seam.candidates:
            label = self.tr("{0} · seed {1}").format(candidate.id, candidate.seed)
            if candidate.id == seam.adopted:
                label += " ✓"
            self.seam_candidate_box.addItem(label, candidate.id)
        adopted_row = self.seam_candidate_box.findData(seam.adopted)
        self.seam_candidate_box.setCurrentIndex(max(adopted_row, 0))
        self.seam_candidate_box.blockSignals(False)
        self._update_seam_preview()
        self._update_seam_buttons()

    def _on_seam_candidate_changed(self, _index: int):
        self._update_seam_preview()
        self._update_seam_buttons()

    def _update_seam_preview(self):
        candidate_id = self.seam_candidate_box.currentData()
        image = self._candidate_images.get(self._seam_key, {}).get(candidate_id)
        if image is None or image.isNull():
            self.seam_preview.clear()
            return
        self.seam_preview.setPixmap(QPixmap.fromImage(image).scaled(
            SEAM_PREVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _update_seam_buttons(self):
        seam = self._ring_seams.get(self._seam_key)
        idle = not self._busy and not self._ring_running and self._ring_operation is None
        chosen = self.seam_candidate_box.currentData()
        self.seam_adopt_btn.setEnabled(
            idle and seam is not None and chosen is not None and chosen != seam.adopted)
        self.seam_redo_btn.setEnabled(idle and seam is not None)
        self.seam_candidate_box.setEnabled(seam is not None and bool(seam.candidates))

    def _emit_seam_adopt(self):
        candidate_id = self.seam_candidate_box.currentData()
        if self._seam_key and candidate_id:
            self.ring_adopt_requested.emit(self._seam_key, candidate_id)

    def _emit_seam_redo(self):
        if self._seam_key:
            self.ring_redo_requested.emit(self._seam_key, self.seam_prompt.text().strip())

    def _set_center_view(self, index: int):
        self.center_stack.setCurrentIndex(index)

    def _show_ring_view(self):
        self.center_stack.setCurrentIndex(1)
        self.ring_view_requested.emit()

    def _show_ring_images(self, after: bool):
        images = self._ring_images_after if after else self._ring_images_before
        self.ring_strip.set_images(images)

    def item_asset_count(self, item_id: str) -> int:
        """一个条目在卡片墙上有几张卡片：删条目时会一起消失的素材数。"""
        return sum(1 for card_item, _ in self._cards if card_item == item_id)

    def asset_card(self, item_id: str, step_id: str) -> Card | None:
        """step_id 可以是附加视图（如分层）的步骤：它和原图共用一张卡片。"""
        return self._cards.get((item_id, card_step_id(self.template, step_id)))

    def set_asset_thumbnail(self, item_id: str, step_id: str, pixmap: QPixmap):
        card = self.asset_card(item_id, step_id)
        if card is not None:
            card.set_thumbnail(pixmap, layered=card_step_id(self.template, step_id) != step_id)

    def set_ring_thumbnail(self, item_id: str, pixmap: QPixmap):
        self.ring_cards.set_thumb(item_id, pixmap)

    def show_image(self, path: str) -> ImagePreviewOverlay:
        # 宿主是中间区域而不是整页：放大时右侧详情仍然可见
        return ImagePreviewOverlay(path, self.center)

    def select_asset(self, item_id: str, step_id: str):
        previous = self.asset_card(*self.selected) if self.selected else None
        if previous is not None:
            previous.set_selected(False)
        self.selected = (item_id, step_id)
        card = self.asset_card(item_id, step_id)
        if card is not None:
            card.set_selected(True)

    def _shown_step(self, item_id: str, base_id: str) -> str:
        """卡片当前展示的步骤：切到分层视图时是分层步骤，否则是原图步骤。"""
        card = self._cards[(item_id, base_id)]
        overlay = overlay_step(self.template, base_id)
        return overlay.id if overlay is not None and card.is_layered() else base_id

    # ---- 素材卡片 ----
    def _asset_groups(self) -> list[tuple[str, list[tuple[CollectionItem, TemplateStepInfo]]]]:
        """按媒体分块的 (条目, 交付步骤)；跳过的步骤（如"不分层"条目的图层）不出卡片。"""
        steps = card_steps(self.template)
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
        for (item_id, base_id), card in self._cards.items():
            overlay = overlay_step(self.template, base_id)
            if overlay is not None:
                item = next(i for i in items if i.id == item_id)
                layers = item.steps.get(overlay.id)
                card.set_layers_available(
                    layers is not None and layers.status == "done" and bool(layers.outputs))
                # 重建卡片后恢复切换：详情栏选中的是分层时，卡片也要停在分层视图
                if self.selected == (item_id, overlay.id) and not card.layer_btn.isHidden():
                    card.layer_btn.setChecked(True)
        self._update_card_progress()
        self._update_card_actions()

    def _update_card_progress(self):
        items = self.pack.manifest.items
        for (item_id, base_id), card in self._cards.items():
            item = next(i for i in items if i.id == item_id)
            progress = asset_progress(
                item, step_chain(self.template, self._shown_step(item_id, base_id)))
            # manifest 里残留的 running 不算：包真的在跑时才让卡片动起来
            card.set_progress(progress, f"{progress.done}/{progress.total}",
                              active=self.pack.running and progress.status == "running")

    def _update_card_actions(self):
        """导出要有已完成的产物；删除按钮的限制和后端一致，前端能判断的就直接置灰并说明原因。"""
        if self.pack is None:
            return
        items = {item.id: item for item in self.pack.manifest.items}
        is_last = len(items) <= 1
        for (item_id, base_id), card in self._cards.items():
            state = items[item_id].steps.get(self._shown_step(item_id, base_id))
            card.export_btn.setEnabled(state is not None and state.status == "done"
                                       and bool(state.outputs))
            card.export_btn.setToolTip(
                self.tr("Export") if card.export_btn.isEnabled()
                else self.tr("Nothing to export yet"))
            card.delete_btn.setEnabled(not self._busy and not is_last)
            if self._busy:
                tip = self.tr("The pack is running; stop it before deleting")
            elif is_last:
                tip = self.tr("A pack needs at least one item; delete the whole pack instead")
            else:
                tip = self.tr("Delete this item and all its assets")
            card.delete_btn.setToolTip(tip)

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
        selected_card = self.asset_card(*self.selected) if self.selected else None
        if selected_card is not None:
            selected_card.set_selected(True)

    def _asset_card(self, kind: str, item_id: str, title: str, step: TemplateStepInfo,
                    steps_in_group: int) -> Card:
        # 没有图片的素材（音频、剧本）占位处显示分类和媒体类型，一眼能认出是什么
        placeholder = f"{category_text(self.pack.manifest.type)}\n{media_text(kind)}"
        card = Card(ASSET_THUMB, placeholder, self.pack.manifest.type)
        # 同一块里一个条目有多个交付物时，名称带上步骤名区分
        card.set_name(f"{title} · {step.label or step.id}" if steps_in_group > 1 else title)
        card.clicked.connect(
            lambda: self._on_card_clicked(item_id, self._shown_step(item_id, step.id)))
        card.double_clicked.connect(
            lambda: self.asset_opened.emit(item_id, self._shown_step(item_id, step.id)))
        card.add_actions()
        card.export_requested.connect(lambda: self.asset_export_requested.emit(
            item_id, self._shown_step(item_id, step.id)))
        card.delete_requested.connect(lambda: self.asset_delete_requested.emit(item_id))
        if overlay_step(self.template, step.id) is not None:
            card.add_layer_toggle()
            card.layers_toggled.connect(lambda _on: self._on_layers_toggled(item_id, step.id))
        self._cards[(item_id, step.id)] = card
        return card

    def _on_card_clicked(self, item_id: str, step_id: str):
        self.select_asset(item_id, step_id)
        self.asset_selected.emit(item_id, step_id)

    def _on_layers_toggled(self, item_id: str, base_id: str):
        self._update_card_progress()
        self._update_card_actions()
        # 这张卡片正被选中：详情栏跟着换成当前展示的那一版
        if self.selected is not None and self.asset_card(*self.selected) is self._cards[
                (item_id, base_id)]:
            shown = self._shown_step(item_id, base_id)
            if self.selected[1] != shown:
                self._on_card_clicked(item_id, shown)

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
