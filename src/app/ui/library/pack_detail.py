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
    QMessageBox,
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
from src.app.ui.library.layer_window import TOOLS, LayerNode, LayerWindow
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
from src.app.ui.library.ring_cards import RingItemPicker
from src.app.ui.library.ring_math import segment_widths
from src.app.ui.library.ring_strip import RingStrip
from src.app.ui.library.segment_canvas import SegmentCanvas
from src.app.ui.message.image_preview import ImagePreviewOverlay
from src.shared.layer_stack import stack_key
from src.shared.schemas import (
    CollectionItem,
    MergedLayer,
    PackResponse,
    RingLayerProgress,
    RingStatusResponse,
    StylePreset,
    TemplateInfo,
    TemplateStepInfo,
    VersionHistory,
)

TREE_WIDTH = 220
# 接缝补丁是窄长条（304×768），预览按高度取，宽度随比例
SEAM_PREVIEW_SIZE = QSize(80, 200)
# 循环拼接用的源图步骤，与后端 ring.GENERATE_STEP 一致
RING_SOURCE_STEP = "generate"


# 每段的图层数，与后端 ring.LAYER_COUNT 一致
LAYER_COUNT = 4
# 节点键的前缀，格式见 docs/tasks/ring-canvas/plan.md 的"目标键"
SCREEN_PREFIX = "screen:"
SLOT_SEPARATOR = "@"
SEGMENT_PREFIX = "segment:"
LAYER_PREFIX = "layer:"
# 工具按钮 -> 它的运行状态；"edit" 不对应任何工具，只让四个工具都禁用
TOOL_OF_ACTION = {"generate": "seams", "redo": "seams", "extend": "extend",
                  "layers": "layer", "cuts": "segment"}


def screen_key(item_id: str) -> str:
    """屏的目标键：外扩、改图、版本都按屏记（同一屏在环上出现几次都共用）。"""
    return f"{SCREEN_PREFIX}{item_id}"


def slot_node_key(item_id: str, index: int) -> str:
    """环上第 index 个位置的树节点键。同一屏可以出现多次，节点键要带上位置才唯一。"""
    return f"{SCREEN_PREFIX}{item_id}{SLOT_SEPARATOR}{index}"


def slot_of(key: str) -> tuple[str, int] | None:
    """位置节点键 -> (条目 id, 位置)；不是位置节点时为 None。按最后一个分隔符拆，id 里有 @ 也不怕。"""
    if not key.startswith(SCREEN_PREFIX) or SLOT_SEPARATOR not in key:
        return None
    item_id, index = key.removeprefix(SCREEN_PREFIX).rsplit(SLOT_SEPARATOR, 1)
    return (item_id, int(index)) if index.isdigit() else None


def node_target(key: str) -> str:
    """树节点键 -> 后端的目标键：位置节点换成它的屏，段 / 层节点原样。"""
    slot = slot_of(key)
    return screen_key(slot[0]) if slot is not None else key


def segment_node_key(start: int, width: int) -> str:
    return f"{SEGMENT_PREFIX}{start}+{width}"


def layer_node_key(segment_key: str, layer_id: str) -> str:
    return f"{LAYER_PREFIX}{segment_key.removeprefix(SEGMENT_PREFIX)}:{layer_id}"


def seam_key(left_id: str, right_id: str) -> str:
    """与后端 ring._seam_key 一致。"""
    return f"{left_id}>{right_id}"


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
    ring_extend_requested = Signal(list)           # 要外扩的屏（条目 id）；空列表 = 全部屏
    ring_generate_requested = Signal(list)         # 要追加候选的接缝 key；空列表 = 只补缺失和过期的
    ring_candidates_requested = Signal(str)        # seam key：要预览它的各个候选
    ring_adopt_requested = Signal(str, str)        # seam key, candidate id
    ring_redo_requested = Signal(str, str)         # seam key, 追加的提示词
    ring_cuts_auto_requested = Signal()
    ring_layers_requested = Signal(list)           # 要分层的段序号（已选中的段）
    ring_images_requested = Signal(list)           # 画布和树还缺的图（相对路径）
    ring_layers_progressed = Signal()              # 分层中已完成的段数变了：该刷新图层了
    ring_layers_merge_requested = Signal(int, list)    # 段序号, 要合并的层编号
    ring_merged_delete_requested = Signal(int, str)    # 段序号, 要删除的合并层编号
    ring_edit_requested = Signal(str, str)         # 目标键, 文字指令
    ring_version_requested = Signal(str, int)      # 目标键, 版本序号（-1 = 原图）
    ring_layer_opened = Signal(str)                # 双击放大：屏、段图或图层的相对路径

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
        # 图层窗口里从上到下的屏（同一屏可重复），也就是条带上从左到右的拼接顺序。
        # 后端已有环时等于 _ring_order；还没凑够两屏时只存在这里，凑够了才发给后端
        self._ring_slots: list[str] = []
        # 各屏的缩略图：环还没建起来（条带没有图）时树节点和添加对话框用它
        self._ring_thumbs: dict[str, QPixmap] = {}
        self._cuts: list[int] = []
        self._cuts_stale = False
        # 段图和每段已完成的图层（相对路径），来自 ring_status；分层进行中进度另从 manifest 算
        self._segments: list[str] = []
        self._segment_layers: list[list[str]] = []
        self._segment_merged: list[list[MergedLayer]] = []
        # 段 / 层 / 屏的图，键是相对路径；画布和树的缩略图都从这里取
        self._segment_images: dict[str, QImage] = {}
        self._requested: set[str] = set()      # 已请求、还没回来的图，别重复请求
        self._refetch = False
        self._screen_files: list[str] = []
        self._versions: dict[str, VersionHistory] = {}
        self._layers_done_seen = 0
        # 正在跑的是哪个长任务（None = 没有）；"edit" 不对应工具按钮但同样占着模型
        self._ring_action: str | None = None
        # 分层任务回传的最新进度；任务结束即清空
        self._layer_live: RingLayerProgress | None = None
        # 外扩 / 改图正在做的目标（屏按 screen:<id>，同一屏的几个位置一起显示）-> 0~1，
        # 显示在树节点的 badge 上
        self._node_progress: dict[str, float] = {}
        self._syncing_selection = False
        self._canvas_anchor = -1
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
        for button in (self.resources_view_btn, self.ring_view_btn):
            button.setCheckable(True)
            button.setObjectName("library_view_toggle")
            switch_layout.addWidget(button)
        switch_layout.addStretch(1)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_group.addButton(self.resources_view_btn, 0)
        self.view_group.addButton(self.ring_view_btn, 1)
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

        # 循环画布页：左边画布（整条环 / 分段后的一排段）+ 卡片条，右边图层窗口（节点树 + 工具按钮）
        ring_page = self.ring_page = QWidget()
        ring_page.setObjectName("library_ring_page")
        page_layout = QHBoxLayout(ring_page)
        page_layout.setContentsMargins(0, 4, 0, 12)
        page_layout.setSpacing(12)
        canvas_side = QWidget()
        ring_layout = QVBoxLayout(canvas_side)
        ring_layout.setContentsMargins(0, 0, 0, 0)
        ring_layout.setSpacing(10)
        self.ring_strip = RingStrip()
        self.segment_canvas = SegmentCanvas()
        # 未分段看整条环；已分段看一排段，可切回整条环
        self.canvas_stack = QStackedWidget()
        self.canvas_stack.setObjectName("ring_canvas_stack")
        self.canvas_stack.addWidget(self.ring_strip)
        self.canvas_stack.addWidget(self.segment_canvas)
        ring_layout.addWidget(self.canvas_stack, 1)
        # 环里的屏在右侧图层窗口里管理：从上到下就是条带上的拼接顺序，可拖动换位、
        # 同一屏能加多次。每次改动立刻交给后端（纯 CPU），条带随之更新

        # 自动排序留作一键起步：按边缘相似度重排，重复的屏会合并成一张
        cards_row = QHBoxLayout()
        self.ring_whole_btn = QPushButton(self.tr("View whole ring"))
        self.ring_whole_btn.setCheckable(True)
        self.ring_whole_btn.setToolTip(self.tr("Switch between the segments and the whole ring"))
        self.ring_whole_btn.toggled.connect(lambda _checked: self._update_canvas_view())
        self.ring_whole_btn.hide()
        self.ring_plan_btn = QPushButton(self.tr("Auto arrange"))
        self.ring_plan_btn.setToolTip(self.tr(
            "Order the screens by how well their edges match. "
            "Screens added more than once are merged into one."))
        self.ring_plan_btn.clicked.connect(
            lambda: self.ring_plan_requested.emit(self.ring_plan_ids()))
        cards_row.addWidget(self.ring_whole_btn, 0, Qt.AlignmentFlag.AlignTop)
        cards_row.addStretch()
        cards_row.addWidget(self.ring_plan_btn, 0, Qt.AlignmentFlag.AlignTop)
        ring_layout.addLayout(cards_row)

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
        self.ring_strip.setToolTip(self.tr(
            "Click a seam to select it · Ctrl+click selects several"))

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
        self.ring_strip.seams_selected.connect(lambda _seams: self._update_tools())
        self.segment_canvas.segment_clicked.connect(self._on_canvas_segment_clicked)
        self.seam_candidate_box.currentIndexChanged.connect(self._on_seam_candidate_changed)
        self.seam_adopt_btn.clicked.connect(self._emit_seam_adopt)
        self.seam_redo_btn.clicked.connect(self._emit_seam_redo)

        self.ring_status_label = QLabel()
        self.ring_status_label.setObjectName("detail_muted")
        self.ring_status_label.setWordWrap(True)
        ring_layout.addWidget(self.ring_status_label)

        self.layer_window = LayerWindow()
        self.layer_window.tool_clicked.connect(self._on_tool_clicked)
        self.layer_window.selection_changed.connect(self._on_nodes_selected)
        self.layer_window.visibility_changed.connect(
            lambda _keys: self._refresh_canvas_segments())
        # 位置节点的版本和改图都作用在它的屏上：节点键换成后端的目标键再发
        self.layer_window.version_requested.connect(
            lambda key, index: self.ring_version_requested.emit(node_target(key), index))
        self.layer_window.node_opened.connect(self._on_node_opened)
        self.layer_window.merge_requested.connect(self._on_merge_requested)
        self.layer_window.delete_requested.connect(self._on_delete_requested)
        self.layer_window.edit_requested.connect(
            lambda key, prompt: self.ring_edit_requested.emit(node_target(key), prompt))
        self.layer_window.add_requested.connect(self._on_add_slots)
        self.layer_window.remove_requested.connect(self._on_remove_slots)
        self.layer_window.reorder_requested.connect(self._on_reorder_slots)
        page_layout.addWidget(canvas_side, 1)
        page_layout.addWidget(self.layer_window)
        self.center_stack.addWidget(ring_page)
        layout.addWidget(self.center_stack, 1)
        self.resources_view_btn.clicked.connect(lambda: self._set_center_view(0))
        self.ring_view_btn.clicked.connect(self._show_ring_view)
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
            self._ring_order, self._ring_seams, self._ring_slots = [], {}, []
            # 缩略图按条目 id 记，不同包的 id 会重名；也避免旧包的图一直占着内存
            self._ring_thumbs = {}
            self._seam_key, self._candidate_images = "", {}
            self._cuts, self._cuts_stale = [], False
            self._segments, self._segment_layers, self._segment_images = [], [], {}
            self._segment_merged, self._screen_files, self._versions = [], [], {}
            self._node_progress, self._requested, self._refetch = {}, set(), False
            self._layers_done_seen = 0
            self.ring_whole_btn.setChecked(False)
            self.ring_strip.set_cuts([])
            self.ring_strip.set_padding([])
            self.ring_strip.reset_view()
            self.segment_canvas.set_images([])
            self.layer_window.set_nodes([])
            self.seam_panel.hide()
        self.pack, self.template, self._busy = pack, template, busy
        manifest = pack.manifest
        # 条目被删或重做后不再可用：从环里拿掉（后端那边的环由下一次状态刷新纠正）。
        # 轮询会频繁走到这里，没变化就不重建树
        ready = {item.id for item in self.ready_items()}
        slots = [item_id for item_id in self._ring_slots if item_id in ready]
        if slots != self._ring_slots:
            self._ring_slots = slots
            self._refresh_nodes()
        # 分层要跑几个小时：轮询刷新 manifest 时顺带检查"已分层几段"有没有变
        self._update_layer_controls()
        self._update_tools()
        ring_available = manifest.type == "scene" and len(manifest.items) >= 2
        self.view_switch.setVisible(ring_available)
        self.ring_view_btn.setVisible(ring_available)
        if not ring_available:
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
        self._update_tools()
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
                # 后端的环是准的；还没有环时保留用户已经加的那一屏
                self._ring_slots = list(status.order)
            self._cuts = list(status.cuts)
            self._cuts_stale = status.cuts_stale
            self._screen_files = list(status.screen_files)
            self._versions = dict(status.versions)
            self.ring_strip.set_cuts(status.cuts)
            self.ring_strip.set_padding(status.screen_padding)
            self._store_segments(list(status.segments), [list(x) for x in status.segment_layers],
                                 [list(x) for x in status.segment_merged])
        self._show_ring_images(self.ring_after_btn.isChecked())
        self._refresh_nodes()
        self._refresh_seam_panel()
        if self._seam_key and self._ring_seams.get(self._seam_key) is not None:
            # 候选可能刚变了（重做出了新的一个），预览图要跟着重取
            self.ring_candidates_requested.emit(self._seam_key)

    def ring_seam(self, key: str):
        return self._ring_seams.get(key)

    def ring_slot_ids(self) -> list[str]:
        return list(self._ring_slots)

    def ring_plan_ids(self) -> list[str]:
        """自动排序用的屏：环里的屏去重；环里还没有屏时用全部可用的屏。"""
        chosen = list(dict.fromkeys(self._ring_slots))
        return chosen or [item.id for item in self.ready_items()]

    def ready_items(self) -> list[CollectionItem]:
        """generate 已完成的条目：只有它们能进环。"""
        items = self.pack.manifest.items if self.pack is not None else []
        return [item for item in items
                if (state := item.steps.get(RING_SOURCE_STEP)) is not None
                and state.status == "done"]

    def _set_ring_slots(self, order: list[str], selected: list[int] | None = None):
        """环里的屏改了（加、删、换位）：重建树，凑够两屏就立刻交给后端。
        selected：改完后要选中的位置（换位后选中跟着被拖的那一屏走）。"""
        ready = {item.id for item in self.ready_items()}
        self._ring_slots = [item_id for item_id in order if item_id in ready]
        self._refresh_nodes()
        if selected is not None:
            self.layer_window.select([slot_node_key(self._ring_slots[index], index)
                                      for index in selected if index < len(self._ring_slots)])
        if len(self._ring_slots) >= 2 and self._ring_slots != self._ring_order:
            self.ring_order_requested.emit(list(self._ring_slots))

    def add_ring_slots(self, item_ids: list[str]):
        """追加到环的末尾；已经在环里的屏也可以再加。"""
        if item_ids:
            self._set_ring_slots([*self._ring_slots, *item_ids])

    def _on_add_slots(self):
        entries = [(item.id, item_title(item), f"{item_title(item)}\n{item.prompt}",
                    self._ring_thumbs.get(item.id)) for item in self.ready_items()]
        picker = RingItemPicker(entries, self)
        if picker.exec():
            self.add_ring_slots(picker.selected_ids())

    def _selected_slots(self, keys: list[str] | None = None) -> list[int]:
        keys = self.layer_window.selected_keys() if keys is None else keys
        return sorted({slot[1] for key in keys if (slot := slot_of(key)) is not None})

    def _can_remove_slots(self, count: int) -> bool:
        """后端的环至少两屏：删完少于两屏就不让删；本来就不到两屏（还没成环）时随便删。"""
        return count > 0 and (len(self._ring_slots) - count >= 2 or len(self._ring_slots) < 2)

    def _on_remove_slots(self, keys: list[str]):
        removed = set(self._selected_slots(keys))
        if not self._can_remove_slots(len(removed)):
            return
        self._set_ring_slots(
            [item_id for index, item_id in enumerate(self._ring_slots) if index not in removed],
            selected=[])

    def _on_reorder_slots(self, keys: list[str]):
        """keys：拖动后顶层节点的新顺序（节点键里带着原来的位置）。"""
        old_positions = [slot[1] for key in keys if (slot := slot_of(key)) is not None]
        if sorted(old_positions) != list(range(len(self._ring_slots))):
            return
        new_index = {old: new for new, old in enumerate(old_positions)}
        selected = [new_index[index] for index in self._selected_slots()]
        self._set_ring_slots([self._ring_slots[old] for old in old_positions], selected=selected)

    # ---- 画布与图层树 ----
    def _store_segments(self, segments: list[str], layers: list[list[str]],
                        merged: list[list[MergedLayer]]):
        # 后端的三个列表本应一一对应；少了的按"没分层"补齐，不让下标越界
        self._segments = segments
        self._segment_layers = layers + [[] for _ in range(len(segments) - len(layers))]
        self._segment_merged = merged + [[] for _ in range(len(segments) - len(merged))]

    def _segmented(self) -> bool:
        """已分段且段图没过期：画布显示一排段，树里一段一个节点。"""
        return bool(self._segments) and not self._cuts_stale

    def _ring_total_width(self) -> int:
        """合成后整条环的宽度：切点就是在这张图上算的。"""
        return sum(image.width() for image in self._ring_images_after if not image.isNull())

    def _segment_keys(self) -> list[str]:
        """与 self._segments 一一对应的段节点键；切点或条带图还没有时是空。"""
        total = self._ring_total_width()
        if not self._cuts or not total:
            return []
        keys = [segment_node_key(start, width)
                for start, width in zip(self._cuts, segment_widths(self._cuts, total))]
        return keys[:len(self._segments)]

    def _layer_stack(self, index: int) -> list[tuple[str, str, int]]:
        """第 index 段的 (层编号, 文件, 叠放位置)，从底到顶；规则与后端合并时一致。"""
        entries = [(f"L{number}", path, number)
                   for number, path in enumerate(self._segment_layers[index], start=1)]
        entries += [(merged.id, merged.path, merged.z) for merged in self._segment_merged[index]]
        return sorted(entries, key=lambda entry: stack_key(entry[0], entry[2]))

    def _version_info(self, key: str) -> tuple[int, int]:
        history = self._versions.get(key)
        return (len(history.versions), history.current) if history is not None else (0, -1)

    def _node_badge(self, key: str, segment_index: int = -1) -> str:
        fraction = self._node_progress.get(node_target(key))
        if fraction is not None:
            return f"{int(fraction * 100)}%"
        if segment_index < 0:
            return ""
        live = self._layer_live
        if live is not None and live.index == segment_index:
            return self.tr("encoding") if live.stage == "encode" else self.tr("denoising")
        if live is not None and segment_index in live.remaining:
            return "…"
        return "✓" if self._segment_layers[segment_index] else ""

    def _build_nodes(self) -> list[LayerNode]:
        if self._segmented():
            return self._segment_nodes()
        return self._screen_nodes()

    def _screen_nodes(self) -> list[LayerNode]:
        """未分段：环上一个位置一个节点，从上到下就是拼接顺序；同一屏出现几次就有几个。
        版本按屏记，同一屏的几个节点显示同一份。"""
        items = {item.id: item for item in (self.pack.manifest.items if self.pack else [])}
        nodes = []
        for index, item_id in enumerate(self._ring_slots):
            key = slot_node_key(item_id, index)
            count, current = self._version_info(screen_key(item_id))
            item = items.get(item_id)
            title = item_title(item) if item is not None else item_id
            nodes.append(LayerNode(
                key, f"{index + 1} · {title}",
                badge=self._node_badge(key), version_count=count, current_version=current))
        return nodes

    def _segment_nodes(self) -> list[LayerNode]:
        """已分段：一段一个节点；分过层的段带子节点（含合并层，从顶到底）。"""
        nodes = []
        for index, segment in enumerate(self._segment_keys()):
            count, current = self._version_info(segment)
            children = []
            for layer_id, _path, _z in reversed(self._layer_stack(index)):
                key = layer_node_key(segment, layer_id)
                layer_count, layer_current = self._version_info(key)
                children.append(LayerNode(
                    key, layer_id, badge=self._node_badge(key), version_count=layer_count,
                    current_version=layer_current))
            nodes.append(LayerNode(
                segment, self.tr("Segment {0}").format(index + 1), children=children,
                badge=self._node_badge(segment, index), version_count=count,
                current_version=current))
        return nodes

    def _node_paths(self) -> dict[str, str]:
        """节点键 -> 它当前显示的图（相对路径）。"""
        paths: dict[str, str] = {}
        screen_files: dict[str, str] = {}
        for index, item_id in enumerate(self._ring_order):
            if index < len(self._screen_files) and self._screen_files[index]:
                screen_files.setdefault(item_id, self._screen_files[index])
        for index, item_id in enumerate(self._ring_slots):
            if item_id in screen_files:
                paths[slot_node_key(item_id, index)] = screen_files[item_id]
        for index, segment in enumerate(self._segment_keys()):
            paths[segment] = self._segments[index]
            for layer_id, path, _z in self._layer_stack(index):
                paths[layer_node_key(segment, layer_id)] = path
        return paths

    def _node_images(self) -> dict[str, QImage]:
        images: dict[str, QImage] = {}
        # 屏的图优先用条带上的（外扩、改图后的当前版本）；环还没建起来时用缩略图
        screens: dict[str, QImage] = {}
        for index, item_id in enumerate(self._ring_order):
            if index < len(self._ring_images_before):
                screens.setdefault(item_id, self._ring_images_before[index])
        for index, item_id in enumerate(self._ring_slots):
            image = screens.get(item_id)
            if image is None and item_id in self._ring_thumbs:
                image = self._ring_thumbs[item_id].toImage()
            if image is not None:
                images[slot_node_key(item_id, index)] = image
        for key, path in self._node_paths().items():
            if not key.startswith(SCREEN_PREFIX) and path in self._segment_images:
                images[key] = self._segment_images[path]
        return images

    def _refresh_nodes(self):
        """按当前状态重建树和画布；选中、折叠、眼睛由 LayerWindow 自己保留。"""
        self.layer_window.set_nodes(self._build_nodes())
        self.layer_window.set_images(self._node_images())
        self._request_missing_images()
        self._update_canvas_view()
        self._refresh_canvas_segments()
        self._update_tools()

    def _request_missing_images(self):
        if not self._segmented():
            return
        # 文件被重写过（重做、改图）时整批重取；取回之前旧图继续显示，画布不会闪空
        refetch, self._refetch = self._refetch, False
        wanted = list(dict.fromkeys(path for key, path in self._node_paths().items()
                                    if not key.startswith(SCREEN_PREFIX)))
        # 用不上的旧图（换了版本、段被重新切）不再占内存
        self._segment_images = {path: image for path, image in self._segment_images.items()
                                if path in wanted}
        missing = [path for path in wanted
                   if (refetch or path not in self._segment_images)
                   and path not in self._requested]
        if missing:
            self._requested.update(missing)
            self.ring_images_requested.emit(missing)

    def _update_canvas_view(self):
        segmented = self._segmented()
        self.ring_whole_btn.setVisible(segmented)
        if not segmented and self.ring_whole_btn.isChecked():
            self.ring_whole_btn.blockSignals(True)
            self.ring_whole_btn.setChecked(False)
            self.ring_whole_btn.blockSignals(False)
        show_canvas = segmented and not self.ring_whole_btn.isChecked()
        self.canvas_stack.setCurrentWidget(self.segment_canvas if show_canvas else self.ring_strip)

    def _refresh_canvas_segments(self):
        """画布上每段显示它可见层的合成；没分层时显示段图。"""
        if not self._segmented():
            self.segment_canvas.set_images([])
            return
        visible = set(self.layer_window.visible_keys())
        images = [self._segment_display(index, segment, visible)
                  for index, segment in enumerate(self._segment_keys())]
        self.segment_canvas.set_images(images)
        self.segment_canvas.set_selected(self._selected_segments())

    def _segment_display(self, index: int, segment: str, visible: set[str]) -> QImage:
        base = self._segment_images.get(self._segments[index])
        if base is None or base.isNull():
            return QImage()
        result = QImage(base.size(), QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        if segment not in visible:
            return result
        stack = self._layer_stack(index)
        if not stack:
            return base
        layers = [(layer_id, self._segment_images.get(path)) for layer_id, path, _z in stack]
        if all(image is None or image.isNull() for _layer_id, image in layers):
            return base
        painter = QPainter(result)
        for layer_id, image in layers:
            if (image is not None and not image.isNull()
                    and layer_node_key(segment, layer_id) in visible):
                painter.drawImage(0, 0, image)
        painter.end()
        return result

    def _selected_segments(self, keys: list[str] | None = None) -> list[int]:
        """选中的节点所在的段序号（选中层节点算它所在的段）。"""
        keys = self.layer_window.selected_keys() if keys is None else keys
        segment_keys = self._segment_keys()
        result = set()
        for key in keys:
            if key.startswith(LAYER_PREFIX):
                key = SEGMENT_PREFIX + key.removeprefix(LAYER_PREFIX).rsplit(":", 1)[0]
            if key in segment_keys:
                result.add(segment_keys.index(key))
        return sorted(result)

    def _selected_screen_ids(self) -> list[str]:
        """选中位置所在的屏，去重（同一屏选了两个位置也只外扩一次）。"""
        return list(dict.fromkeys(
            slot[0] for key in self.layer_window.selected_keys()
            if (slot := slot_of(key)) is not None))

    def _selected_seam_keys(self) -> list[str]:
        """条带上选中的接缝：第 i 个接缝是 order[i] 与 order[i + 1]（环）。
        画布不在条带上时选中的接缝不可见，当作没选。"""
        order = self._ring_order
        if len(order) < 2 or self.canvas_stack.currentWidget() is not self.ring_strip:
            return []
        return list(dict.fromkeys(
            seam_key(order[index], order[(index + 1) % len(order)])
            for index in self.ring_strip.selected_seams() if index < len(order)))

    def _on_nodes_selected(self, keys: list[str]):
        if self._segmented():
            self.segment_canvas.set_selected(self._selected_segments(keys))
        else:
            # 条带上黄框框住最后选中的那个位置；环还没按新顺序刷新回来时不框，免得框错屏
            slots = [slot for key in keys if (slot := slot_of(key)) is not None]
            item_id, index = slots[-1] if slots else ("", -1)
            if 0 <= index < len(self._ring_order) and self._ring_order[index] == item_id:
                self.ring_strip.set_active_image(index)
            else:
                self.ring_strip.set_active_image(-1, focus=False)
        self._update_tools()

    def _on_canvas_segment_clicked(self, index: int, modifiers):
        """画布上点段，规则与图层窗口一致；结果交给图层窗口的选择，再由它回头高亮画布。"""
        keys = self._segment_keys()
        if not 0 <= index < len(keys):
            return
        selected = self._selected_segments()
        if modifiers & Qt.KeyboardModifier.ShiftModifier and 0 <= self._canvas_anchor < len(keys):
            low, high = sorted((self._canvas_anchor, index))
            chosen = list(range(low, high + 1))
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            chosen = ([i for i in selected if i != index] if index in selected
                      else [*selected, index])
            self._canvas_anchor = index
        else:
            chosen = [index]
            self._canvas_anchor = index
        self.layer_window.select([keys[i] for i in chosen])

    def _on_node_opened(self, key: str):
        path = self._node_paths().get(key)
        if path:
            self.ring_layer_opened.emit(path)

    def _on_merge_requested(self, keys: list[str]):
        segments = self._selected_segments(keys)
        if len(segments) == 1:
            self.ring_layers_merge_requested.emit(
                segments[0], [key.rsplit(":", 1)[1] for key in keys])

    def _on_delete_requested(self, key: str):
        segments = self._selected_segments([key])
        if segments:
            self.ring_merged_delete_requested.emit(segments[0], key.rsplit(":", 1)[1])

    def apply_layer_status(self, status: RingStatusResponse):
        """分层进行中取回的最新状态：只更新段 / 层和树，不动接缝和条带。
        重做的段沿用原文件名，缓存的图要作废，随后重新取。"""
        self._refetch, self._requested = True, set()
        self._store_segments(list(status.segments), [list(x) for x in status.segment_layers],
                             [list(x) for x in status.segment_merged])
        self._versions = dict(status.versions)
        self._refresh_nodes()

    def clear_segment_images(self):
        """缓存的图作废；页面随后会重新取状态，set_ring 时整批重取。
        不清空图和段列表：那会让画布闪退成空白。"""
        self._refetch, self._requested = True, set()

    def set_segment_images(self, images: dict[str, QImage]):
        """页面取回的段图 / 图层，键是相对路径。"""
        self._segment_images.update(images)
        self._requested -= set(images)
        self.layer_window.set_images(self._node_images())
        self._refresh_canvas_segments()

    def ring_images(self) -> list[QImage]:
        """条带当前显示的环图（分段对话框用同一组，切点坐标才对得上）。"""
        return list(self._ring_images_after if self.ring_after_btn.isChecked()
                    else self._ring_images_before)

    def _layer_done_flags(self) -> list[bool]:
        """每段是否已分好层。分层要跑几个小时，直接从轮询刷新的 manifest 里算，
        不必每次都重取图。"""
        ring = self.pack.manifest.ring if self.pack is not None else None
        total = self._ring_total_width()
        if ring is None or not ring.segments or not ring.cuts or not total:
            # manifest 还没带上切段（或条带还没有图）：用上次取到的状态
            return [bool(layers) for layers in self._segment_layers]
        flags = []
        for start, width in zip(ring.cuts, segment_widths(ring.cuts, total)):
            state = ring.segment_layers.get(f"{start}+{width}")
            flags.append(state is not None and state.source_sig == ring.cuts_sig
                         and len(state.layers) == LAYER_COUNT)
        return flags

    def _update_layer_controls(self):
        """分层要跑几个小时：每做完一段就让页面去取新的图层，不必等整个任务结束。"""
        if not self._segmented():
            return
        done = sum(self._layer_done_flags())
        if done != self._layers_done_seen:
            self._layers_done_seen = done
            if self._ring_action == "layers":
                self.ring_layers_progressed.emit()

    # ---- 工具按钮 ----
    def _idle(self) -> bool:
        return not self._busy and not self._ring_running and self._ring_operation is None

    def _running_tool(self) -> str | None:
        if self._ring_running:
            return TOOL_OF_ACTION.get(self._ring_action or "")
        # 自动切割是秒到分钟级的单次操作，但同样可以停：按钮显示"停止分段"
        return "segment" if self._ring_operation == "cuts" else None

    def _update_tools(self):
        idle = self._idle()
        running = self._running_tool()
        has_ring = len(self._ring_order) >= 2
        for tool in TOOLS:
            if running == tool:
                self.layer_window.set_tool_state(tool, True, True)
                continue
            enabled, tip = idle and has_ring, ""
            if tool == "extend" and self._segments:
                enabled, tip = False, self.tr("Extending is only possible before segmenting")
            elif tool == "extend":
                tip = self.tr("Add 128 px above and below the selected screens (all if none)")
            elif tool == "seams":
                tip = self.tr("Add candidates for the selected seams; with none selected, "
                              "fill in the missing and stale ones")
            elif tool == "segment":
                tip = self.tr("Find cut points automatically, then confirm them")
            elif tool == "layer":
                if not self._segmented():
                    enabled, tip = False, self.tr("Segment the ring first")
                elif not self._selected_segments():
                    enabled = False
                    tip = self.tr("Select segments first (Ctrl+A selects all)")
                else:
                    tip = self.tr("Split the selected segments into layers")
            self.layer_window.set_tool_state(tool, False, enabled, tip)
        self.layer_window.set_idle(idle)
        ready = len(self.ready_items())
        self.ring_plan_btn.setEnabled(idle and ready >= 2)
        # 分段后树里是段和层，屏的增删换位只在分段前做
        self.layer_window.set_list_mode(
            not self._segmented(), editable=idle, can_add=ready > 0,
            can_remove=self._can_remove_slots(len(self._selected_slots())))

    def _on_tool_clicked(self, tool: str):
        if self._running_tool() == tool:
            self.stop_requested.emit()
            return
        if not self.layer_window.tool_buttons[tool].isEnabled():
            return
        if tool == "extend":
            self._request_extend()
        elif tool == "seams":
            self.ring_generate_requested.emit(self._selected_seam_keys())
        elif tool == "segment":
            self.ring_cuts_auto_requested.emit()
        elif tool == "layer":
            self.ring_layers_requested.emit(self._selected_segments())

    def _request_extend(self):
        """外扩会改屏的图：已有的接缝候选和分层都对不上了，先让用户确认。"""
        has_seams = any(seam.candidates or seam.has_candidates
                        for seam in self._ring_seams.values())
        has_layers = any(self._segment_layers)
        if has_seams or has_layers:
            answer = QMessageBox.question(
                self, self.tr("Extend screens"),
                self.tr("Extending changes the screens, so the existing seams and layers "
                        "will become invalid. Continue?"))
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.ring_extend_requested.emit(self._selected_screen_ids())

    def set_ring_candidate_images(self, key: str, images: dict[str, QImage]):
        self._candidate_images[key] = images
        if key == self._seam_key:
            self._update_seam_preview()

    def set_ring_progress(self, key: str, fraction: float):
        """生成接缝的进度。接缝进度条以后再放，先在状态栏写百分比。"""
        left, right = key.split(">", 1) if ">" in key else (key, "")
        self.ring_status_label.setText(
            self.tr("Generating seam {0} → {1}… {2}%").format(left, right, int(fraction * 100)))

    def set_layer_progress(self, progress: RingLayerProgress):
        """分层任务回传的进度：给每段打上状态，并在说明里写当前段和阶段。"""
        self._layer_live = progress
        self._update_layer_controls()
        self._refresh_nodes()

    def set_node_progress(self, key: str, fraction: float):
        """外扩 / 文字改图：节点 badge 显示百分比。直接改那几行，不重建整棵树。
        key 是目标键：屏在环上出现几次，几个位置节点都显示。"""
        self._node_progress[key] = fraction
        for node_key in self.layer_window.node_keys():
            row = self.layer_window.row(node_key)
            if row is not None and node_target(node_key) == key:
                row.badge.setText(f"{int(fraction * 100)}%")
                row.badge.setToolTip("")
                row.badge.setVisible(True)

    def set_ring_running(self, running: bool, action: str = ""):
        """action：这次长任务是什么（generate / redo / extend / layers / edit），
        决定哪个工具按钮变成"停止"、状态栏写什么。"""
        if not running:
            self._layer_live = None
            self._node_progress = {}
        self._ring_running = running
        self._ring_action = action if running else None
        if running:
            texts = {"layers": self.tr("Layering segments…"),
                     "extend": self.tr("Extending screens…"),
                     "edit": self.tr("Editing…")}
            self.ring_status_label.setText(
                texts.get(action, self.tr("Generating seams…")))
        self.stop_btn.setVisible(self._busy or running)
        self.stop_btn.setEnabled(self._busy or running)
        self._update_tools()
        self._update_seam_buttons()
        self._refresh_nodes()

    def set_ring_operation(self, operation: str | None):
        self._ring_operation = operation
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
        elif operation == "version":
            self.ring_status_label.setText(self.tr("Switching version…"))
        self._update_tools()
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
        self._ring_thumbs[item_id] = pixmap
        # 环还没建起来时树节点靠缩略图显示；建起来后用条带的图，不用重设
        if item_id in self._ring_slots and not self._segmented():
            self.layer_window.set_images(self._node_images())

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
