"""循环预览控件与详情页签契约，不连接后端。"""
import threading

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from src.app.ui.library.library_page import LibraryPage
from src.app.ui.library.pack_detail import PackDetail
from src.app.ui.library.ring_cards import PICKER_THUMB, RingCardBar, RingItemPicker
from src.app.ui.library.ring_strip import OVERLAY_MARGIN, ZOOM_MAX, ZOOM_MIN, RingStrip
from src import resources  # noqa: F401  注册 :/svg 图标，图层行的眼睛要用
from src.shared.schemas import (
    CollectionItem,
    ImageVersion,
    Manifest,
    MergedLayer,
    PackResponse,
    RingLayerProgress,
    RingSeamStatus,
    RingStatusResponse,
    SeamCandidate,
    StepState,
    TemplateInfo,
    TemplateStepInfo,
    VersionHistory,
)


def _scene_pack(pack_type="scene", item_count=2):
    items = [
        CollectionItem(
            id=f"screen{index}",
            prompt=f"scene {index}",
            steps={"generate": StepState(status="done", outputs=[f"screen{index}/generate.png"])},
        )
        for index in range(item_count)
    ]
    manifest = Manifest(id="scene_pack", type=pack_type, template="scene_background", items=items)
    template = TemplateInfo(
        id="scene_background", type=pack_type, name="Scene", steps=["generate"],
        step_details=[TemplateStepInfo(id="generate", type="image.generate", deliverable=True)],
    )
    return PackResponse(manifest=manifest, media_base="/media/library/scene_pack"), template


def test_ring_strip_empty_images_loop_playback_and_timer(qapp):
    strip = RingStrip()
    # 控件最小高度 180：图的高度取同值，条带按高度等比缩放后总宽才等于图宽之和
    strip.resize(300, 180)
    strip.set_images([])
    strip.render(QImage(300, 180, QImage.Format.Format_ARGB32))
    assert strip.seam_count() == 0

    images = [
        QImage(50, 180, QImage.Format.Format_RGB32),
        QImage(75, 180, QImage.Format.Format_RGB32),
    ]
    images[0].fill(QColor("red"))
    images[1].fill(QColor("blue"))
    strip.set_images(images)
    assert strip.seam_count() == 2

    def frame(offset):
        strip.set_offset(offset)
        target = QImage(strip.size(), QImage.Format.Format_ARGB32)
        target.fill(Qt.GlobalColor.transparent)
        strip.render(target)
        return target.convertToFormat(QImage.Format.Format_RGBA8888).bits().tobytes()

    total_width = sum(image.width() for image in images)
    assert frame(0) == frame(total_width)
    # 静止画布：不再自动滚动，偏移只随用户操作变
    moved = frame(10)
    assert strip._offset == 10
    assert moved != frame(0)


def test_ring_strip_only_scales_the_screens_that_are_painted(qapp):
    strip = RingStrip()
    strip.resize(300, 100)
    # 每屏 400 宽，条带 300 宽：一次最多看到两屏，第 3~6 屏不该被缩放
    strip.set_images([QImage(400, 200, QImage.Format.Format_RGB32) for _ in range(6)])
    strip.grab()
    built = [pixmap is not None for pixmap in strip._pixmaps]
    assert built[0] and not all(built)
    # 缩放后位图作废，重画时才按新尺寸重新生成
    strip.set_zoom(1.5)
    assert not any(pixmap is not None for pixmap in strip._pixmaps)


def test_pack_detail_ring_view_visibility_signals_and_busy_state(qapp):
    detail = PackDetail()
    pack, template = _scene_pack()
    detail.show_pack(pack, template, busy=False)
    assert not detail.view_switch.isHidden()
    assert detail.center_stack.currentWidget() is detail.scroll

    requested = []
    detail.ring_view_requested.connect(lambda: requested.append("view"))
    detail.ring_plan_requested.connect(lambda _item_ids: requested.append("plan"))
    detail.ring_generate_requested.connect(lambda keys: requested.append(("generate", keys)))
    detail.ring_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.ring_page
    assert requested == ["view"]
    # 环还没建起来（后端没有顺序）：接缝按钮不可点
    assert not detail.layer_window.tool_buttons["seams"].isEnabled()
    detail.set_ring([], [], "", RingStatusResponse(order=["screen0", "screen1"]))
    detail.ring_plan_btn.click()
    detail.layer_window.tool_buttons["seams"].click()
    assert requested == ["view", "plan", ("generate", [])]

    detail.set_ring_running(True, action="generate")
    assert not detail.ring_plan_btn.isEnabled()
    assert detail.ring_status_label.text() == detail.tr("Generating seams…")
    detail.set_ring_running(False)
    assert detail.ring_plan_btn.isEnabled()
    detail.resources_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.scroll


def test_ring_cards_add_repeat_remove_and_send_the_order(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=4)
    # 没完成 generate 的屏不能进环
    pack.manifest.items[3].steps["generate"].status = "pending"
    detail.show_pack(pack, template, busy=False)
    assert detail.ring_card_ids() == []

    orders = []
    detail.ring_order_requested.connect(orders.append)
    # 一屏时还不是环：只留在卡片条上，不发给后端
    detail.add_ring_cards(["screen0"])
    assert detail.ring_card_ids() == ["screen0"] and orders == []
    # 同一屏可以再加；凑够两屏就发
    detail.add_ring_cards(["screen0", "screen2"])
    assert detail.ring_card_ids() == ["screen0", "screen0", "screen2"]
    assert orders == [["screen0", "screen0", "screen2"]]
    # 没完成的屏加不进去
    detail.add_ring_cards(["screen3"])
    assert detail.ring_card_ids() == ["screen0", "screen0", "screen2"]

    # 删第二张；删到只剩两张后不能再删
    detail.ring_cards._cards[1].remove_btn.click()
    assert detail.ring_card_ids() == ["screen0", "screen2"]
    assert orders[-1] == ["screen0", "screen2"]
    assert not detail.ring_cards._cards[0].remove_btn.isEnabled()

    # 卡片条上的屏顺序对应后端的环；后端状态回来后以它为准
    status = RingStatusResponse(order=["screen2", "screen0", "screen1"], seams={})
    detail.set_ring([], [], "", status)
    assert detail.ring_card_ids() == ["screen2", "screen0", "screen1"]
    assert detail.ring_cards.ids() == ["screen2", "screen0", "screen1"]
    # 轮询刷新同一个包不改卡片
    detail.show_pack(pack, template, busy=False)
    assert detail.ring_card_ids() == ["screen2", "screen0", "screen1"]


def test_ring_thumbnails_reach_cards_and_picker_and_reset_on_pack_change(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.add_ring_cards(["screen0", "screen1"])

    image = QImage(144, 81, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    detail.set_ring_thumbnail("screen0", QPixmap.fromImage(image))
    assert not detail.ring_cards._cards[0].thumb.pixmap().isNull()
    assert detail.ring_cards.thumb("screen1") is None
    # 之后再建的卡片也带着缩略图
    detail.ring_cards.set_cards([("screen0", "")])
    assert not detail.ring_cards._cards[0].thumb.pixmap().isNull()

    # 换包后旧包的图作废：条目 id 会重名，也不该一直占着内存
    other, other_template = _scene_pack(item_count=2)
    other.manifest.id = "other"
    detail.show_pack(other, other_template, busy=False)
    assert detail.ring_cards.thumb("screen0") is None


def _link_states(detail):
    return [link.bar.property("state") if not link.bar.isHidden() else None
            for link in detail.ring_cards._links]


def test_seam_links_sit_between_cards_and_show_progress_only_for_the_running_seam(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=3)
    detail.show_pack(pack, template, busy=False)
    detail.set_ring([], [], "", RingStatusResponse(order=["screen0", "screen1", "screen2"]))
    links = detail.ring_cards._links
    # 三张卡片三根连接器（含末尾回到开头），都还没有候选
    assert len(links) == 3
    assert _link_states(detail) == ["none", "none", "none"]

    detail.set_ring_running(True)
    detail.set_ring_progress("screen1>screen2", 0.5)
    assert _link_states(detail) == ["none", "running", "none"]
    assert links[1].bar.value() == 50
    # 换到下一条接缝（回环那条）：只留当前这一条
    detail.set_ring_progress("screen2>screen0", 0.2)
    assert _link_states(detail) == ["none", "none", "running"]
    assert links[2].bar.value() == 20
    detail.set_ring_running(False)
    assert _link_states(detail) == ["none", "none", "none"]


def test_seam_links_show_candidate_adopted_and_stale_states(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=3)
    detail.show_pack(pack, template, busy=False)
    candidate = SeamCandidate(
        id="c1", file="/media/c1.png", seed=1, prompt="p", left_sig="a", right_sig="b",
        harmonize=0.3)
    detail.set_ring([], [], "", RingStatusResponse(
        order=["screen0", "screen1", "screen2"],
        seams={
            "screen0>screen1": RingSeamStatus(
                candidates=[candidate], adopted="c1", has_candidates=True),
            "screen1>screen2": RingSeamStatus(candidates=[candidate], has_candidates=True),
            "screen2>screen0": RingSeamStatus(
                candidates=[candidate], adopted="c1", stale=True, has_candidates=True),
        }))
    assert _link_states(detail) == ["adopted", "candidate", "stale"]


def test_seam_links_follow_the_card_order_and_hide_below_two_cards(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=3)
    detail.show_pack(pack, template, busy=False)
    detail.add_ring_cards(["screen0"])
    assert _link_states(detail) == [None]
    detail.add_ring_cards(["screen1"])
    assert _link_states(detail) == ["none", "none"]


def test_ring_cards_drag_moves_a_card_to_the_slot_under_the_drop(qapp):
    bar = RingCardBar()
    bar.set_cards([("a", ""), ("b", ""), ("c", "")])
    bar.resize(600, 130)
    bar.show()
    qapp.processEvents()
    moved = []
    bar.reordered.connect(moved.append)

    def x_between(left: int, right: int) -> int:
        # 两张卡片之间的缝；-1 / len 表示最左 / 最右
        edges = bar._cards
        lo = edges[left].geometry().right() if left >= 0 else -10
        hi = edges[right].x() if right < len(edges) else lo + 20
        return (lo + hi) // 2

    bar.move_card(0, bar.slot_at(x_between(2, 3)))      # a 拖到最后
    assert moved[-1] == ["b", "c", "a"]
    bar.move_card(2, bar.slot_at(x_between(-1, 0)))     # c 拖到最前
    assert moved[-1] == ["c", "a", "b"]
    n = len(moved)
    bar.move_card(1, bar.slot_at(x_between(1, 2)))      # 放回原位：不算换位
    bar.move_card(1, bar.slot_at(x_between(0, 1)))      # 放在自己前面一格：也是原位
    assert len(moved) == n


def test_ring_cards_are_locked_while_busy(qapp):
    bar = RingCardBar()
    bar.set_cards([("a", ""), ("b", ""), ("c", "")])
    removed = []
    bar.remove_requested.connect(removed.append)
    bar.set_editable(False)
    assert not bar.add_btn.isEnabled()
    assert not bar._cards[0].remove_btn.isEnabled()
    bar.set_editable(True)
    bar._cards[2].remove_btn.click()
    assert removed == [2]


def test_ring_item_picker_returns_the_ticked_screens_in_pack_order(qapp):
    picker = RingItemPicker([("a", "A", "", None), ("b", "B", "", None), ("c", "C", "", None)])
    assert not picker.buttons.button(picker.buttons.StandardButton.Ok).isEnabled()
    picker.list.item(2).setSelected(True)
    picker.list.item(0).setSelected(True)
    assert picker.selected_ids() == ["a", "c"]
    assert picker.buttons.button(picker.buttons.StandardButton.Ok).isEnabled()


def test_auto_arrange_sends_each_screen_once_or_all_when_empty(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=3)
    detail.show_pack(pack, template, busy=False)
    planned = []
    detail.ring_plan_requested.connect(planned.append)
    detail.ring_plan_btn.click()
    assert planned == [["screen0", "screen1", "screen2"]]
    detail.add_ring_cards(["screen1", "screen1", "screen0"])
    detail.ring_plan_btn.click()
    assert planned[-1] == ["screen1", "screen0"]


def test_strip_cut_lines_drag_and_emit_sorted_cuts(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    # 两张 300×180 的图：控件高 180，缩放倍数 1，屏幕 x = 长图 x - 偏移
    images = [QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    strip.set_images(images)
    strip.set_cuts([350, 100])
    assert strip.cuts() == [100, 350] and strip.source_width() == 600
    strip.set_offset(0)
    strip.grab()

    edited, seams = [], []
    strip.cuts_edited.connect(edited.append)
    strip.seam_selected.connect(seams.append)
    # 拖第二条切线（屏幕 350）到 屏幕 20 之后：切点要重新排序
    QTest.mousePress(strip, Qt.MouseButton.LeftButton, pos=QPoint(352, 90))
    QTest.mouseMove(strip, QPoint(200, 90))
    QTest.mouseMove(strip, QPoint(20, 90))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, pos=QPoint(20, 90))
    assert edited == [[20, 100]]
    assert strip.cuts() == [20, 100]
    # 拖切线不算点接缝，也不平移条带
    assert seams == [] and strip._offset == 0


def test_strip_shift_click_adds_and_shift_right_click_removes_cuts(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    strip.set_images([QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)])
    strip.set_cuts([100, 350])
    strip.set_offset(0)
    strip.grab()
    edited, seams = [], []
    strip.cuts_edited.connect(edited.append)
    strip.seam_selected.connect(seams.append)
    shift = Qt.KeyboardModifier.ShiftModifier

    # Shift+左键：在点击处加一条，切点保持升序；不算点接缝，也不拖动
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, shift, QPoint(200, 90))
    assert strip.cuts() == [100, 200, 350] and edited == [[100, 200, 350]]
    assert seams == []

    # 点在已有切线上不重复加
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, shift, QPoint(103, 90))
    assert strip.cuts() == [100, 200, 350]

    # 条带视角偏移后，点击位置按世界坐标换算
    strip.set_offset(50)
    strip.grab()
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, shift, QPoint(260, 90))
    assert strip.cuts() == [100, 200, 310, 350]
    strip.set_offset(0)
    strip.grab()

    # Shift+右键：删掉最近的一条；离得太远不删；不按 Shift 的右键不删
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, shift, QPoint(203, 90))
    assert strip.cuts() == [100, 310, 350]
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, shift, QPoint(250, 90))
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(100, 90))
    assert strip.cuts() == [100, 310, 350]
    assert edited[-1] == [100, 310, 350]

    # 至少留一条：后端不接受没有切点
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, shift, QPoint(100, 90))
    # 连续删（中间不重绘）也要对：不能拿重绘前的旧序号越界或删错
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, shift, QPoint(310, 90))
    assert strip.cuts() == [350]
    # 最后一条删不掉
    QTest.mouseClick(strip, Qt.MouseButton.RightButton, shift, QPoint(350, 90))
    assert strip.cuts() == [350]


SEG0 = "segment:504+1200"
SEG1 = "segment:1704+1200"
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
NONE = Qt.KeyboardModifier.NoModifier
SEGMENT_FILES = ["ring/segments/01.png", "ring/segments/02.png"]
LAYER_FILES = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
MERGED_FILE = "ring/layers/504_1200/merged_1.png"


def _wide_images():
    """两屏各 1200 宽：切点 [504, 1704] 正好切成两段 1200。"""
    return [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]


def _segmented_status(layers=None, merged=None, **extra):
    return RingStatusResponse(
        order=["screen0", "screen1"], cuts=[504, 1704], segments=list(SEGMENT_FILES),
        segment_layers=[list(layers or []), []], segment_merged=[list(merged or []), []],
        **extra)


def _ring_detail(status=None):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = _wide_images()
    detail.set_ring(images, images, "", status or RingStatusResponse(
        order=["screen0", "screen1"]))
    return detail


def _tool(detail, name):
    return detail.layer_window.tool_buttons[name]


def _collect(detail, *names):
    """把若干信号的发出记录收进一个列表：(信号名, 参数...)。"""
    events = []
    for name in names:
        getattr(detail, name).connect(lambda *args, n=name: events.append((n, *args)))
    return events


def test_unsegmented_canvas_has_one_node_per_screen_with_version_info(qapp):
    history = VersionHistory(
        base_sig="x", current=1,
        versions=[ImageVersion(file="a.png", kind="extend"),
                  ImageVersion(file="b.png", kind="edit", prompt="p")])
    detail = _ring_detail(RingStatusResponse(
        order=["screen0", "screen1", "screen0"], versions={"screen:screen0": history}))
    window = detail.layer_window
    # 同一屏在环上出现两次也只建一个节点
    assert window.visible_keys() == ["screen:screen0", "screen:screen1"]
    assert window.row("screen:screen0").version.label.text() == "2/2"
    assert window.row("screen:screen1").version is None
    assert detail.canvas_stack.currentWidget() is detail.ring_strip
    assert detail.ring_whole_btn.isHidden()


def test_segmented_canvas_has_segment_nodes_with_layer_children(qapp):
    merged = [MergedLayer(id="M1", path=MERGED_FILE, sources=["L1", "L2"], z=2)]
    history = VersionHistory(base_sig="x", current=0,
                             versions=[ImageVersion(file="e.png", kind="edit")])
    detail = _ring_detail(_segmented_status(
        LAYER_FILES, merged, versions={"layer:504+1200:L2": history}))
    window = detail.layer_window
    # 段节点在前；子节点从顶到底，合并层压在位置相同的原层上面
    assert window.visible_keys() == [
        SEG0, "layer:504+1200:L4", "layer:504+1200:L3", "layer:504+1200:M1",
        "layer:504+1200:L2", "layer:504+1200:L1", SEG1]
    assert window.row("layer:504+1200:L2").version.label.text() == "1/1"
    # 分好层的段 badge 是 ✓，没分层的没有
    assert window.row(SEG0).badge.text() == "✓" and window.row(SEG1).badge.isHidden()
    assert detail.canvas_stack.currentWidget() is detail.segment_canvas
    assert detail.segment_canvas.segment_count() == 2
    # 切点失效（屏变了）：退回整条环和屏节点
    detail.set_ring(_wide_images(), _wide_images(), "",
                    _segmented_status(LAYER_FILES, cuts_stale=True))
    assert window.visible_keys() == ["screen:screen0", "screen:screen1"]
    assert detail.canvas_stack.currentWidget() is detail.ring_strip


def test_whole_ring_toggle_switches_back_to_the_strip(qapp):
    detail = _ring_detail(_segmented_status())
    assert not detail.ring_whole_btn.isHidden()
    detail.ring_whole_btn.click()
    assert detail.canvas_stack.currentWidget() is detail.ring_strip
    detail.ring_whole_btn.click()
    assert detail.canvas_stack.currentWidget() is detail.segment_canvas


def test_status_padding_reaches_the_strip(qapp):
    detail = _ring_detail(RingStatusResponse(
        order=["screen0", "screen1"], screen_padding=[40, 0]))
    assert detail.ring_strip._padding == [40, 0]


def test_segment_images_are_requested_once_and_missing_ones_only(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    requested = []
    detail.ring_images_requested.connect(requested.append)
    images = _wide_images()
    detail.set_ring(images, images, "", _segmented_status(LAYER_FILES[:2]))
    assert requested == [[*SEGMENT_FILES[:1], *LAYER_FILES[:2], SEGMENT_FILES[1]]]
    # 轮询刷新同一个状态：已经请求过、还没回来的不重复请求
    detail.set_ring(images, images, "", _segmented_status(LAYER_FILES[:2]))
    assert len(requested) == 1
    detail.set_segment_images({path: QImage(40, 20, QImage.Format.Format_ARGB32)
                               for path in requested[0]})
    detail.set_ring(images, images, "", _segmented_status(LAYER_FILES[:2]))
    assert len(requested) == 1
    # 作废缓存后下一次状态到来时整批重取，旧图在取回之前继续显示
    detail.clear_segment_images()
    detail.set_ring(images, images, "", _segmented_status(LAYER_FILES[:2]))
    assert len(requested) == 2 and set(requested[1]) == set(requested[0])


def test_tool_buttons_follow_segmentation_and_selection(qapp):
    detail = _ring_detail()
    tips = {name: _tool(detail, name).toolTip() for name in ("layer",)}
    # 未分段：外扩 / 接缝 / 分段可用；分层要先分段
    assert [_tool(detail, name).isEnabled() for name in ("extend", "seams", "segment", "layer")] \
        == [True, True, True, False]
    assert tips["layer"] == detail.tr("Segment the ring first")

    segmented = _ring_detail(_segmented_status(LAYER_FILES))
    # 已分段：外扩不可用；分层要先选段，并提示 Ctrl+A
    assert not _tool(segmented, "extend").isEnabled()
    assert not _tool(segmented, "layer").isEnabled()
    assert _tool(segmented, "layer").toolTip() == segmented.tr(
        "Select segments first (Ctrl+A selects all)")
    segmented.layer_window.select([SEG1])
    assert _tool(segmented, "layer").isEnabled()
    # 选中层节点算选中它所在的段
    segmented.layer_window.select([])
    assert not _tool(segmented, "layer").isEnabled()
    segmented.layer_window.select(["layer:504+1200:L2"])
    assert _tool(segmented, "layer").isEnabled()


def test_seams_tool_sends_the_selected_seams_or_nothing(qapp):
    detail = _ring_detail()
    events = _collect(detail, "ring_generate_requested")
    _tool(detail, "seams").click()
    assert events == [("ring_generate_requested", [])]
    # 第 i 个接缝是 order[i] 与 order[i+1]，最后一个接回开头
    detail.ring_strip.set_selected_seams([1, 0])
    _tool(detail, "seams").click()
    assert events[-1] == ("ring_generate_requested", ["screen0>screen1", "screen1>screen0"])
    detail.ring_strip.set_selected_seams([1])
    _tool(detail, "seams").click()
    assert events[-1] == ("ring_generate_requested", ["screen1>screen0"])


def test_extend_tool_targets_selected_screens_and_asks_before_invalidating(qapp, monkeypatch):
    detail = _ring_detail()
    events = _collect(detail, "ring_extend_requested")
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: (asked.append(args), QMessageBox.StandardButton.No)[1])
    # 没选 = 全部；没有接缝候选和分层时不需要确认
    _tool(detail, "extend").click()
    assert events == [("ring_extend_requested", [])] and asked == []
    detail.layer_window.select(["screen:screen1"])
    _tool(detail, "extend").click()
    assert events[-1] == ("ring_extend_requested", ["screen1"])

    # 已有接缝候选：先确认；选"否"不发
    candidate = SeamCandidate(id="c1", file="/m/c1.png", seed=1, prompt="p", left_sig="a",
                              right_sig="b", harmonize=0.3)
    with_seam = RingStatusResponse(
        order=["screen0", "screen1"],
        seams={"screen0>screen1": RingSeamStatus(candidates=[candidate], has_candidates=True)})
    detail.set_ring(_wide_images(), _wide_images(), "", with_seam)
    count = len(events)
    _tool(detail, "extend").click()
    assert len(asked) == 1 and len(events) == count
    # 选"是"才发
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    _tool(detail, "extend").click()
    assert len(events) == count + 1


def test_segment_tool_asks_for_auto_cuts(qapp):
    detail = _ring_detail()
    events = _collect(detail, "ring_cuts_auto_requested")
    _tool(detail, "segment").click()
    assert events == [("ring_cuts_auto_requested",)]


def test_layer_tool_sends_the_segments_of_the_selected_nodes(qapp):
    detail = _ring_detail(_segmented_status(LAYER_FILES))
    events = _collect(detail, "ring_layers_requested")
    detail.layer_window.select([SEG1])
    _tool(detail, "layer").click()
    assert events[-1] == ("ring_layers_requested", [1])
    # 选中层节点 + 另一段：层算它所在的段，去重后按段序排
    detail.layer_window.select([SEG1, "layer:504+1200:L3"])
    _tool(detail, "layer").click()
    assert events[-1] == ("ring_layers_requested", [0, 1])


def test_running_tool_shows_stop_and_the_others_are_disabled(qapp):
    detail = _ring_detail(_segmented_status(LAYER_FILES))
    detail.layer_window.select([SEG1])
    stops = _collect(detail, "stop_requested")

    detail.set_ring_running(True, action="layers")
    assert _tool(detail, "layer").text() == detail.tr("Stop layer")
    assert _tool(detail, "layer").isEnabled()
    assert not any(_tool(detail, name).isEnabled() for name in ("extend", "seams", "segment"))
    # 再点同一个按钮 = 停止
    _tool(detail, "layer").click()
    assert stops == [("stop_requested",)]
    detail.set_ring_running(False)
    assert _tool(detail, "layer").text() == detail.tr("Layer")
    assert _tool(detail, "seams").isEnabled()

    # 接缝 / 外扩各自对应自己的按钮
    detail.set_ring_running(True, action="redo")
    assert _tool(detail, "seams").text() == detail.tr("Stop seams")
    detail.set_ring_running(True, action="extend")
    assert _tool(detail, "extend").text() == detail.tr("Stop extend")
    # 自动切割是单次操作，同样显示"停止分段"
    detail.set_ring_running(False)
    detail.set_ring_operation("cuts")
    assert _tool(detail, "segment").text() == detail.tr("Stop segment")
    assert _tool(detail, "segment").isEnabled()
    assert not _tool(detail, "seams").isEnabled()
    detail.set_ring_operation(None)
    assert _tool(detail, "segment").text() == detail.tr("Segment")

    # 文字改图占着模型但不对应任何工具：四个都禁用，没有"停止"字样
    detail.set_ring_running(True, action="edit")
    assert not any(_tool(detail, name).isEnabled() for name in ("extend", "seams", "segment", "layer"))
    assert all(not _tool(detail, name).text().startswith(detail.tr("Stop"))
               for name in ("extend", "seams", "segment", "layer"))
    # 短操作（合并、切版本）期间也禁用
    detail.set_ring_running(False)
    detail.set_ring_operation("version")
    assert not any(_tool(detail, name).isEnabled() for name in ("extend", "seams", "segment", "layer"))
    assert not detail.layer_window.send_btn.isEnabled()


def test_node_progress_shows_a_percentage_badge_and_clears_after_the_task(qapp):
    detail = _ring_detail()
    detail.set_ring_running(True, action="extend")
    detail.set_node_progress("screen:screen1", 0.5)
    assert detail.layer_window.row("screen:screen1").badge.text() == "50%"
    # 重建树（例如状态刷新）后百分比还在
    detail.set_ring([], [], "", RingStatusResponse(order=["screen0", "screen1"]))
    assert detail.layer_window.row("screen:screen1").badge.text() == "50%"
    detail.set_ring_running(False)
    assert detail.layer_window.row("screen:screen1").badge.isHidden()


def test_layering_progress_marks_the_running_and_waiting_segments(qapp):
    detail = _ring_detail(_segmented_status())
    detail.set_ring_running(True, action="layers")
    detail.set_layer_progress(RingLayerProgress(stage="encode", index=0, remaining=[0, 1]))
    window = detail.layer_window
    # 阶段名太长时 badge 会省略显示，完整文字在提示里；离屏环境的字宽不可靠，按提示断言
    assert window.row(SEG0).badge.toolTip() == detail.tr("encoding")
    assert window.row(SEG1).badge.text() == "…"
    detail.set_layer_progress(RingLayerProgress(stage="denoise", index=1, remaining=[1]))
    assert window.row(SEG0).badge.isHidden()
    assert window.row(SEG1).badge.toolTip() == detail.tr("denoising")


def test_layers_progress_signal_fires_only_while_layering_and_once_per_change(qapp):
    detail = _ring_detail(_segmented_status())
    refreshed = _collect(detail, "ring_layers_progressed")

    # 不在分层时，完成段数变化不触发刷新（轮询别的东西时不重复取）
    detail._layers_done_seen = 1
    detail._update_layer_controls()
    assert refreshed == []

    detail.set_ring_running(True, action="layers")
    detail._layers_done_seen = 1
    detail._update_layer_controls()
    assert len(refreshed) == 1
    detail._update_layer_controls()
    assert len(refreshed) == 1

    # 取回的新状态里第 1 段已有图层：树里出现这一段的层
    detail.apply_layer_status(_segmented_status(LAYER_FILES))
    assert detail.layer_window.row("layer:504+1200:L1") is not None


def test_canvas_clicks_and_tree_selection_stay_in_sync(qapp):
    detail = _ring_detail(_segmented_status(LAYER_FILES))
    window, canvas = detail.layer_window, detail.segment_canvas
    # 画布上单击 / Ctrl+单击 / Shift+单击，规则与图层窗口一致
    canvas.segment_clicked.emit(1, NONE)
    assert window.selected_keys() == [SEG1] and canvas.selected() == [1]
    canvas.segment_clicked.emit(0, CTRL)
    assert canvas.selected() == [0, 1] and set(window.selected_keys()) == {SEG0, SEG1}
    canvas.segment_clicked.emit(1, CTRL)
    assert canvas.selected() == [0]
    canvas.segment_clicked.emit(0, NONE)
    canvas.segment_clicked.emit(1, SHIFT)
    assert canvas.selected() == [0, 1]
    # 树里选层节点，画布高亮它所在的段
    window.select(["layer:504+1200:L3"])
    assert canvas.selected() == [0]
    window.select([])
    assert canvas.selected() == []


def test_screen_node_selection_highlights_the_screen_on_the_strip(qapp):
    detail = _ring_detail()
    detail.layer_window.select(["screen:screen1"])
    assert detail.ring_strip.active_image() == 1
    detail.layer_window.select([])
    assert detail.ring_strip.active_image() == -1


def test_edit_version_merge_and_delete_are_forwarded(qapp):
    merged = [MergedLayer(id="M1", path=MERGED_FILE, sources=["L1", "L2"], z=2)]
    detail = _ring_detail(_segmented_status(LAYER_FILES, merged))
    window = detail.layer_window
    events = _collect(detail, "ring_edit_requested", "ring_version_requested",
                      "ring_layers_merge_requested", "ring_merged_delete_requested",
                      "ring_layer_opened")

    window.select([SEG0])
    window.prompt_edit.setPlainText("make the ground steeper")
    window.send_btn.click()
    assert events[-1] == ("ring_edit_requested", SEG0, "make the ground steeper")

    window.version_requested.emit("layer:504+1200:L2", 0)
    assert events[-1] == ("ring_version_requested", "layer:504+1200:L2", 0)

    # 合并 / 删除合并层：层 key 换算成段序号和层编号
    window.select(["layer:504+1200:L1", "layer:504+1200:L2"])
    window.merge_btn.click()
    assert events[-1][:2] == ("ring_layers_merge_requested", 0)
    assert sorted(events[-1][2]) == ["L1", "L2"]
    window.select(["layer:504+1200:M1"])
    window.delete_btn.click()
    assert events[-1] == ("ring_merged_delete_requested", 0, "M1")

    # 双击节点：放大它当前显示的图
    window.node_opened.emit(SEG1)
    assert events[-1] == ("ring_layer_opened", SEGMENT_FILES[1])
    window.node_opened.emit("layer:504+1200:L1")
    assert events[-1] == ("ring_layer_opened", LAYER_FILES[0])


def test_eyes_decide_which_layers_the_canvas_composites(qapp):
    detail = _ring_detail(_segmented_status(LAYER_FILES[:2]))
    red, blue, base = (QImage(8, 4, QImage.Format.Format_ARGB32) for _ in range(3))
    base.fill(QColor("white"))
    red.fill(QColor("red"))
    blue.fill(Qt.GlobalColor.transparent)
    for x in range(4):
        for y in range(4):
            blue.setPixelColor(x, y, QColor("blue"))
    detail.set_segment_images({SEGMENT_FILES[0]: base, SEGMENT_FILES[1]: base,
                               LAYER_FILES[0]: red, LAYER_FILES[1]: blue})
    canvas = detail.segment_canvas

    # 层按从底到顶叠：L2 的蓝压在 L1 的红上，右半边没被盖住
    shown = canvas._images[0]
    assert shown.pixelColor(1, 1) == QColor("blue") and shown.pixelColor(6, 1) == QColor("red")
    # 关掉 L2 的眼睛：蓝色消失
    detail.layer_window.toggle_visible("layer:504+1200:L2")
    assert canvas._images[0].pixelColor(1, 1) == QColor("red")
    # 没分层的段显示段图
    assert canvas._images[1].pixelColor(1, 1) == QColor("white")
    # 段节点的眼睛关掉：整段不画
    detail.layer_window.toggle_visible(SEG1)
    assert canvas._images[1].pixelColor(1, 1).alpha() == 0


def test_card_selection_selects_its_screen_node(qapp):
    detail = _ring_detail()
    detail.ring_cards.card_selected.emit(1)
    assert detail.layer_window.selected_keys() == ["screen:screen1"]
    assert detail.ring_strip.active_image() == 1


def test_non_scene_and_single_item_keep_ring_switch_hidden(qapp):
    detail = PackDetail()
    character_pack, character_template = _scene_pack("character")
    detail.show_pack(character_pack, character_template, busy=False)
    assert detail.view_switch.isHidden()
    scene_pack, scene_template = _scene_pack(item_count=1)
    detail.show_pack(scene_pack, scene_template, busy=False)
    assert detail.view_switch.isHidden()


def _two_screens():
    red, blue = QImage(100, 180, QImage.Format.Format_RGB32), QImage(100, 180, QImage.Format.Format_RGB32)
    red.fill(QColor("red"))
    blue.fill(QColor("blue"))
    return [red, blue]


def test_strip_click_selects_nearest_seam_and_ignores_far_clicks(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_offset(0)
    strip.grab()
    selected = []
    strip.seam_selected.connect(selected.append)
    # 偏移 0：x=0 是"第 2 张图左边"的交界，即第 0 张图的接缝回绕点之外的接缝 1（蓝→红）
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(100 + 5, 90))
    assert selected == [0]
    assert strip.selected_seam() == 0
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(50, 90))
    assert selected == [0]


def _ring_status():
    candidate = lambda cid, seed: SeamCandidate(
        id=cid, file=f"/media/{cid}.png", seed=seed, prompt="p", left_sig="a",
        right_sig="b", harmonize=0.3)
    return RingStatusResponse(
        order=["screen0", "screen1"],
        seams={
            "screen0>screen1": RingSeamStatus(
                candidates=[candidate("c1", 11), candidate("c2", 22)], adopted="c1",
                has_candidates=True),
            "screen1>screen0": RingSeamStatus(),
        })


def test_seam_panel_adopt_redo_and_busy_state(qapp):
    detail = PackDetail()
    pack, template = _scene_pack()
    detail.show_pack(pack, template, busy=False)
    images = _two_screens()
    detail.set_ring(images, images, "status", _ring_status())
    assert detail.seam_panel.isHidden()

    events = []
    detail.ring_candidates_requested.connect(lambda key: events.append(("preview", key)))
    detail.ring_adopt_requested.connect(lambda key, cid: events.append(("adopt", key, cid)))
    detail.ring_redo_requested.connect(lambda key, extra: events.append(("redo", key, extra)))
    detail.ring_strip.seam_selected.emit(0)
    assert not detail.seam_panel.isHidden()
    assert events ==[("preview", "screen0>screen1")]
    # 默认选中已采用的候选，所以"采用"不可点；换到另一个才可点
    assert detail.seam_candidate_box.currentData() == "c1"
    assert not detail.seam_adopt_btn.isEnabled()
    detail.seam_candidate_box.setCurrentIndex(1)
    assert detail.seam_adopt_btn.isEnabled()
    detail.seam_adopt_btn.click()
    detail.seam_prompt.setText(" a stream ")
    detail.seam_redo_btn.click()
    assert events[1:] == [("adopt", "screen0>screen1", "c2"),
                          ("redo", "screen0>screen1", "a stream")]

    detail.set_ring_running(True)
    assert not detail.seam_redo_btn.isEnabled() and not detail.seam_adopt_btn.isEnabled()
    detail.set_ring_running(False)
    assert detail.seam_redo_btn.isEnabled()

    # 还没有候选的接缝：只能重做
    detail.ring_strip.seam_selected.emit(1)
    assert not detail.seam_adopt_btn.isEnabled() and detail.seam_redo_btn.isEnabled()
    assert not detail.seam_candidate_box.isEnabled()


class _FakeRingClient:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def generate_ring(self, pack_id, stop_event=None, keys=None, extra_prompt=None,
                      on_progress=None):
        self.calls.append(("generate", keys, extra_prompt))
        return PackResponse.from_error("boom") if self.fail_at == "generate" else PackResponse()

    def ring_status(self, pack_id):
        self.calls.append("status")
        return _ring_status()

    def adopt_ring(self, pack_id, key, candidate_id):
        self.calls.append(("adopt", key, candidate_id))
        return PackResponse()

    def render_ring(self, pack_id):
        self.calls.append("render")
        return PackResponse()


def test_redo_adopts_the_newest_candidate_then_renders(qapp):
    page = LibraryPage()
    page._client = _FakeRingClient()
    result = page._redo_seam("pack", "screen0>screen1", "a stream", threading.Event())
    assert result.ok
    assert page._client.calls == [("generate", ["screen0>screen1"], "a stream"), "status",
                                  ("adopt", "screen0>screen1", "c2"), "render"]


def test_redo_stops_at_the_first_failure_and_reports_it(qapp):
    page = LibraryPage()
    page._client = _FakeRingClient(fail_at="generate")
    result = page._redo_seam("pack", "screen0>screen1", "", threading.Event())
    assert not result.ok and result.error == "boom"
    assert page._client.calls == [("generate", ["screen0>screen1"], None)]


def test_next_seam_progress_refreshes_the_preview_of_the_one_just_finished(qapp):
    page = LibraryPage()
    page.detail.set_ring_progress = lambda key, fraction: None
    loads = []
    page._load_ring_preview = lambda: loads.append(1)
    page._on_ring_progress("a>b", 0.0)
    page._on_ring_progress("a>b", 0.6)
    assert loads == []
    # 换到下一条接缝：上一条已经存盘并合成
    page._on_ring_progress("b>c", 0.0)
    assert loads == [1]


def test_ring_preview_reuses_cached_images_until_their_version_changes(qapp):
    page = LibraryPage()
    downloads = []
    image = QImage(8, 4, QImage.Format.Format_RGB32)
    page._fetch_ring_image = lambda pack, relative: (downloads.append(relative), image)[1]
    pack, _ = _scene_pack()
    used = set()
    page._fetch_cached_ring_image(pack, "a.png", "v1", used)
    page._fetch_cached_ring_image(pack, "a.png", "v1", used)
    assert downloads == ["a.png"]
    # 内容版本变了（重合成）才重新下载
    page._fetch_cached_ring_image(pack, "a.png", "v2", used)
    assert downloads == ["a.png", "a.png"]


def test_picker_icons_share_one_tile_size_whatever_the_screen_resolution(qapp):
    wide, tall = QPixmap(300, 100), QPixmap(50, 200)
    wide.fill(QColor("red"))
    tall.fill(QColor("blue"))
    picker = RingItemPicker([("a", "A", "", wide), ("b", "B", "", tall)])
    sizes = [picker.list.item(row).icon().pixmap(PICKER_THUMB).size() for row in range(2)]
    assert sizes[0] == sizes[1] == PICKER_THUMB


def test_strip_wheel_zoom_keeps_anchor_point_still(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_offset(40)
    strip.grab()
    anchor, anchor_y = 120.0, 60.0
    world_before = (strip._offset + anchor) / strip._total_width
    # 锚点下方的图内纵向位置（占图高比例）也要不动
    y_before = (anchor_y - strip._pan_y) / strip._image_height
    strip.set_zoom(2.0, anchor, anchor_y)
    assert strip.zoom() == 2.0
    world_after = (strip._offset + anchor) / strip._total_width
    assert abs(world_before - world_after) < 1e-6
    assert abs(y_before - (anchor_y - strip._pan_y) / strip._image_height) < 1e-6


def test_strip_zoom_range_is_wide_and_clamped(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.grab()
    strip.set_zoom(100)
    assert strip.zoom() == ZOOM_MAX
    strip.set_zoom(0.0001)
    assert strip.zoom() == ZOOM_MIN


def test_strip_middle_drag_pans_both_axes_and_left_drag_does_not(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.grab()
    before = (strip._offset, strip._pan_y)
    # 左键在空白处拖：画面不动
    QTest.mousePress(strip, Qt.MouseButton.LeftButton, pos=QPoint(50, 90))
    QTest.mouseMove(strip, QPoint(20, 120))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, pos=QPoint(20, 120))
    assert (strip._offset, strip._pan_y) == before
    # 中键拖：横向、纵向都能移，图不再钉在控件顶部
    QTest.mousePress(strip, Qt.MouseButton.MiddleButton, pos=QPoint(200, 90))
    QTest.mouseMove(strip, QPoint(180, 100))
    QTest.mouseMove(strip, QPoint(150, 120))
    QTest.mouseRelease(strip, Qt.MouseButton.MiddleButton, pos=QPoint(150, 120))
    assert strip._offset != before[0]
    assert strip._pan_y == before[1] + 30
    assert strip.selected_seam() == -1


def test_strip_middle_drag_near_a_cut_line_pans_instead_of_moving_it(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    strip.set_images([QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)])
    strip.set_cuts([100])
    strip.set_offset(0)
    strip.grab()
    edited = []
    strip.cuts_edited.connect(edited.append)
    QTest.mousePress(strip, Qt.MouseButton.MiddleButton, pos=QPoint(101, 90))
    QTest.mouseMove(strip, QPoint(60, 90))
    QTest.mouseRelease(strip, Qt.MouseButton.MiddleButton, pos=QPoint(60, 90))
    assert strip.cuts() == [100] and edited == []
    assert strip._offset == 41


def test_strip_lines_only_span_the_image_height(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_cuts([50])
    strip.set_zoom(0.5)
    strip._pan_y = 60
    strip.set_offset(0)
    target = QImage(strip.size(), QImage.Format.Format_ARGB32)
    target.fill(Qt.GlobalColor.transparent)
    strip.render(target)
    # 图高 90：在 y 60～150 之间；接缝线（x=0）在上下两侧的空白处不画，和旁边的背景一样
    assert target.pixelColor(0, 20) == target.pixelColor(5, 20)
    assert target.pixelColor(0, 170) == target.pixelColor(5, 170)
    assert target.pixelColor(0, 100) != target.pixelColor(5, 100)


def test_compare_toggle_floats_top_right_and_plan_sits_with_the_members(qapp):
    """前/后开关浮在条带右上角，规划顺序和屏幕选择同一行。"""
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.resize(800, 600)
    detail.ring_view_btn.click()
    detail.ring_strip.resize(400, 180)
    # 未显示的控件不发 resizeEvent，要先 show 让浮层按新尺寸重新定位
    detail.show()
    qapp.processEvents()
    # 前/后开关是条带的子控件，贴右上角（离边 OVERLAY_MARGIN）
    bar = detail.ring_compare_bar
    assert bar.parent() is detail.ring_strip
    assert bar.x() + bar.width() == detail.ring_strip.width() - OVERLAY_MARGIN
    assert bar.y() == OVERLAY_MARGIN
    # 规划顺序和屏幕选择在同一行（纵向中心对齐到屏幕选择区），不在按钮行里
    members = detail.ring_cards
    plan_top = detail.ring_plan_btn.mapTo(detail, detail.ring_plan_btn.rect().topLeft()).y()
    members_top = members.mapTo(detail, members.rect().topLeft()).y()
    assert abs(plan_top - members_top) < members.height()
    # 画布在左、图层窗口在右，工具按钮都在图层窗口里
    canvas_right = detail.ring_strip.mapTo(detail, detail.ring_strip.rect().topRight()).x()
    window_left = detail.layer_window.mapTo(detail, detail.layer_window.rect().topLeft()).x()
    assert canvas_right <= window_left


def test_switching_before_after_keeps_the_view_position(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_offset(40)
    strip._pan_y = -20
    strip.grab()
    # 前/后切换（换一组同样的图）和轮询刷新都不能把画面拉回起点
    strip.set_images(_two_screens())
    strip.grab()
    assert strip._offset == 40
    assert strip._pan_y == -20
    strip.reset_view()
    assert strip._offset == 0.0 and strip._pan_y == 0.0


def test_strip_double_click_resets_zoom_and_offset(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_offset(77)
    strip.set_zoom(2.0)
    strip._pan_y = 33
    strip.grab()
    QTest.mouseDClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(150, 90))
    assert strip.zoom() == 1.0
    assert strip._offset == 0.0
    assert strip._pan_y == 0.0


def test_strip_padding_mask_is_drawn_and_does_not_break_painting(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    image = QImage(300, 180, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    strip.set_images([image, image])
    strip.set_padding([60, 0])
    strip.set_offset(0)
    shot = strip.grab().toImage()
    # 第一屏上缘补齐 30 原图像素 = 30 屏幕像素：被深色遮罩压暗；第二屏不补，仍是白的
    assert shot.pixelColor(100, 10).lightness() < 200
    assert shot.pixelColor(100, 90).lightness() > 240
    assert shot.pixelColor(350, 10).lightness() > 240
    # padding 比图少或多都不报错
    strip.set_padding([10])
    strip.grab()
    strip.set_padding([])
    strip.grab()


def test_strip_ctrl_click_multi_selects_seams_and_plain_click_resets(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    strip.set_images([QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)])
    strip.set_offset(0)
    strip.grab()
    single, multi = [], []
    strip.seam_selected.connect(single.append)
    strip.seams_selected.connect(multi.append)
    ctrl = Qt.KeyboardModifier.ControlModifier

    # 屏幕 x=300 是接缝 0，x=0 是接缝 1
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(300, 90))
    assert strip.selected_seam() == 0 and strip.selected_seams() == [0]
    assert single == [0] and multi[-1] == [0]

    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, ctrl, QPoint(2, 90))
    assert strip.selected_seams() == [0, 1] and multi[-1] == [0, 1]
    assert single == [0]

    # Ctrl 再点已选中的：取消
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, ctrl, QPoint(300, 90))
    assert strip.selected_seams() == [1] and strip.selected_seam() == 1

    # 普通单击重置为单选，并照旧发 seam_selected
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, ctrl, QPoint(300, 90))
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     QPoint(300, 90))
    assert strip.selected_seams() == [0] and single[-1] == 0

    strip.set_selected_seams([1, 0, 5])
    assert strip.selected_seams() == [0, 1]
    strip.grab()
