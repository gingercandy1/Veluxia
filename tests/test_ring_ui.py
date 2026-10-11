"""循环预览控件与详情页签契约，不连接后端。"""
import threading

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QWidget

from src.app.ui.library.library_page import LibraryPage
from src.app.ui.library.pack_detail import PackDetail
from src.app.ui.library.ring_cards import PICKER_THUMB, RingCardBar, RingItemPicker
from src.app.ui.library.ring_strip import OVERLAY_MARGIN, ZOOM_MAX, ZOOM_MIN, RingStrip
from src import resources  # noqa: F401  注册 :/svg 图标，图层行的眼睛要用
from src.shared.schemas import (
    CollectionItem,
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
    detail.ring_generate_requested.connect(lambda: requested.append("generate"))
    detail.ring_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.ring_strip.parentWidget()
    assert requested == ["view"]
    detail.ring_plan_btn.click()
    detail.ring_generate_btn.click()
    assert requested == ["view", "plan", "generate"]

    detail.set_ring_running(True)
    assert all(not button.isEnabled() for button in (
        detail.ring_plan_btn, detail.ring_generate_btn))
    assert detail.ring_status_label.text() == detail.tr("Generating seams…")
    detail.set_ring_running(False)
    assert all(button.isEnabled() for button in (
        detail.ring_plan_btn, detail.ring_generate_btn))
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


def test_pack_detail_cut_controls_follow_status_and_edits(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    status = RingStatusResponse(order=["screen0", "screen1"], seams={}, cuts=[500, 1700])
    detail.set_ring(images, images, "", status)
    assert detail.ring_auto_cut_btn.isEnabled()
    # 有切点但还没导出段图：可以确认
    assert detail.ring_confirm_cut_btn.isEnabled()
    assert "2" in detail.ring_cut_info.text()

    confirmed = []
    detail.ring_cuts_confirm_requested.connect(confirmed.append)
    detail.ring_confirm_cut_btn.click()
    assert confirmed == [[500, 1700]]

    done = RingStatusResponse(order=["screen0", "screen1"], seams={}, cuts=[500, 1700],
                              segments=["ring/segments/01.png", "ring/segments/02.png"])
    detail.set_ring(images, images, "", done)
    assert not detail.ring_confirm_cut_btn.isEnabled()

    # 拖成一段太窄：不能确认
    detail.ring_strip.set_cuts([500, 600])
    detail.ring_strip.cuts_edited.emit([500, 600])
    assert not detail.ring_confirm_cut_btn.isEnabled()
    detail.ring_strip.set_cuts([600, 1700])
    detail.ring_strip.cuts_edited.emit([600, 1700])
    assert detail.ring_confirm_cut_btn.isEnabled()

    detail.set_ring_operation("cuts")
    assert not detail.ring_auto_cut_btn.isEnabled()
    assert not detail.ring_confirm_cut_btn.isEnabled()
    detail.set_ring_operation(None)
    assert detail.ring_auto_cut_btn.isEnabled()


def test_layer_panel_progress_thumbs_and_redo(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    requested, layered, opened = [], [], []
    detail.ring_segment_requested.connect(requested.append)
    detail.ring_layers_requested.connect(layered.append)
    detail.ring_layer_opened.connect(opened.append)

    # 没确认切段：不显示分层面板
    detail.set_ring(images, images, "", RingStatusResponse(order=["screen0", "screen1"]))
    assert detail.layer_view_btn.isHidden()

    first_layers = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
    status = RingStatusResponse(
        order=["screen0", "screen1"], cuts=[504, 1704],
        segments=["ring/segments/01.png", "ring/segments/02.png"],
        segment_layers=[first_layers, []])
    detail.set_ring(images, images, "", status)
    # 确认切段后出现第三个页签；点它切到分层页，点回循环页分层页就让位
    assert not detail.layer_view_btn.isHidden()
    detail.layer_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.layer_page
    chips = detail.layer_chips.buttons()
    assert [chip.text() for chip in chips] == ["1 ✓", "2"]
    assert "1/2" in detail.layer_info.text()
    assert detail.layer_run_btn.isEnabled()
    # 选中第 1 段：要取段图 + 4 层
    assert requested == [0]
    thumb = QImage(40, 20, QImage.Format.Format_ARGB32)
    detail.set_segment_images({path: thumb for path in detail.segment_paths(0)})
    stack = detail.layer_stack
    assert stack.row("L4") is not None and stack.row("L1") is not None
    assert stack.row("ref") is not None
    # 双击一行：放大这一层
    QTest.mouseDClick(stack.row("L2"), Qt.MouseButton.LeftButton)
    assert opened == [first_layers[1]]

    # 第 2 段还没分层：只显示段图
    chips[1].click()
    assert requested == [0, 1]
    assert stack.row("ref") is not None and stack.row("L1") is None
    assert detail.selected_segment() == 1

    detail.layer_run_btn.click()
    detail.layer_redo_btn.click()
    assert layered == [[], [1]]

    # 空闲时：已分层的段是完成，其余待做
    assert _segment_cell_states(detail) == ["done", "pending"]

    detail.set_ring_running(True, layering=True)
    assert not detail.layer_run_btn.isEnabled() and not detail.layer_redo_btn.isEnabled()
    assert detail.ring_status_label.text() == detail.tr("Layering segments…")
    # 任务回传：第 2 段正在编码，只剩它自己没做完
    detail.set_layer_progress(RingLayerProgress(stage="encode", index=1, remaining=[1]))
    assert _segment_cell_states(detail) == ["done", "running"]
    assert detail.tr("encoding") in detail.layer_info.text()
    # 强制重做第 1 段：它在剩余里，就算已有图层也不算完成
    detail.set_layer_progress(RingLayerProgress(stage="denoise", index=0, remaining=[0]))
    assert _segment_cell_states(detail) == ["running", "pending"]
    assert detail.tr("denoising") in detail.layer_info.text()
    detail.set_ring_running(False)
    assert detail.layer_redo_btn.isEnabled()
    assert _segment_cell_states(detail) == ["done", "pending"]

    # 全部做完：不再能点"分层"，但可以单段重做
    done = status.model_copy(update={"segment_layers": [first_layers, first_layers]})
    detail.set_ring(images, images, "", done)
    assert not detail.layer_run_btn.isEnabled() and detail.layer_redo_btn.isEnabled()
    # 切点失效：页签收起，正看着分层页时退回循环页
    detail.set_ring(images, images, "", done.model_copy(update={"cuts_stale": True}))
    assert detail.layer_view_btn.isHidden()
    assert detail.center_stack.currentIndex() == 1 and detail.ring_view_btn.isChecked()


def test_merge_refresh_keeps_the_layer_page(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    first_layers = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
    detail.set_ring(images, images, "", _merge_status(first_layers, []))
    detail.layer_view_btn.click()
    requested = []
    detail.ring_segment_requested.connect(requested.append)

    # 合并完成后页面的刷新顺序：作废缓存图 -> 轮询刷新 manifest -> 重新取状态
    detail.clear_segment_images()
    detail.show_pack(pack, template, busy=False)
    assert detail.center_stack.currentWidget() is detail.layer_page
    assert detail.layer_view_btn.isChecked() and not detail.layer_view_btn.isHidden()
    # 缓存作废后当前段的图要重新取
    assert requested == [0]

    merged = [MergedLayer(id="M1", path="ring/layers/504_1200/merged_1.png",
                          sources=["L1", "L2"], z=2)]
    detail.set_ring(images, images, "", _merge_status(first_layers, merged))
    assert detail.center_stack.currentWidget() is detail.layer_page
    assert detail.layer_stack.row("M1") is not None


def test_layers_appear_as_each_segment_finishes(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    first_layers = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
    status = _merge_status([], [])
    detail.set_ring(images, images, "", status)
    refreshed = []
    detail.ring_layers_progressed.connect(lambda: refreshed.append(1))

    # 不在分层时，完成段数变化不触发刷新（轮询别的东西时不重复取）
    detail._layers_done_seen = 1
    detail._update_layer_controls()
    assert refreshed == []

    # 分层中完成段数变了：通知页面去取最新状态；没变就不重复取
    detail.set_ring_running(True, layering=True)
    detail._layers_done_seen = 1
    detail._update_layer_controls()
    assert refreshed == [1]
    detail._update_layer_controls()
    assert refreshed == [1]

    # 取回的新状态里第 1 段已有图层：选段按钮带 ✓，图层列表出现这一段的层
    detail.apply_layer_status(_merge_status(first_layers, []))
    assert detail.layer_chips.buttons()[0].text() == "1 ✓"
    assert detail.layer_stack.row("L1") is not None


def _segment_cell_states(detail):
    return [cell.property("state") for cell in detail.layer_bar._cells]


def _merge_status(first_layers, merged):
    return RingStatusResponse(
        order=["screen0", "screen1"], cuts=[504, 1704],
        segments=["ring/segments/01.png", "ring/segments/02.png"],
        segment_layers=[first_layers, []], segment_merged=[merged, []])


def test_layer_stack_groups_eyes_and_merge_selection(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=2)
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    merge_requests, delete_requests = [], []
    detail.ring_layers_merge_requested.connect(lambda *args: merge_requests.append(args))
    detail.ring_merged_delete_requested.connect(lambda *args: delete_requests.append(args))

    first_layers = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
    merged = [MergedLayer(id="M1", path="ring/layers/504_1200/merged_1.png",
                          sources=["L1", "L2"], z=2)]
    detail.set_ring(images, images, "", _merge_status(first_layers, merged))
    stack = detail.layer_stack
    image = QImage(40, 20, QImage.Format.Format_ARGB32)
    detail.set_segment_images({path: image for path in detail.segment_paths(0)})

    # 分层和合并分成两组，各组从顶到底；合并层的路径也要去取图
    assert merged[0].path in detail.segment_paths(0)
    assert stack.group_layout() == [
        (detail.tr("Merged"), ["M1"]), (detail.tr("Layers"), ["L4", "L3", "L2", "L1"])]

    # 眼睛是 SVG 图标，所有行（含段图行）的眼睛左边缘对齐，不随分组缩进
    detail.resize(1400, 900)
    detail.show()
    detail.layer_view_btn.click()
    left_edges = {
        stack.row(layer_id).eye.mapTo(stack.row(layer_id), QPoint(0, 0)).x()
        for layer_id in ("M1", "L4", "L1", "ref")}
    assert len(left_edges) == 1
    eye = stack.row("L2").eye
    assert not eye.icon().isNull()
    open_key = eye.icon().cacheKey()

    # 预览按层序叠：段图在最底下，合并层压在位置相同的原层上面
    assert stack.visible_ids() == ["ref", "L1", "L2", "M1", "L3", "L4"]
    row_before = stack.row("L3")
    QTest.mouseClick(row_before.eye, Qt.MouseButton.LeftButton)
    assert stack.visible_ids() == ["ref", "L1", "L2", "M1", "L4"]
    # 点眼睛只切换显示：不重建行、不改选中，图标换成闭眼
    assert stack.row("L3") is row_before and stack.selected_ids() == []
    assert row_before.eye.icon().cacheKey() != open_key
    stack.toggle_visible("ref")
    assert "ref" not in stack.visible_ids()

    # 选一层不能合并；选两层才行，段图不算
    assert not stack.merge_btn.isEnabled()
    QTest.mouseClick(stack.row("ref"), Qt.MouseButton.LeftButton)
    assert stack.selected_ids() == []
    QTest.mouseClick(stack.row("L1"), Qt.MouseButton.LeftButton)
    assert not stack.merge_btn.isEnabled()
    QTest.mouseClick(stack.row("M1"), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.ControlModifier)
    assert stack.selected_ids() == ["M1", "L1"]
    # 点眼睛不影响选中
    QTest.mouseClick(stack.row("L1").eye, Qt.MouseButton.LeftButton)
    assert stack.selected_ids() == ["M1", "L1"]
    assert stack.merge_btn.isEnabled() and not stack.delete_btn.isEnabled()
    stack.merge_btn.click()
    assert merge_requests == [(0, ["M1", "L1"])]

    # Shift 选一段范围（按列表从上到下）
    QTest.mouseClick(stack.row("L4"), Qt.MouseButton.LeftButton)
    QTest.mouseClick(stack.row("L2"), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    assert stack.selected_ids() == ["L4", "L3", "L2"]

    # 只选一张合并层才能删
    QTest.mouseClick(stack.row("M1"), Qt.MouseButton.LeftButton)
    assert stack.selected_ids() == ["M1"]
    assert stack.delete_btn.isEnabled()
    stack.delete_btn.click()
    assert delete_requests == [(0, "M1")]

    # 刚合并出新层、眼睛状态保留；换一段则全部睁开
    more = merged + [MergedLayer(id="M2", path="ring/layers/504_1200/merged_2.png",
                                 sources=["L3", "L4"], z=4)]
    detail.set_ring(images, images, "", _merge_status(first_layers, more))
    assert "L3" not in stack.visible_ids() and "M2" in stack.visible_ids()
    detail.layer_chips.buttons()[1].click()
    assert stack.visible_ids() == ["ref"]
    detail.layer_chips.buttons()[0].click()
    assert "L3" in stack.visible_ids()

    # 长任务进行中不能改图层
    stack.select(["L1", "L2"])
    assert stack.merge_btn.isEnabled()
    detail.set_ring_running(True)
    assert not stack.merge_btn.isEnabled()


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
    assert detail.ring_plan_btn.y() < detail.ring_generate_btn.mapTo(
        detail, detail.ring_generate_btn.rect().topLeft()).y()


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
