"""循环预览控件与详情页签契约，不连接后端。"""
import threading

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest

from src.app.ui.library.library_page import LibraryPage
from src.app.ui.library.pack_detail import PackDetail
from src.app.ui.library.ring_strip import RingStrip
from src.shared.schemas import (
    CollectionItem,
    Manifest,
    PackResponse,
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
    assert not strip._ticker.is_running()

    images = [
        QImage(50, 180, QImage.Format.Format_RGB32),
        QImage(75, 180, QImage.Format.Format_RGB32),
    ]
    images[0].fill(QColor("red"))
    images[1].fill(QColor("blue"))
    strip.set_images(images)
    assert strip.seam_count() == 2
    assert strip._ticker.is_running()
    strip.set_playing(False)

    def frame(offset):
        strip.set_offset(offset)
        target = QImage(strip.size(), QImage.Format.Format_ARGB32)
        target.fill(Qt.GlobalColor.transparent)
        strip.render(target)
        return target.convertToFormat(QImage.Format.Format_RGBA8888).bits().tobytes()

    total_width = sum(image.width() for image in images)
    assert frame(0) == frame(total_width)
    assert not strip.is_playing() and not strip._ticker.is_running()
    strip.set_playing(True)
    assert strip.is_playing() and strip._ticker.is_running()
    strip._last_tick -= 0.5
    old_offset = strip._offset
    strip._advance()
    assert strip._offset != old_offset


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
    detail.ring_render_requested.connect(lambda: requested.append("render"))
    detail.ring_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.ring_strip.parentWidget()
    assert requested == ["view"]
    detail.ring_plan_btn.click()
    detail.ring_generate_btn.click()
    detail.ring_render_btn.click()
    assert requested == ["view", "plan", "generate", "render"]

    detail.set_ring_running(True)
    assert all(not button.isEnabled() for button in (
        detail.ring_plan_btn, detail.ring_generate_btn, detail.ring_render_btn))
    assert detail.ring_status_label.text() == detail.tr("Generating seams…")
    detail.set_ring_running(False)
    assert all(button.isEnabled() for button in (
        detail.ring_plan_btn, detail.ring_generate_btn, detail.ring_render_btn))
    detail.resources_view_btn.click()
    assert detail.center_stack.currentWidget() is detail.scroll


def test_ring_members_choose_screens_and_survive_refresh(qapp):
    detail = PackDetail()
    pack, template = _scene_pack(item_count=4)
    # 没完成 generate 的屏不能参与循环，不出勾选框
    pack.manifest.items[3].steps["generate"].status = "pending"
    detail.show_pack(pack, template, busy=False)
    assert detail.ring_member_ids() == ["screen0", "screen1", "screen2"]

    planned = []
    detail.ring_plan_requested.connect(planned.append)
    detail._ring_member_boxes["screen1"].setChecked(False)
    detail.ring_plan_btn.click()
    assert planned == [["screen0", "screen2"]]

    # 少于两屏不能规划
    detail._ring_member_boxes["screen2"].setChecked(False)
    assert not detail.ring_plan_btn.isEnabled()
    detail._ring_member_boxes["screen2"].setChecked(True)
    assert detail.ring_plan_btn.isEnabled()

    # 轮询刷新同一个包不能把用户的勾选冲掉
    detail.show_pack(pack, template, busy=False)
    assert detail.ring_member_ids() == ["screen0", "screen2"]

    # 规划结果回来后按环成员重勾；勾选和环不一致时提示要重新规划
    status = RingStatusResponse(order=["screen2", "screen0"], seams={})
    detail.set_ring([], [], "", status)
    assert detail.ring_member_ids() == ["screen0", "screen2"]
    assert detail.ring_members_hint.isHidden()
    detail._ring_member_boxes["screen1"].setChecked(True)
    assert not detail.ring_members_hint.isHidden()


def test_strip_cut_lines_drag_and_emit_sorted_cuts(qapp):
    strip = RingStrip()
    strip.resize(400, 180)
    # 两张 300×180 的图：控件高 180，缩放倍数 1，屏幕 x = 长图 x - 偏移
    images = [QImage(300, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    strip.set_images(images)
    strip.set_playing(False)
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
    assert detail.layer_panel.isHidden()

    first_layers = [f"ring/layers/504_1200/layer_{i}.png" for i in range(1, 5)]
    status = RingStatusResponse(
        order=["screen0", "screen1"], cuts=[504, 1704],
        segments=["ring/segments/01.png", "ring/segments/02.png"],
        segment_layers=[first_layers, []])
    detail.set_ring(images, images, "", status)
    assert not detail.layer_panel.isHidden()
    assert detail.layer_segment_box.count() == 2
    assert detail.layer_segment_box.itemText(0).endswith("✓")
    assert "1/2" in detail.layer_info.text()
    assert detail.layer_run_btn.isEnabled()
    # 选中第 1 段：要取段图 + 4 层
    assert requested == [0]
    thumb = QImage(40, 20, QImage.Format.Format_ARGB32)
    detail.set_segment_images({path: thumb for path in detail.segment_paths(0)})
    assert all(not t.isHidden() and t.pixmap() is not None for t in detail.layer_thumbs)
    QTest.mouseDClick(detail.layer_thumbs[2], Qt.MouseButton.LeftButton)
    assert opened == [first_layers[1]]

    # 第 2 段还没分层：只显示段图
    detail.layer_segment_box.setCurrentIndex(1)
    assert requested == [0, 1]
    assert not detail.layer_thumbs[0].isHidden() and detail.layer_thumbs[1].isHidden()

    detail.layer_run_btn.click()
    detail.layer_redo_btn.click()
    assert layered == [[], [1]]

    detail.set_ring_running(True, layering=True)
    assert not detail.layer_run_btn.isEnabled() and not detail.layer_redo_btn.isEnabled()
    assert detail.ring_status_label.text() == detail.tr("Layering segments…")
    detail.set_ring_running(False)
    assert detail.layer_redo_btn.isEnabled()

    # 全部做完：不再能点"分层"，但可以单段重做
    done = status.model_copy(update={"segment_layers": [first_layers, first_layers]})
    detail.set_ring(images, images, "", done)
    assert not detail.layer_run_btn.isEnabled() and detail.layer_redo_btn.isEnabled()
    # 切点失效：分层面板收起
    detail.set_ring(images, images, "", done.model_copy(update={"cuts_stale": True}))
    assert detail.layer_panel.isHidden()


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
    strip.set_playing(False)
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
    assert not detail.ring_strip.is_playing()
    assert events == [("preview", "screen0>screen1")]
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

    def generate_ring(self, pack_id, stop_event=None, keys=None, extra_prompt=None):
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


def test_strip_wheel_zoom_keeps_anchor_point_still(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_playing(False)
    strip.set_offset(40)
    strip.grab()
    anchor = 120.0
    world_before = (strip._offset + anchor) / strip._total_width
    strip.set_zoom(2.0, anchor)
    assert strip.zoom() == 2.0
    world_after = (strip._offset + anchor) / strip._total_width
    assert abs(world_before - world_after) < 1e-6


def test_strip_drag_pans_stops_playback_and_click_still_selects(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_playing(True)
    strip.grab()
    states = []
    strip.playing_changed.connect(states.append)
    before = strip._offset
    QTest.mousePress(strip, Qt.MouseButton.LeftButton, pos=QPoint(200, 90))
    QTest.mouseMove(strip, QPoint(180, 90))
    QTest.mouseMove(strip, QPoint(150, 90))
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, pos=QPoint(150, 90))
    assert not strip.is_playing()
    assert states == [False]
    assert strip._offset != before
    assert strip.selected_seam() == -1


def test_strip_double_click_resets_zoom_and_offset(qapp):
    strip = RingStrip()
    strip.resize(300, 180)
    strip.set_images(_two_screens())
    strip.set_playing(False)
    strip.set_offset(77)
    strip.set_zoom(2.0)
    strip.grab()
    QTest.mouseDClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(150, 90))
    assert strip.zoom() == 1.0
    assert strip._offset == 0.0
