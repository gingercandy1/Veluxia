"""框选局部改图前端契约：窗口分发、画布拖框、详情页中转、页面透传。不连后端。"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest

from src import resources  # noqa: F401  注册 :/svg 图标，图层行的眼睛要用
from src.app.ui.library.layer_window import LayerNode, LayerWindow
from src.app.ui.library.pack_detail import PackDetail
from src.app.ui.library.ring_strip import RingStrip
from src.app.ui.library.segment_canvas import SegmentCanvas
from src.shared.schemas import (
    CollectionItem,
    Manifest,
    PackResponse,
    RingStatusResponse,
    StepState,
    TemplateInfo,
    TemplateStepInfo,
)

SEG0 = "segment:504+1200"
SEG1 = "segment:1704+1200"
SEGMENT_FILES = ["ring/segments/01.png", "ring/segments/02.png"]
NONE = Qt.KeyboardModifier.NoModifier


def _approx(box, expected):
    return all(abs(a - b) < 0.02 for a, b in zip(box, expected))


def test_layer_window_sends_region_edit_only_with_a_box(qapp):
    win = LayerWindow()
    win.resize(400, 600)
    win.set_nodes([LayerNode(SEG0, "Segment 1"), LayerNode(SEG1, "Segment 2")])
    full, region = [], []
    win.edit_requested.connect(lambda key, prompt: full.append((key, prompt)))
    win.region_edit_requested.connect(
        lambda key, prompt, bbox: region.append((key, prompt, bbox)))
    win.select([SEG0])
    win.prompt_edit.setPlainText("steeper")
    win.set_region_bbox([0.1, 0.2, 0.5, 0.4])
    win.send_btn.click()
    assert len(region) == 1 and full == []
    assert region[0][:2] == (SEG0, "steeper") and _approx(region[0][2], [0.1, 0.2, 0.5, 0.4])
    # 非法框当没有框：退回整图改
    win.set_region_bbox([0.9, 0.9, 0.5, 0.5])
    assert win.region_bbox() is None
    win.prompt_edit.setPlainText("whole")
    win.send_btn.click()
    assert full == [(SEG0, "whole")] and len(region) == 1


def _red_image(width, height):
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    return image


def test_segment_canvas_drag_creates_and_right_click_clears(qapp):
    canvas = SegmentCanvas()
    canvas.resize(800, 300)
    canvas.show()
    canvas.set_images([_red_image(400, 200)])
    canvas.set_selected([0])
    canvas.set_region_mode(True)
    bounds = canvas.segment_rect(0)
    assert not bounds.isEmpty()
    got = []
    canvas.region_changed.connect(got.append)
    origin = bounds.topLeft().toPoint() + QPoint(100, 50)
    target = bounds.topLeft().toPoint() + QPoint(300, 200)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, NONE, origin)
    QTest.mouseMove(canvas, target)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, NONE, target)
    assert got and _approx(got[-1], [100 / bounds.width(), 50 / bounds.height(),
                                     200 / bounds.width(), 150 / bounds.height()])
    assert _approx(canvas.region_bbox(), got[-1])
    # 右键清框并外发 None
    QTest.mouseClick(canvas, Qt.MouseButton.RightButton, NONE, origin)
    assert got[-1] is None and canvas.region_bbox() is None


def test_ring_strip_region_targets_the_highlighted_screen(qapp):
    strip = RingStrip()
    strip.resize(600, 180)
    strip.show()
    strip.set_images([_red_image(1200, 180), _red_image(1200, 180)])
    strip.set_active_image(1, focus=False)
    strip.set_offset(1200)          # 把第 1 屏挪进视口
    strip.set_region_mode(True)
    bounds = strip._region_bounds()
    assert bounds is not None and bounds.width() == 1200.0
    got = []
    strip.region_changed.connect(got.append)
    origin = QPoint(int(bounds.x()) + 50, int(bounds.y()) + 20)
    target = QPoint(int(bounds.x()) + 250, int(bounds.y()) + 100)
    QTest.mousePress(strip, Qt.MouseButton.LeftButton, NONE, origin)
    QTest.mouseMove(strip, target)
    QTest.mouseRelease(strip, Qt.MouseButton.LeftButton, NONE, target)
    assert got and _approx(got[-1], [50 / 1200, 20 / 180, 200 / 1200, 80 / 180])


def _scene_pack():
    items = [
        CollectionItem(
            id=f"screen{index}", prompt=f"scene {index}",
            steps={"generate": StepState(status="done", outputs=[f"screen{index}/generate.png"])})
        for index in range(3)
    ]
    manifest = Manifest(id="scene_pack", type="scene", template="scene_background", items=items)
    template = TemplateInfo(
        id="scene_background", type="scene", name="Scene", steps=["generate"],
        step_details=[TemplateStepInfo(id="generate", type="image.generate",
                                       deliverable=True)])
    return PackResponse(manifest=manifest, media_base="/media/library/scene_pack"), template


def _segmented_detail(qapp):
    detail = PackDetail()
    detail.resize(1000, 700)
    detail.show()
    qapp.processEvents()
    pack, template = _scene_pack()
    detail.show_pack(pack, template, busy=False)
    detail.ring_view_btn.click()
    images = [_red_image(1200, 180), _red_image(1200, 180)]
    detail.set_ring(images, images, "", RingStatusResponse(
        order=["screen0", "screen1"], cuts=[504, 1704], segments=list(SEGMENT_FILES),
        segment_layers=[[], []], segment_merged=[[], []]))
    detail.set_segment_images({path: _red_image(600, 300) for path in SEGMENT_FILES})
    qapp.processEvents()
    return detail


def test_pack_detail_region_tool_selects_send_and_exits(qapp):
    detail = _segmented_detail(qapp)
    canvas = detail.segment_canvas
    assert detail.canvas_stack.currentWidget() is canvas
    events = []
    detail.ring_region_edit_requested.connect(
        lambda target, prompt, bbox: events.append((target, prompt, bbox)))
    detail.layer_window.select([SEG0])
    assert detail.layer_window.tool_buttons["region"].isEnabled()
    detail.layer_window.tool_buttons["region"].click()
    assert canvas.region_mode()
    bounds = canvas.segment_rect(0)
    assert not bounds.isEmpty()
    origin = bounds.topLeft().toPoint() + QPoint(60, 60)
    target = bounds.topLeft().toPoint() + QPoint(260, 160)
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, NONE, origin)
    QTest.mouseMove(canvas, target)
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, NONE, target)
    assert detail.layer_window.region_bbox() is not None
    detail.layer_window.prompt_edit.setPlainText("add a pine")
    detail.layer_window.send_btn.click()
    assert len(events) == 1
    assert events[0][0] == SEG0 and events[0][1] == "add a pine"
    assert _approx(events[0][2], [60 / bounds.width(), 60 / bounds.height(),
                                  200 / bounds.width(), 100 / bounds.height()])
    # 开跑即退模式：框清掉，按钮回到 Region
    detail.set_ring_running(True, action="edit")
    assert not canvas.region_mode() and detail.layer_window.region_bbox() is None
    assert detail.layer_window.tool_buttons["region"].text() == "Region"


def test_page_edit_with_bbox_passes_bbox_to_client(qapp, monkeypatch):
    from src.app.ui.library.library_page import LibraryPage

    page = LibraryPage()
    pack, template = _scene_pack()
    page._client = _RecordingClient()
    page.detail.show_pack(pack, template, busy=False)
    started = []

    def fake_start(fn, *args, on_ok, on_error=None, **kwargs):
        started.append((fn, args, kwargs))
        return object()

    monkeypatch.setattr(page, "_start", fake_start)
    monkeypatch.setattr(page, "_refresh_packs", lambda: None)
    monkeypatch.setattr(page, "_update_polling", lambda: None)
    bbox = [0.1, 0.1, 0.5, 0.5]
    page._run_ring_action("edit", key="segment:0+10", argument="steeper", bbox=bbox)
    fn, args, call_kwargs = started[-1]
    fn(*args, **call_kwargs)
    assert page._client.calls[-1] == (
        "edit_ring_target", ("scene_pack", "segment:0+10", "steeper"), {"bbox": bbox})


class _RecordingClient:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, {k: v for k, v in kwargs.items()
                                            if k not in ("stop_event", "on_progress")}))
            return PackResponse()

        return call
