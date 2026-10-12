"""循环画布页的页面层契约：任务分发、分段对话框、完成后的刷新。不连接后端、不起线程。"""
import threading

from PySide6.QtGui import QImage

from src.app.ui.library import library_page
from src.app.ui.library.library_page import LibraryPage
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


def _scene_pack():
    items = [
        CollectionItem(
            id=f"screen{index}", prompt=f"scene {index}",
            steps={"generate": StepState(status="done", outputs=[f"screen{index}/generate.png"])})
        for index in range(2)
    ]
    manifest = Manifest(id="scene_pack", type="scene", template="scene_background", items=items)
    template = TemplateInfo(
        id="scene_background", type="scene", name="Scene", steps=["generate"],
        step_details=[TemplateStepInfo(id="generate", type="image.generate", deliverable=True)])
    return PackResponse(manifest=manifest, media_base="/media/library/scene_pack"), template


class _RecordingClient:
    """记下被调用的接口和参数，返回成功。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, {k: v for k, v in kwargs.items()
                                            if k not in ("stop_event", "on_progress")}))
            return PackResponse()
        return call


def _page(monkeypatch):
    """页面里打开了一个场景包；_start 不起线程，只记下要执行的函数和参数。"""
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
    return page, started


def _run(page, started, action, **kwargs):
    page._run_ring_action(action, **kwargs)
    fn, args, call_kwargs = started[-1]
    fn(*args, **call_kwargs)
    return page._client.calls[-1]


def test_generate_passes_selected_seams_or_none_for_all(qapp, monkeypatch):
    page, started = _page(monkeypatch)
    assert _run(page, started, "generate", keys=[]) == (
        "generate_ring", ("scene_pack",), {"keys": None})
    page._ring_worker = None
    assert _run(page, started, "generate", keys=["a>b"])[2] == {"keys": ["a>b"]}


def test_extend_edit_and_version_call_the_client(qapp, monkeypatch):
    page, started = _page(monkeypatch)
    # 没选屏 = 全部：传 None
    assert _run(page, started, "extend", item_ids=[]) == (
        "extend_ring", ("scene_pack",), {"item_ids": None})
    page._ring_worker = None
    assert _run(page, started, "extend", item_ids=["screen1"])[2] == {"item_ids": ["screen1"]}
    page._ring_worker = None
    assert _run(page, started, "edit", key="segment:0+10", argument="steeper") == (
        "edit_ring_target", ("scene_pack", "segment:0+10", "steeper"), {})
    page._ring_worker = None
    assert _run(page, started, "version", key="screen:screen0", index=-1) == (
        "select_ring_version", ("scene_pack", "screen:screen0", -1), {})


def test_extend_and_edit_are_long_tasks_but_version_is_not(qapp, monkeypatch):
    page, started = _page(monkeypatch)
    running = []
    monkeypatch.setattr(page.detail, "set_ring_running",
                        lambda value, action="": running.append((value, action)))
    operations = []
    monkeypatch.setattr(page.detail, "set_ring_operation", operations.append)
    for action, kwargs in (("extend", {}), ("edit", {"key": "screen:screen0", "argument": "p"})):
        page._ring_worker = None
        page._run_ring_action(action, **kwargs)
        assert action in library_page.RING_GENERATING
    assert running == [(True, "extend"), (True, "edit")] and operations == []
    page._ring_worker = None
    page._run_ring_action("version", key="screen:screen0", index=0)
    assert operations == ["version"]
    assert "version" not in library_page.RING_GENERATING


def test_stop_reaches_the_running_ring_task_including_auto_cuts(qapp, monkeypatch):
    page, started = _page(monkeypatch)
    for action in ("extend", "edit", "cuts"):
        page._ring_worker = None
        page._run_ring_action(action, key="screen:screen0", argument="p")
        page.detail.stop_requested.emit()
        assert page._ring_stop.is_set(), action
    # 切版本是短操作，不能停
    page._ring_worker = None
    page._run_ring_action("version", key="screen:screen0")
    page._run_stop = threading.Event()
    page._stop_run()
    assert not page._ring_stop.is_set()


def _status_with_suggestion(seams=None):
    return RingStatusResponse(order=["screen0", "screen1"], suggested_cuts=[504, 1704],
                              seams=seams or {})


class _FakeDialog:
    created = []
    result = 1

    def __init__(self, images, cuts, warning="", parent=None):
        self.args = (images, cuts, warning)
        _FakeDialog.created.append(self)

    def exec(self):
        return _FakeDialog.result

    def cuts(self):
        return [10, 20]


def _loaded(page, status):
    images = [QImage(1200, 180, QImage.Format.Format_RGB32) for _ in range(2)]
    page._on_cuts_preview_loaded("scene_pack", (images, images, "", status))


def test_finished_cuts_open_the_dialog_and_ok_confirms(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    _FakeDialog.created, _FakeDialog.result = [], 1
    monkeypatch.setattr(library_page, "CutDialog", _FakeDialog)
    actions = []
    monkeypatch.setattr(page, "_run_ring_action",
                        lambda action, **kwargs: actions.append((action, kwargs)))
    _loaded(page, _status_with_suggestion())
    assert len(_FakeDialog.created) == 1
    images, cuts, warning = _FakeDialog.created[0].args
    assert len(images) == 2 and cuts == [504, 1704]
    # 接缝都还没有候选：提示条数
    assert warning == page.tr("{0} seams are not ready; the cuts may be affected.").format(2)
    assert actions == [("confirm", {"cuts": [10, 20]})]


def test_cancelling_the_cut_dialog_changes_nothing(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    _FakeDialog.created, _FakeDialog.result = [], 0
    monkeypatch.setattr(library_page, "CutDialog", _FakeDialog)
    actions = []
    monkeypatch.setattr(page, "_run_ring_action",
                        lambda action, **kwargs: actions.append((action, kwargs)))
    _loaded(page, _status_with_suggestion())
    assert len(_FakeDialog.created) == 1 and actions == []


def test_cut_dialog_warning_counts_only_missing_or_stale_seams(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    candidate = SeamCandidate(id="c1", file="/m/c1.png", seed=1, prompt="p", left_sig="a",
                              right_sig="b", harmonize=0.3)
    ready = RingSeamStatus(candidates=[candidate], adopted="c1", has_candidates=True)
    stale = RingSeamStatus(candidates=[candidate], adopted="c1", stale=True, has_candidates=True)
    both_ready = _status_with_suggestion({"screen0>screen1": ready, "screen1>screen0": ready})
    assert page._cuts_warning(both_ready) == ""
    one_stale = _status_with_suggestion({"screen0>screen1": ready, "screen1>screen0": stale})
    assert "1 seams" in page._cuts_warning(one_stale)


def test_no_dialog_when_there_is_no_suggestion(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    _FakeDialog.created = []
    monkeypatch.setattr(library_page, "CutDialog", _FakeDialog)
    _loaded(page, RingStatusResponse(order=["screen0", "screen1"]))
    assert _FakeDialog.created == []


def _finish(page, monkeypatch, action):
    """按完成回调的顺序记下后处理：作废缓存图、重新取状态、（分段时）弹对话框。"""
    events = []
    monkeypatch.setattr(page, "_show_detail", lambda response: events.append("show"))
    monkeypatch.setattr(page, "_update_run_state", lambda: None)
    monkeypatch.setattr(page.detail, "clear_segment_images", lambda: events.append("clear"))
    monkeypatch.setattr(page, "_load_ring_preview", lambda: events.append("preview"))
    monkeypatch.setattr(page, "_review_suggested_cuts", lambda: events.append("review"))
    monkeypatch.setattr(page, "_run_ring_action",
                        lambda *args, **kwargs: events.append(("run", args)))
    page._ring_action = action
    page._ring_pack_id = "scene_pack"
    page._on_ring_action_finished(PackResponse())
    return events


def test_extend_edit_version_and_confirm_refresh_images_and_status(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    for action in ("extend", "edit", "version", "confirm"):
        events = _finish(page, monkeypatch, action)
        assert "clear" in events and "preview" in events, action
        assert events.index("clear") < events.index("preview")


def test_confirming_cuts_does_not_start_layering(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    events = _finish(page, monkeypatch, "confirm")
    # 分层只对选中的段做：确认切段不再自动开始
    assert not any(isinstance(event, tuple) and event[0] == "run" for event in events)


def test_finished_auto_cuts_review_the_suggestion_instead_of_plain_refresh(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    events = _finish(page, monkeypatch, "cuts")
    assert "review" in events and "preview" not in events


def test_stopped_extend_still_shows_the_screens_already_done(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    events = []
    monkeypatch.setattr(page, "_update_run_state", lambda: None)
    monkeypatch.setattr(page.detail, "clear_segment_images", lambda: events.append("clear"))
    monkeypatch.setattr(page, "_load_ring_preview", lambda: events.append("preview"))
    page._ring_action = "extend"
    page._ring_pack_id = "scene_pack"
    page._ring_stop.set()
    page._on_ring_action_finished(PackResponse.from_error("cancelled"))
    assert events == ["clear", "preview"]
    # 用户自己点的停止不当成错误显示
    assert page.detail.error_label.isHidden()


def test_extend_progress_goes_to_the_screen_node_and_edit_progress_to_the_target(qapp, monkeypatch):
    page, _ = _page(monkeypatch)
    nodes = []
    monkeypatch.setattr(page.detail, "set_node_progress",
                        lambda key, fraction: nodes.append((key, fraction)))
    monkeypatch.setattr(page, "_load_ring_preview", lambda: None)
    page._ring_action = "extend"
    page._on_ring_progress("screen0", 0.25)
    page._ring_action = "edit"
    page._on_ring_progress("segment:0+10", 0.5)
    assert nodes == [("screen:screen0", 0.25), ("segment:0+10", 0.5)]
