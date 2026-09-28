"""资料库界面：进度与分类计算、花园卡片、新建表单、剧本审阅、详情与执行状态。不连后端。"""
import json

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QTableWidgetItem

from src.app.ui.library.detail_panel import DetailPanel, ScriptEditor
from src.app.ui.library.garden import GardenView
from src.app.ui.library.library_page import LibraryPage
from src.app.ui.library.pack_detail import RUNNING_ROLE, PackDetail
from src.app.ui.library.pack_form import PackForm
from src.app.ui.library.pack_status import (
    asset_progress,
    fallback_template,
    latest_image,
    media_kind,
    pack_cover,
    pack_progress,
    step_chain,
)
from src.shared.schemas import (
    CastMember,
    CollectionItem,
    CollectionStyle,
    FieldOption,
    Manifest,
    NewCollectionItem,
    PackListResponse,
    PackResponse,
    StepState,
    StylePreset,
    TemplateFieldInfo,
    TemplateInfo,
    TemplateStepInfo,
)

CHARACTER = TemplateInfo(
    id="character_basic", type="character", name="角色", cover="trim",
    fields=[TemplateFieldInfo(id="name", label="角色名", required=True),
            TemplateFieldInfo(id="pose", label="姿势", default="stand",
                              options=[FieldOption(value="stand", label="站立"),
                                       FieldOption(value="sit", label="坐姿")])],
    steps=["generate", "trim", "voice"],
    step_details=[
        TemplateStepInfo(id="generate", type="image.generate"),
        TemplateStepInfo(id="trim", type="image.trim", deliverable=True, inputs=["generate"]),
        TemplateStepInfo(id="voice", type="speech.generate", deliverable=True),
    ])
DIALOGUE = TemplateInfo(
    id="dialogue_scene", type="dialogue", steps=["script", "speak"],
    step_details=[
        TemplateStepInfo(id="script", type="text.dialogue", deliverable=True, review=True),
        TemplateStepInfo(id="speak", type="speech.dialogue", deliverable=True, inputs=["script"]),
    ])


def _hero(**steps: StepState) -> CollectionItem:
    return CollectionItem(id="hero", prompt="a knight", fields={"name": "亚瑟"}, steps=steps)


def _character_pack(running: bool = False) -> PackResponse:
    manifest = Manifest(id="c1", name="主角", type="character", template="character_basic", items=[
        _hero(generate=StepState(status="done", outputs=["hero/generate.png"]),
              trim=StepState(status="done", outputs=["hero/trim.png"]),
              voice=StepState(status="error", error="显存不足")),
        CollectionItem(id="mage", prompt="a mage"),
    ])
    return PackResponse(manifest=manifest, media_base="/media/library/c1", running=running)


# ---- 进度计算 ----
def test_asset_progress_counts_only_its_own_chain():
    item = _character_pack().manifest.items[0]
    chain = step_chain(CHARACTER, "trim")
    assert [s.id for s in chain] == ["generate", "trim"]
    # 立绘做完了，不会因为声线出错显示成未完成
    assert asset_progress(item, chain).status == "done"
    assert asset_progress(item, step_chain(CHARACTER, "voice")).status == "error"


def test_media_kind_treats_audio_post_processing_as_audio():
    assert media_kind("speech.generate") == "audio"
    assert media_kind("audio.loop") == "audio"
    assert media_kind("animation.generate") == "video"
    assert media_kind("dialogue.script") == "text"


def test_review_status_outranks_error_and_running():
    manifest = Manifest(id="d1", type="dialogue", template="dialogue_scene", items=[
        CollectionItem(id="a", steps={"script": StepState(status="done")}),
        CollectionItem(id="b", steps={"script": StepState(status="error")}),
    ])
    assert pack_progress(manifest, DIALOGUE, running=True).status == "review"
    manifest.items[0].steps["script"].approved = True
    assert pack_progress(manifest, DIALOGUE, running=True).status == "running"


def test_cover_and_latest_image():
    pack = _character_pack()
    assert pack_cover(pack.manifest, CHARACTER) == "hero/trim.png"
    assert latest_image(pack.manifest.items[1], step_chain(CHARACTER, "trim")) == ""
    assert pack_cover(Manifest(id="d", type="dialogue", template="x"), DIALOGUE) == ""


def test_fallback_template_chains_recorded_steps():
    template = fallback_template(_character_pack().manifest)
    assert template.steps == ["generate", "trim", "voice"]
    assert template.step_details[-1].deliverable
    assert template.step_details[1].inputs == ["generate"]


# ---- 花园 ----
def test_garden_updates_cards_in_place(qapp):
    garden = GardenView()
    templates = {"character_basic": CHARACTER}
    garden.set_packs([_character_pack()], templates)
    card = garden._cards["c1"]
    assert card.meta_label.text() == "0/2"
    assert not card.badge.isHidden() and card.badge.property("status") == "error"

    garden.set_packs([_character_pack(running=True)], templates)
    assert garden._cards["c1"] is card and card.badge.property("status") == "running"
    # 执行中的包卡片进度条走流光，停下后流光也停
    assert card.progress.is_active()
    garden.set_packs([_character_pack()], templates)
    assert not card.progress.is_active()


# ---- 新建表单 ----
def test_form_builds_request_with_fields_and_paste(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.preselect_category("character")
    assert form.cast_section.isHidden()
    assert form.items_table.columnCount() == 3

    form.items_table.setCurrentCell(0, 0)
    form.items_table.paste_rows("a knight\t亚瑟\t坐姿\na mage\t梅林\n")
    request = form.build_request()
    assert request.template == "character_basic"
    assert [(i.prompt, i.fields) for i in request.items] == [
        ("a knight", {"name": "亚瑟", "pose": "sit"}),
        ("a mage", {"name": "梅林", "pose": "stand"}),
    ]


def test_form_requires_items(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    emitted = []
    form.create_requested.connect(emitted.append)
    form.create_btn.click()
    assert not emitted and form.error_label.text()


def test_form_draft_requires_theme(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    emitted = []
    form.draft_requested.connect(emitted.append)
    form.draft_btn.click()
    assert not emitted and form.error_label.text()


def test_form_draft_keeps_user_rows_and_replaces_untouched_drafts(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    form.items_table.setCurrentCell(0, 0)
    form.items_table.paste_rows("my own knight\t亚瑟")
    form.apply_drafts("character_basic", [
        NewCollectionItem(prompt="a mage", fields={"name": "梅林", "pose": "sit"}),
        NewCollectionItem(prompt="a thief", fields={"name": "罗宾"}),
    ])
    # 空行被换掉，手填的行保留在前
    assert [form.items_table.cell_text(r, 0) for r in range(form.items_table.rowCount())] == [
        "my own knight", "a mage", "a thief"]
    assert form.items_table.cell_text(1, 2) == "sit"

    # 改过的起草行算用户的，再起草时保留；没动过的被换掉
    form.items_table.set_cell_text(1, 1, "大法师")
    emitted = []
    form.draft_requested.connect(emitted.append)
    form.draft_theme_edit.setText("奇幻冒险队")
    form.draft_btn.click()
    assert emitted[0].exclude == ["my own knight", "a mage", "a thief"]
    assert emitted[0].theme == "奇幻冒险队" and emitted[0].template == "character_basic"
    form.apply_drafts("character_basic", [NewCollectionItem(prompt="a bard")])
    assert [form.items_table.cell_text(r, 0) for r in range(form.items_table.rowCount())] == [
        "my own knight", "a mage", "a bard"]


def test_form_ignores_drafts_for_another_template(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.preselect_category("dialogue")
    form.apply_drafts("character_basic", [NewCollectionItem(prompt="a mage")])
    assert form.items_table.cell_text(0, 0) == ""


def test_form_dialogue_cast_links_character(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.set_characters([("主角 / 亚瑟", "c1/hero")])
    form.preselect_category("dialogue")
    assert not form.cast_section.isHidden()
    form.items_table.paste_rows("两人在酒馆相遇")
    form.cast_table.setItem(0, 0, QTableWidgetItem("亚瑟"))
    form.cast_table.cellWidget(0, 1).setCurrentIndex(1)
    request = form.build_request()
    assert request.cast == [CastMember(name="亚瑟", character="c1/hero")]


# ---- 剧本审阅 ----
MOTION = TemplateInfo(
    id="character_motion", type="character", source="character", cover="sheet",
    steps=["video", "sheet"],
    step_details=[
        TemplateStepInfo(id="video", type="animation.generate", inputs=["@source"]),
        TemplateStepInfo(id="sheet", type="frames.sheet", deliverable=True, inputs=["video"]),
    ])


def test_form_motion_template_requires_source_character(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, MOTION])
    form.preselect_category("character")
    form.template_combo.setCurrentIndex(form.template_combo.findData("character_motion"))
    assert not form.source_section.isHidden()
    form.items_table.paste_rows("walk to the right")
    with pytest.raises(ValueError):
        form.build_request()  # 还没有可绑定的角色
    form.set_characters([("主角 / 亚瑟", "c1/hero")])
    assert form.build_request().source == "c1/hero"

    form.template_combo.setCurrentIndex(form.template_combo.findData("character_basic"))
    assert form.source_section.isHidden()
    form.items_table.paste_rows("a knight\t亚瑟")
    assert form.build_request().source == ""


def test_media_kind_shows_frames_as_images():
    assert media_kind("frames.sheet") == "image"


def test_page_does_not_offer_motion_packs_as_characters(qapp):
    page = _page(qapp)
    motion = Manifest(id="m1", type="character", template="character_motion", source="c1/hero",
                      items=[CollectionItem(id="walk", prompt="walk")])
    page._packs = [_character_pack(), PackResponse(manifest=motion)]
    assert [value for _, value in page._character_options()] == ["c1/hero", "c1/mage"]


def test_form_preset_fills_style_and_is_dropped_after_manual_edit(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    preset = StylePreset(id="s1", name="手绘", prompt="hand-painted", negative="3d")
    form.set_styles([preset])
    assert not form.delete_preset_btn.isEnabled()
    form.preset_combo.setCurrentIndex(1)
    form.preset_combo.activated.emit(1)
    assert form.style_edit.toPlainText() == "hand-painted" and form.negative_edit.text() == "3d"
    assert form.delete_preset_btn.isEnabled()
    form.items_table.paste_rows("a knight\t亚瑟")
    assert form.build_request().style_preset == "s1"
    form.style_edit.setPlainText("hand-painted, gloomy")
    assert form.build_request().style_preset == ""

    # 刷新列表时选中刚保存的预设
    form.set_styles([preset, StylePreset(id="s2", name="水彩")], select_name="水彩")
    assert form.preset_combo.currentData() == "s2"


def test_form_save_preset_with_existing_name_overwrites(qapp, monkeypatch):
    form = PackForm()
    form.set_styles([StylePreset(id="s1", name="手绘", prompt="old")])
    form.style_edit.setPlainText("new")
    monkeypatch.setattr("src.app.ui.library.pack_form.QInputDialog.getText",
                        lambda *args, **kwargs: ("手绘", True))
    emitted = []
    form.style_save_requested.connect(emitted.append)
    form.save_preset_btn.click()
    assert emitted == [StylePreset(id="s1", name="手绘", prompt="new")]


def test_pack_detail_notices_changed_preset(qapp):
    detail = PackDetail()
    pack = _character_pack()
    pack.manifest.style_preset = "s1"
    pack.manifest.style = CollectionStyle(prompt="hand-painted")
    detail.show_pack(pack, CHARACTER, busy=False)
    detail.set_styles([StylePreset(id="s1", name="手绘", prompt="hand-painted")])
    assert detail.style_notice.isHidden()
    detail.set_styles([StylePreset(id="s1", name="手绘", prompt="watercolor")])
    assert not detail.style_notice.isHidden() and "手绘" in detail.style_notice_label.text()
    detail.set_styles([])  # 预设删了：包里是副本，不提示
    assert detail.style_notice.isHidden()


def test_pack_detail_notices_changed_source(qapp):
    detail = PackDetail()
    pack = _character_pack()
    detail.show_pack(pack, CHARACTER, busy=False)
    assert detail.source_notice.isHidden()
    requested = []
    detail.source_refresh_requested.connect(lambda: requested.append(True))
    detail.show_pack(pack.model_copy(update={"source_changed": True}), CHARACTER, busy=False)
    assert not detail.source_notice.isHidden()
    detail.set_running(True, other_running=False)
    assert not detail.refresh_source_btn.isEnabled()
    detail.set_running(False, other_running=False)
    detail.refresh_source_btn.click()
    assert requested == [True]


def test_script_editor_sends_none_when_unchanged(qapp):
    text = json.dumps({"lines": [{"speaker": "亚瑟", "text": "你好", "emotion": "平静"}]})
    editor = ScriptEditor(text, ["亚瑟", "梅林"], approved=False)
    emitted = []
    editor.approve_requested.connect(emitted.append)
    editor.approve_btn.click()
    assert emitted == [None]

    editor.table.item(0, 2).setText("你好啊")
    editor.table.cellWidget(0, 0).setCurrentText("梅林")
    editor.approve_btn.click()
    assert json.loads(emitted[1])["lines"][0] == {"speaker": "梅林", "emotion": "平静",
                                                  "text": "你好啊"}


# ---- 详情 ----
def test_detail_panel_reports_changes_and_flow(qapp):
    panel = DetailPanel()
    pack = _character_pack()
    item = pack.manifest.items[0]
    step = CHARACTER.step_details[1]
    assert panel.show_asset(pack, CHARACTER, item, step, busy=False)
    assert not panel.show_asset(pack, CHARACTER, item, step, busy=False)
    assert len(panel._redo_buttons) == 2
    panel.set_busy(True)
    assert not any(button.isEnabled() for button in panel._redo_buttons)
    assert "站立" not in panel.fields_label.text() and "亚瑟" in panel.fields_label.text()


def test_detail_panel_set_busy_after_clear(qapp):
    # 清空后流程按钮已被删除，切换资源包时 set_busy 不能再碰它们
    panel = DetailPanel()
    pack = _character_pack()
    panel.show_asset(pack, CHARACTER, pack.manifest.items[0], CHARACTER.step_details[1],
                     busy=False)
    panel.clear()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    panel.set_busy(True)


def test_pack_detail_groups_assets_by_media(qapp):
    view = PackDetail()
    view.show_pack(_character_pack(), CHARACTER, busy=False)
    assert set(view._cards) == {("hero", "trim"), ("mage", "trim"),
                                ("hero", "voice"), ("mage", "voice")}
    assert view._cards[("hero", "voice")].badge.property("status") == "error"


def test_pack_detail_animates_only_what_is_running(qapp):
    view = PackDetail()
    pack = _character_pack(running=True)
    pack.manifest.items[1].steps["generate"] = StepState(status="running")
    view.set_tree([pack], {"character_basic": CHARACTER})
    view.show_pack(pack, CHARACTER, busy=True)
    entry = view._tree_items["c1"]
    # 执行中用跳动的点代替文字状态
    assert entry.data(0, RUNNING_ROLE) and "·" not in entry.text(0)
    assert view._tree_ticker.is_running() and view.progress_strip.is_active()
    assert view._cards[("mage", "trim")].progress.is_active()
    assert not view._cards[("hero", "trim")].progress.is_active()

    # 程序被关掉后 manifest 里残留的 running 不算在跑
    stale = _character_pack()
    stale.manifest.items[1].steps["generate"] = StepState(status="running")
    view.set_tree([stale], {"character_basic": CHARACTER})
    view.show_pack(stale, CHARACTER, busy=False)
    assert not entry.data(0, RUNNING_ROLE) and not view._tree_ticker.is_running()
    assert not view.progress_strip.is_active()
    assert not view._cards[("mage", "trim")].progress.is_active()


# ---- 页面 ----
def _page(qapp) -> LibraryPage:
    page = LibraryPage()
    page._template_list = [CHARACTER]
    page._load_thumbnail = lambda *args: None
    return page


def test_page_run_state_follows_packs(qapp):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack(running=True)]))
    page._open_pack("c1")
    assert page.stack.currentWidget() is page.detail
    assert page.detail.run_btn.isHidden() and page.detail.stop_btn.isEnabled()
    assert page._poll_timer.isActive()

    page._on_packs(PackListResponse(packs=[_character_pack()]))
    assert not page.detail.run_btn.isHidden() and page.detail.run_btn.isEnabled()
    assert not page._poll_timer.isActive()


def test_page_reloads_thumbnails_for_rebuilt_cards(qapp):
    # 切换资源包会重建素材卡片，切回来时新卡片必须重新加载，不能被旧记录跳过
    page = LibraryPage()
    page._template_list = [CHARACTER]
    fetched = []
    page._start = lambda fn, *args, **kwargs: fetched.append(args[0])
    other = _character_pack()
    other.manifest.id, other.media_base = "c2", "/media/library/c2"
    page._on_packs(PackListResponse(packs=[_character_pack(), other]))
    hero_thumb = "/media/library/c1/hero/trim.png"
    page._open_pack("c1")
    page._on_packs(PackListResponse(packs=[_character_pack(), other]))
    assert fetched.count(hero_thumb) == 2  # 花园封面一次、素材卡片一次，轮询不重复加载
    page._open_pack("c2")
    page._open_pack("c1")
    assert fetched.count(hero_thumb) == 3


def test_page_runs_after_approve_when_idle(qapp):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    started = []
    page._run_pack = started.append
    page._after_update("c1")(_character_pack())
    assert started == ["c1"]

    page._packs = [_character_pack(running=True)]
    page._refresh_packs = lambda: None
    page._after_update("c1")(_character_pack())
    assert started == ["c1"]
