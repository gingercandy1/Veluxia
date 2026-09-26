"""资料库界面：进度与分类计算、花园卡片、新建表单、剧本审阅、详情与执行状态。不连后端。"""
import json

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QTableWidgetItem

from src.app.ui.library.detail_panel import DetailPanel, ScriptEditor
from src.app.ui.library.garden import GardenView
from src.app.ui.library.library_page import LibraryPage
from src.app.ui.library.pack_detail import PackDetail
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
    FieldOption,
    Manifest,
    PackListResponse,
    PackResponse,
    StepState,
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
