"""资料库界面：进度与分类计算、花园卡片、新建表单、剧本审阅、详情与执行状态。不连后端。"""
import json

import pytest
from PySide6.QtCore import QElapsedTimer, QEvent, QPointF, Qt
from PySide6.QtGui import QEnterEvent, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractItemView, QLabel, QLineEdit, QStackedWidget, QWidget

from src.app.ui.base.anim import page_enter
from src.app.ui.library import library_page
from src.app.ui.library.activity import LoadingStrip
from src.app.ui.library.detail_panel import DetailPanel, ScriptEditor
from src.app.ui.library.garden import GardenView
from src.app.ui.library.library_page import LibraryPage, export_file_name
from src.app.ui.library.pack_detail import RUNNING_ROLE, PackDetail
from src.app.ui.library.pack_form import PAGE_BACKGROUND, PackForm
from src.app.ui.library.pack_status import (
    asset_progress,
    fallback_template,
    latest_image,
    media_kind,
    pack_cover,
    pack_progress,
    step_chain,
    step_fraction,
    step_skipped,
)
from src.shared.schemas import (
    BaseResponse,
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


def _garden_with_two_packs(qapp) -> GardenView:
    garden = GardenView()
    villain = _character_pack()
    villain.manifest.id, villain.manifest.name = "c2", "反派"
    villain.manifest.items = [CollectionItem(id="lich", prompt="an undead Lich king",
                                             fields={"name": "巫妖"})]
    garden.set_packs([_character_pack(), villain], {"character_basic": CHARACTER})
    return garden


def test_garden_search_matches_names_items_and_fields(qapp):
    garden = _garden_with_two_packs(qapp)
    section, heading, add, _ids = garden._section_parts["character"]

    garden.search_edit.setText("lich")          # 条目描述，不分大小写
    assert garden._cards["c1"].isHidden() and not garden._cards["c2"].isHidden()
    assert add.isHidden() and heading.text().endswith("1/2")
    garden.search_edit.setText("亚瑟")           # 条目字段
    assert not garden._cards["c1"].isHidden() and garden._cards["c2"].isHidden()
    garden.search_edit.setText("主角 knight")    # 多个词都要命中
    assert not garden._cards["c1"].isHidden()
    garden.search_edit.setText("主角 lich")
    assert section.isHidden() and not garden.empty_label.isHidden()

    garden.search_edit.clear()
    assert not any(card.isHidden() for card in garden._cards.values())
    assert not add.isHidden() and heading.text().endswith("2") and garden.empty_label.isHidden()


def test_garden_search_survives_polling_refresh(qapp):
    garden = _garden_with_two_packs(qapp)
    card = garden._cards["c2"]
    garden.search_edit.setText("巫妖")
    renamed = _character_pack()
    renamed.manifest.name = "巫妖的仆从"         # 轮询时名字变了，过滤跟着更新
    villain = _character_pack()
    villain.manifest.id, villain.manifest.items = "c2", []
    garden.set_packs([renamed, villain], {"character_basic": CHARACTER})
    assert garden._cards["c2"] is card          # 没有重建，缩略图不丢
    assert not garden._cards["c1"].isHidden() and card.isHidden()


def test_garden_shows_loading_only_while_empty(qapp):
    garden = GardenView()
    garden.set_loading(True)
    assert not garden.loading_strip.isHidden() and garden.loading_strip.is_active()
    assert not garden.loading_label.isHidden()
    garden.set_packs([_character_pack()], {"character_basic": CHARACTER})
    garden.set_loading(False)
    assert garden.loading_strip.isHidden() and not garden.loading_strip.is_active()
    garden.set_loading(True)               # 再次进入页面刷新：已有卡片，不闪加载条
    assert garden.loading_strip.isHidden()


def test_page_marks_thumbnails_loading_until_they_arrive(qapp):
    page = LibraryPage()
    page._template_list = [CHARACTER]
    started = []
    page._start = lambda fn, *args, **kwargs: started.append(kwargs)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    card = page.garden.card("c1")
    assert card.is_loading()
    started[-1]["on_error"]("boom")
    assert not card.is_loading()
    card.set_loading(True)
    card.set_thumbnail(QPixmap(10, 10))
    assert not card.is_loading()


def test_detail_panel_shows_loading_for_finished_asset(qapp):
    panel = DetailPanel()
    pack = _character_pack()
    item = pack.manifest.items[0]
    panel.show_asset(pack, CHARACTER, item, CHARACTER.step_details[1], busy=False)  # trim 已完成
    strips = panel.findChildren(LoadingStrip)
    assert strips and strips[0].is_active()
    panel.set_image(panel.key, QPixmap(20, 20))
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not panel.findChildren(LoadingStrip)
    # 没生成的素材不显示加载，只说还没生成
    panel.show_asset(pack, CHARACTER, item, CHARACTER.step_details[2], busy=False)
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not panel.findChildren(LoadingStrip)


# ---- 新建表单 ----
class _Memory:
    """记住上次选择的内存版：测试不读写用户真实的 config/setting.json。"""

    def __init__(self):
        self.store = {}

    def load(self, template_id):
        return dict(self.store.get(template_id, {}))

    def save(self, template_id, defaults, preset):
        self.store[template_id] = {"defaults": defaults, "preset": preset}


@pytest.fixture(autouse=True)
def _isolated_form_memory(monkeypatch):
    monkeypatch.setattr("src.app.ui.library.pack_form.FormMemory", _Memory)


SCENE_FORM = TemplateInfo(
    id="scene_background", type="scene", name="场景", summary="整张场景图",
    fields=[TemplateFieldInfo(id="length", label="长度", default="1",
                              options=[FieldOption(value="1", label="单屏"),
                                       FieldOption(value="2", label="2 屏")]),
            TemplateFieldInfo(id="segment_desc", label="分段描述（用 | 分隔）",
                              segments_field="length")],
    steps=["generate"], step_details=[TemplateStepInfo(id="generate", type="image.generate")])
PROPS = TemplateInfo(id="scene_props", type="scene", name="道具", steps=["generate"],
                     step_details=[TemplateStepInfo(id="generate", type="image.generate")])


def _prompts(form) -> list[str]:
    table = form.items_table
    return [table.cell_text(row, 0) for row in table.filled_rows()]


EFFECT_FORM = TemplateInfo(
    id="effect_skill", type="effect", name="技能", prompt_label="画面 / 动画描述",
    fields=[TemplateFieldInfo(id="sfx", label="音效描述", required=True),
            TemplateFieldInfo(id="reference", label="导入图片", kind="image")],
    steps=["icon"], step_details=[TemplateStepInfo(id="icon", type="image.generate")])


def test_form_image_column_is_not_typed_and_delete_only_clears_it(qapp):
    form = PackForm()
    form.set_templates([EFFECT_FORM])
    table = form.items_table
    column = table.columns()[2]
    assert column.id == "reference" and column.is_image
    table.paste_rows("a fire spell\twhoosh")
    # 手打路径容易写错，图片格只能双击选文件
    assert not table.item(0, 2).flags() & Qt.ItemFlag.ItemIsEditable

    table.set_cell_text(0, 2, "C:/pics/fire.png")
    assert form.build_request().items[0].fields == {"sfx": "whoosh", "reference": "C:/pics/fire.png"}

    table.setCurrentCell(0, 2)
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert table.cell_text(0, 0) == "a fire spell" and table.cell_text(0, 2) == ""
    assert "reference" not in form.build_request().items[0].fields


def test_form_optional_dropdowns_move_to_pack_defaults(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.preselect_category("character")
    assert form.cast_section.isHidden()
    # 可选的"姿势"进了整包设置，表格只剩描述和必填的角色名
    assert form.items_table.columnCount() == 2
    assert form.defaults.values() == {"pose": "stand"}

    form.items_table.setCurrentCell(0, 0)
    form.items_table.paste_rows("a knight\t亚瑟\na mage\t梅林\n")
    assert form.items_table.rowCount() == 3  # 末尾自动留一个空行
    form.defaults.set_values({"pose": "sit"})
    form.items_table.set_row_overrides(1, {"pose": "stand"})
    request = form.build_request()
    assert request.template == "character_basic"
    assert [(i.prompt, i.fields) for i in request.items] == [
        ("a knight", {"name": "亚瑟", "pose": "sit"}),
        ("a mage", {"name": "梅林", "pose": "stand"}),
    ]


def test_form_blank_row_grows_and_rows_can_be_removed(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    table = form.items_table
    assert table.rowCount() == 1
    table.set_cell_text(0, 0, "a knight")
    assert table.rowCount() == 2 and table.is_empty_row(1)
    table.remove_row(0)
    assert table.rowCount() == 1 and table.is_empty_row(0)


def test_form_create_button_counts_items(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    assert not form.create_btn.isEnabled()
    with pytest.raises(ValueError):
        form.build_request()
    form.items_table.paste_rows("a knight\t亚瑟\na mage\t梅林")
    assert form.create_btn.isEnabled() and "2" in form.create_btn.text()
    form.set_busy(True)
    assert not form.create_btn.isEnabled()


def test_form_marks_missing_required_cell(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    emitted = []
    form.create_requested.connect(emitted.append)
    form.items_table.paste_rows("a knight")
    form.create_btn.click()
    assert not emitted and "角色名" in form.error_label.text()
    assert (0, 1) in form.items_table._errors
    form.items_table.set_cell_text(0, 1, "亚瑟")
    assert not form.items_table._errors


def test_form_auto_names_pack_from_template_and_first_item(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    form.items_table.paste_rows("a very long knight description\t亚瑟")
    assert form.build_request().name == "角色 · a very long "
    form.name_edit.setText("骑士团")
    assert form.build_request().name == "骑士团"


def test_form_segment_column_follows_length(qapp):
    form = PackForm()
    form.set_templates([SCENE_FORM])
    table = form.items_table
    assert table.isColumnHidden(1)
    form.defaults.set_values({"length": "2"})
    assert not table.isColumnHidden(1)
    form.defaults.set_values({"length": "1"})
    assert table.isColumnHidden(1)
    # 单独把某一行改成多屏，列也要出现
    table.set_cell_text(0, 0, "幽暗森林")
    table.set_row_overrides(0, {"length": "2"})
    assert not table.isColumnHidden(1)
    table.set_cell_text(0, 1, "森林|村庄|城堡")
    with pytest.raises(ValueError):
        form.build_request()
    assert (0, 1) in table._errors
    table.set_cell_text(0, 1, "森林|村庄")
    assert form.build_request().items[0].fields == {
        "length": "2", "segment_desc": "森林|村庄"}


def test_form_single_column_hides_header_and_summary_is_shown(qapp):
    form = PackForm()
    form.set_templates([SCENE_FORM, PROPS])
    form.picker.select_template("scene_props")
    assert form.items_table.horizontalHeader().isHidden()
    form.picker.select_template("scene_background")
    assert not form.items_table.horizontalHeader().isHidden()
    assert form.picker.summary_label.text() == "整张场景图"


def test_form_keeps_prompts_within_category_and_clears_across(qapp):
    form = PackForm()
    form.set_templates([SCENE_FORM, PROPS, CHARACTER])
    form.preselect_category("scene")
    form.items_table.paste_rows("幽暗森林\n林间小屋")
    form.picker.select_template("scene_props")
    assert _prompts(form) == ["幽暗森林", "林间小屋"]
    form.preselect_category("character")
    assert _prompts(form) == []


def test_form_remembers_pack_defaults_per_template(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    form.defaults.set_values({"pose": "sit"})
    form.items_table.paste_rows("a knight\t亚瑟")
    form._on_create()
    # 后端还没确认创建成功（可能被拒绝），这时不记
    assert form._memory.store == {}
    form.reset()
    assert form.defaults.values() == {"pose": "sit"}
    assert _prompts(form) == []


def test_form_does_not_remember_settings_of_a_rejected_pack(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    form.defaults.set_values({"pose": "sit"})
    form.items_table.paste_rows("a knight\t亚瑟")
    form._on_create()
    form.set_error("创建失败")
    form.reset()
    assert form._memory.store == {}
    # 改回站立后再成功创建：记住的是成功那次的设置
    form.items_table.paste_rows("a knight\t亚瑟")
    form.defaults.set_values({"pose": "stand"})
    form._on_create()
    form.reset()
    assert form._memory.store["character_basic"]["defaults"] == {"pose": "stand"}


@pytest.fixture
def shown_form(qapp):
    """显示并激活的表单：键盘测试要靠焦点；断言失败时也要关窗，否则后面的测试抢不到激活。"""
    form = PackForm()
    form.set_templates([PROPS])
    form.resize(900, 700)
    form.show()
    form.activateWindow()
    assert QTest.qWaitForWindowActive(form)
    yield form
    form.close()


def test_form_enter_on_last_row_adds_a_row_and_keeps_typing(qapp, shown_form):
    form = shown_form
    table = form.items_table
    table.setCurrentCell(0, 0)
    table.editItem(table.item(0, 0))
    # 编辑框不一定拿到焦点（前面的测试窗口可能还占着激活），直接从视口里找
    editor = table.viewport().findChild(QLineEdit)
    assert editor is not None
    # keyClicks 逐字转成按键，非 ASCII 字符会让进程直接崩掉
    QTest.keyClicks(editor, "lantern")
    QTest.keyClick(editor, Qt.Key.Key_Return)
    # 委托收到回车后是排队提交的
    qapp.processEvents()
    assert _prompts(form) == ["lantern"]
    assert table.rowCount() == 2
    # 光标落到新的空行，直接接着写下一条
    assert table.currentRow() == 1
    assert table.state() == QAbstractItemView.State.EditingState


def test_form_delete_key_removes_selected_row(shown_form):
    form = shown_form
    table = form.items_table
    table.paste_rows("石灯笼\n木桶\n")
    table.setFocus()
    table.selectRow(0)
    QTest.keyClick(table, Qt.Key.Key_Delete)
    assert _prompts(form) == ["木桶"]


def test_form_escape_leaves_input_first_then_goes_back(shown_form):
    form = shown_form
    back = []
    form.back_requested.connect(lambda: back.append(True))
    form.name_edit.setFocus()
    QTest.keyClick(form.name_edit, Qt.Key.Key_Escape)
    assert back == [] and not form.name_edit.hasFocus()
    QTest.keyClick(form, Qt.Key.Key_Escape)
    assert back == [True]


def test_form_ctrl_enter_creates(shown_form):
    form = shown_form
    emitted = []
    form.create_requested.connect(emitted.append)
    form.items_table.paste_rows("石灯笼\n")
    form.name_edit.setFocus()
    QTest.keyClick(form.name_edit, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert [request.items[0].prompt for request in emitted] == ["石灯笼"]


def _run_animations(qapp, ms: int = 450):
    timer = QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        qapp.processEvents()


def test_create_button_bounces_and_returns_to_its_place(qapp, shown_form):
    form = shown_form
    button = form.create_btn
    form.items_table.paste_rows("石灯笼\n")
    home = button.geometry()
    sizes = []
    timer = QElapsedTimer()
    timer.start()
    form._on_create()
    while timer.elapsed() < 450:
        qapp.processEvents()
        sizes.append(button.width())
    # 先缩小再弹回，最后严丝合缝停在原位
    assert min(sizes) < home.width()
    assert button.geometry() == home


def test_page_slide_ends_exactly_in_place(qapp):
    """切页的滑入动画结束后页面必须回到原位：偏了几十像素右边就被切掉，拖一下窗口才恢复。"""
    stack = QStackedWidget()
    pages = [QWidget(), QWidget()]
    for page in pages:
        stack.addWidget(page)
    stack.resize(900, 600)
    stack.show()
    _run_animations(qapp, 50)
    for forward in (True, False, True, False):
        page = pages[1] if forward else pages[0]
        stack.setCurrentWidget(page)
        page_enter(page, PAGE_BACKGROUND, 12 if forward else -12)
        _run_animations(qapp, 400)
        assert page.geometry() == stack.contentsRect(), forward
    stack.close()


def test_form_blocks_stay_in_place_after_animated_switches(qapp):
    """切到只有一个模板的分类时模板小块那排会隐藏、内容整体上移；
    动画不能把内容留在旧位置，否则会和下面的"风格"重叠。"""
    form = PackForm()
    form.set_templates([SCENE_FORM, PROPS, DIALOGUE, CHARACTER])
    form.resize(900, 700)
    form.show()
    _run_animations(qapp, 100)
    for category in ("dialogue", "scene", "character", "dialogue"):
        form.picker.tabs._buttons[category].click()
        _run_animations(qapp)
        before = form.content.pos()
        layout = form.content.parentWidget().layout()
        layout.invalidate()
        layout.activate()
        assert form.content.pos() == before, category
        assert form.content.geometry().bottom() < form.style_toggle.geometry().top(), category
    form.close()


def test_form_draft_requires_theme(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    emitted = []
    form.draft_requested.connect(emitted.append)
    form.draft_btn.click()
    assert not emitted and form.error_label.text()
    assert form.draft_panel.is_expanded()


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
    assert _prompts(form) == ["my own knight", "a mage", "a thief"]
    # 模型给这一条选了和整包不同的姿势：记成这一行的单独设置
    assert form.items_table.row_overrides(1) == {"pose": "sit"}

    # 改过的起草行算用户的，再起草时保留；没动过的被换掉
    form.items_table.set_cell_text(1, 1, "大法师")
    emitted = []
    form.draft_requested.connect(emitted.append)
    form.draft_theme_edit.setText("奇幻冒险队")
    form.draft_btn.click()
    assert emitted[0].exclude == ["my own knight", "a mage", "a thief"]
    assert emitted[0].theme == "奇幻冒险队" and emitted[0].template == "character_basic"
    form.apply_drafts("character_basic", [NewCollectionItem(prompt="a bard")])
    assert _prompts(form) == ["my own knight", "a mage", "a bard"]


def test_form_ignores_drafts_for_another_template(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.preselect_category("dialogue")
    form.apply_drafts("character_basic", [NewCollectionItem(prompt="a mage")])
    assert _prompts(form) == []


def test_form_dialogue_cast_links_character(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, DIALOGUE])
    form.set_characters([("主角 / 亚瑟", "c1/hero")])
    form.preselect_category("dialogue")
    assert not form.cast_section.isHidden()
    form.items_table.paste_rows("两人在酒馆相遇")
    form.cast_chips.add_member(CastMember(name="亚瑟", character="c1/hero"))
    form.cast_chips.add_member(CastMember(name=""))  # 什么都没填的不算
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
    form.picker.select_template("character_motion")
    assert not form.source_section.isHidden()
    form.items_table.paste_rows("walk to the right")
    with pytest.raises(ValueError):
        form.build_request()  # 还没有可绑定的角色
    form.set_characters([("主角 / 亚瑟", "c1/hero")])
    assert form.build_request().source == "c1/hero"

    form.picker.select_template("character_basic")
    assert form.source_section.isHidden()
    form.items_table.paste_rows("a knight\t亚瑟")
    assert form.build_request().source == ""


def test_form_common_motions_drafts_for_the_bound_character(qapp):
    form = PackForm()
    form.set_templates([CHARACTER, MOTION])
    form.preselect_category("character")
    assert form.common_motions_btn.isHidden()
    form.picker.select_template("character_motion")
    assert not form.common_motions_btn.isHidden()

    emitted = []
    form.draft_requested.connect(emitted.append)
    form.common_motions_btn.click()
    assert not emitted and form.error_label.text()

    form.set_characters([("球 / 红球", "c1/ball")])
    form.common_motions_btn.click()
    [request] = emitted
    assert request.template == "character_motion" and request.source == "c1/ball"
    assert request.theme and request.count == 6

    form.draft_theme_edit.setText("战斗动作")
    form.draft_btn.click()
    assert emitted[1].source == "c1/ball" and emitted[1].theme == "战斗动作"


def _motion_form(qapp):
    """带预设的动作模板：播放方式是必填下拉，每行一列。"""
    from src.shared.schemas import FieldOption, TemplateFieldInfo, TemplatePresetInfo
    modes = [FieldOption(value=v, label=v) for v in ("loop", "pingpong", "once")]
    template = TemplateInfo(
        id="motion_presets", type="character", source="character", cover="sheet",
        prompt_label="Action",
        fields=[TemplateFieldInfo(id="action", label="Action name"),
                TemplateFieldInfo(id="loop_mode", label="Playback", required=True,
                                  default="loop", options=modes)],
        presets=[TemplatePresetInfo(name="待机", prompt="stands and breathes",
                                    fields={"action": "idle", "loop_mode": "pingpong"}),
                 TemplatePresetInfo(name="死亡", prompt="falls to the ground",
                                    fields={"action": "death", "loop_mode": "once"})],
        steps=["video", "sheet"],
        step_details=[
            TemplateStepInfo(id="video", type="animation.generate", inputs=["@source"]),
            TemplateStepInfo(id="sheet", type="frames.sheet", deliverable=True, inputs=["video"]),
        ])
    form = PackForm()
    form.set_templates([CHARACTER, template])
    form.preselect_category("character")
    form.picker.select_template("motion_presets")
    form.set_characters([("主角 / 亚瑟", "c1/hero")])
    return form


def test_form_shows_the_presets_button_only_for_templates_that_have_presets(qapp):
    form = _motion_form(qapp)
    assert not form.preset_items_btn.isHidden()
    form.picker.select_template("character_basic")
    assert form.preset_items_btn.isHidden()


def test_form_add_presets_fills_rows_with_their_own_loop_mode(qapp):
    form = _motion_form(qapp)
    form.add_presets()
    request = form.build_request()
    assert [(i.prompt, i.fields["action"], i.fields["loop_mode"]) for i in request.items] == [
        ("stands and breathes", "idle", "pingpong"), ("falls to the ground", "death", "once")]


def test_form_filled_rows_are_contiguous_with_one_trailing_blank(qapp):
    from src.shared.schemas import NewCollectionItem
    form = _motion_form(qapp)
    form.apply_drafts("motion_presets", [
        NewCollectionItem(prompt=p, fields={"action": p, "loop_mode": "loop"}) for p in "abc"])
    form.add_presets()
    table = form.items_table
    blanks = [row for row in range(table.rowCount()) if table.is_empty_row(row)]
    assert blanks == [table.rowCount() - 1]


def test_form_add_presets_twice_does_not_duplicate_rows(qapp):
    form = _motion_form(qapp)
    form.add_presets()
    form.add_presets()
    assert len(form.build_request().items) == 2


def test_form_add_presets_keeps_rows_the_user_wrote_and_edited(qapp):
    form = _motion_form(qapp)
    form.items_table.paste_rows("jump over a box")
    form.add_presets()
    prompts = [i.prompt for i in form.build_request().items]
    assert prompts[0] == "jump over a box" and len(prompts) == 3
    # 改过的预设行和删掉的预设行：再点一次只补回被删的，不覆盖用户的修改
    table = form.items_table
    row = next(r for r in table.filled_rows() if table.cell_text(r, 0) == "stands and breathes")
    table.set_cell_text(row, 0, "stands tall and sways")
    table.removeRow(next(r for r in table.filled_rows() if table.cell_text(r, 0) == "falls to the ground"))
    form.add_presets()
    prompts = [i.prompt for i in form.build_request().items]
    assert "stands tall and sways" in prompts and "falls to the ground" in prompts


def test_form_add_presets_rows_survive_a_later_ai_draft(qapp):
    form = _motion_form(qapp)
    form.add_presets()
    form.apply_drafts("motion_presets", [NewCollectionItem(
        prompt="slides under a gap", fields={"action": "slide", "loop_mode": "once"})])
    prompts = [i.prompt for i in form.build_request().items]
    # AI 起草只换掉空行和上次起草的行；预设行不是 AI 写的，不能被清掉
    assert prompts == ["stands and breathes", "falls to the ground", "slides under a gap"]


def test_form_add_presets_does_nothing_for_templates_without_presets(qapp):
    form = PackForm()
    form.set_templates([CHARACTER])
    form.preselect_category("character")
    form.add_presets()
    assert form.items_table.filled_rows() == []


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


def test_new_pack_form_fills_the_page_after_switching(qapp):
    page = _page(qapp)
    page.resize(1000, 700)
    page.show()
    _run_animations(qapp, 50)
    for show in (page._show_form, page._show_garden, page._show_form):
        show()
        _run_animations(qapp, 400)
        assert page.stack.currentWidget().geometry() == page.stack.contentsRect()
    page.close()


def test_page_runs_new_pack_right_after_creating(qapp):
    page = _page(qapp)
    started = []
    page._run_pack = started.append
    page._on_created(_character_pack())
    assert page.stack.currentWidget() is page.detail and started == ["c1"]

    # 别的资源包正在执行时只打开，不抢着执行
    page = _page(qapp)
    started = []
    page._run_pack = started.append
    page._is_busy = lambda: True
    page._on_created(_character_pack())
    assert started == []


def test_add_form_keeps_pack_settings_and_drafts_around_existing_items(qapp):
    form = PackForm(append=True)
    pack = _character_pack().manifest
    pack.style.prompt = "teal"
    form.set_target(pack, CHARACTER)
    # 模板、名称、风格锁都沿用包里的，不再让用户选
    assert form.picker.isHidden() and form.name_edit.isHidden() and form.style_toggle.isHidden()
    assert "主角" in form.title_label.text()
    form.items_table.setCurrentCell(0, 0)
    form.items_table.paste_rows("a bard\t丹德里恩")
    added = []
    form.add_requested.connect(lambda pack_id, request: added.append((pack_id, request)))
    form.create_btn.click()
    assert added[0][0] == "c1"
    assert [(i.prompt, i.fields["name"]) for i in added[0][1].items] == [("a bard", "丹德里恩")]

    drafts = []
    form.draft_requested.connect(drafts.append)
    form.draft_theme_edit.setText("冒险队")
    form.draft_btn.click()
    # 包里已有的条目也告诉模型，避免起草出重复的
    assert drafts[0].exclude == ["a bard", "a knight", "a mage"] and drafts[0].style == "teal"

    # 回到同一个包保留没提交的行，换了包才清空
    form.set_target(pack, CHARACTER)
    assert _prompts(form) == ["a bard"]
    other = pack.model_copy(update={"id": "c2"})
    form.set_target(other, CHARACTER)
    assert _prompts(form) == []


def test_add_form_does_not_ask_for_source_character(qapp):
    form = PackForm(append=True)
    motion = Manifest(id="m1", type="character", template="character_motion", source="c1/hero",
                      items=[CollectionItem(id="walk", prompt="walk")])
    form.set_target(motion, MOTION)
    assert form.source_section.isHidden()
    form.items_table.setCurrentCell(0, 0)
    form.items_table.paste_rows("run")
    assert [i.prompt for i in form.build_add_request().items] == ["run"]


def test_page_runs_pack_right_after_adding_items(qapp):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    page.detail.add_btn.click()
    assert page.stack.currentWidget() is page.add_form

    started = []
    page._run_pack = started.append
    grown = _character_pack()
    grown.manifest.items.append(CollectionItem(id="03", prompt="a bard"))
    page._on_items_added(grown)
    assert page.stack.currentWidget() is page.detail and started == ["c1"]
    assert [i.id for i in page.detail.pack.manifest.items] == ["hero", "mage", "03"]

    # 执行中不能追加
    page._on_packs(PackListResponse(packs=[_character_pack(running=True)]))
    assert not page.detail.add_btn.isEnabled()


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


def test_card_action_buttons_follow_pack_state(qapp):
    view = PackDetail()
    pack = _character_pack()
    view.show_pack(pack, CHARACTER, busy=False)
    trim = view.asset_card("hero", "trim")
    voice = view.asset_card("hero", "voice")
    assert trim.delete_btn.isEnabled() and trim.export_btn.isEnabled()
    # 没做完的素材没有东西可导出
    assert not voice.export_btn.isEnabled() and voice.export_btn.toolTip()

    view.set_running(True, other_running=False)  # 执行中删条目会被执行器存盘时写回
    assert not trim.delete_btn.isEnabled() and trim.delete_btn.toolTip()
    view.set_running(False, other_running=False)
    assert trim.delete_btn.isEnabled()

    pack.manifest.items = pack.manifest.items[:1]  # 最后一个条目：要删就删整个资源包
    view.show_pack(pack, CHARACTER, busy=False)
    assert not view.asset_card("hero", "trim").delete_btn.isEnabled()


def test_card_actions_show_only_on_hover_and_emit_item_and_step(qapp):
    view = PackDetail()
    view.show_pack(_character_pack(), CHARACTER, busy=False)
    card = view.asset_card("hero", "trim")
    assert card._actions.isHidden()  # 平时不显示
    card.enterEvent(QEnterEvent(QPointF(), QPointF(), QPointF()))
    assert not card._actions.isHidden()
    card.leaveEvent(QEvent(QEvent.Type.Leave))
    assert card._actions.isHidden()

    exported, deleted, selected = [], [], []
    view.asset_export_requested.connect(lambda item, step: exported.append((item, step)))
    view.asset_delete_requested.connect(deleted.append)
    view.asset_selected.connect(lambda item, step: selected.append((item, step)))
    card.export_btn.click()
    card.delete_btn.click()
    assert exported == [("hero", "trim")] and deleted == ["hero"]
    assert selected == []  # 点按钮不触发卡片的选中


def test_item_asset_count_is_the_cards_that_go_with_the_item(qapp):
    view = PackDetail()
    view.show_pack(_character_pack(), CHARACTER, busy=False)
    assert view.item_asset_count("hero") == 2  # 立绘 + 声线
    assert view.item_asset_count("nobody") == 0


def _answer(monkeypatch, button):
    monkeypatch.setattr(library_page.QMessageBox, "question", lambda *args, **kwargs: button)


def test_page_deletes_item_only_after_confirmation(qapp, monkeypatch):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    started = []
    page._start = lambda fn, *args, **kwargs: started.append((fn.__name__, args))

    _answer(monkeypatch, library_page.QMessageBox.StandardButton.No)
    page._delete_item("hero")
    assert started == []

    _answer(monkeypatch, library_page.QMessageBox.StandardButton.Yes)
    page._delete_item("nobody")  # 条目已经不在了：不发请求
    assert started == []
    page._delete_item("hero")
    assert started == [("delete_item", ("c1", "hero"))]


def test_page_shows_remaining_items_after_delete(qapp):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    page.detail.select_asset("hero", "trim")
    started = []
    page._start = lambda fn, *args, **kwargs: started.append(fn)

    remaining = _character_pack()
    remaining.manifest.items = [remaining.manifest.items[1]]
    page._on_item_deleted(remaining)
    assert [i.id for i in page.detail.pack.manifest.items] == ["mage"]
    assert page.detail.selected is None and page.detail.detail.key is None
    assert ("hero", "trim") not in page.detail._cards
    assert page.stack.currentWidget() is page.detail
    assert not any(fn is page._client.run_pack for fn in started)  # 删除不触发执行


def test_page_reports_delete_item_failure(qapp):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    page._on_item_deleted(PackResponse.from_error("条目 hero 正被资源包 剧情 引用"))
    assert "正被资源包 剧情 引用" in page.detail.error_label.text()
    assert [i.id for i in page.detail.pack.manifest.items] == ["hero", "mage"]


def test_export_button_needs_a_finished_deliverable(qapp):
    view = PackDetail()
    empty = _character_pack()
    empty.manifest.items[0].steps["trim"] = StepState(status="pending")
    view.show_pack(empty, CHARACTER, busy=False)
    assert not view.export_btn.isEnabled()  # 只有中间产物（generate）做完了不算
    view.show_pack(_character_pack(), CHARACTER, busy=False)
    assert view.export_btn.isEnabled()
    view.set_exporting(True)
    assert not view.export_btn.isEnabled()
    view.set_exporting(False)
    assert view.export_btn.isEnabled()


def test_export_file_name_strips_illegal_characters():
    assert export_file_name('主角:第1版/草稿?') == "主角_第1版_草稿_"
    assert export_file_name(" ... ") == "pack"


def test_page_exports_single_asset_with_its_own_format(qapp, monkeypatch, tmp_path):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    seen = []
    monkeypatch.setattr(library_page.QFileDialog, "getSaveFileName",
                        lambda *args: seen.append(args[2]) or (str(tmp_path / "hero"), ""))
    started = []
    page._start = lambda fn, *args, **kwargs: started.append((fn, args))
    page._export_asset("hero", "trim")
    fn, args = started[0]
    assert fn == page._client.export_asset
    assert args == ("c1", "hero", "trim", tmp_path / "hero.png")  # 单文件沿用产物的扩展名
    assert seen[0].endswith("_trim.png")  # 默认文件名：条目标题_步骤名

    page._export_asset("hero", "voice")  # 没做完：没有产物，不弹保存框也不发请求
    page._export_asset("nobody", "trim")
    assert len(started) == 1 and len(seen) == 1


def test_page_exports_layered_asset_as_zip(qapp, monkeypatch, tmp_path):
    page = _page(qapp)
    pack = _character_pack()
    pack.manifest.items[0].steps["trim"].outputs = ["hero/trim_1.png", "hero/trim_2.png"]
    page._on_packs(PackListResponse(packs=[pack]))
    page._open_pack("c1")
    monkeypatch.setattr(library_page.QFileDialog, "getSaveFileName",
                        lambda *args: (str(tmp_path / "layers"), ""))
    started = []
    page._start = lambda fn, *args, **kwargs: started.append(args)
    page._export_asset("hero", "trim")
    assert started[0] == ("c1", "hero", "trim", tmp_path / "layers.zip")  # 多文件打成 zip


def test_page_exports_to_chosen_zip(qapp, monkeypatch, tmp_path):
    page = _page(qapp)
    page._on_packs(PackListResponse(packs=[_character_pack()]))
    page._open_pack("c1")
    monkeypatch.setattr(library_page.QFileDialog, "getSaveFileName",
                        lambda *args: (str(tmp_path / "hero"), ""))
    started = []
    page._start = lambda fn, *args, **kwargs: started.append((fn, args, kwargs))
    page._export_current()
    fn, args, kwargs = started[0]
    assert fn == page._client.export_pack and args == ("c1", tmp_path / "hero.zip")
    assert not page.detail.export_btn.isEnabled()

    kwargs["on_ok"](BaseResponse.from_error("这个资源包还没有可导出的成品"))
    assert page.detail.export_btn.isEnabled()
    assert "还没有可导出的成品" in page.detail.error_label.text()


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


# ---- 场景：条件步骤（分层）----
SCENE = TemplateInfo(
    id="scene_background", type="scene", name="场景", cover="upscale",
    fields=[TemplateFieldInfo(id="length", default="1",
                              options=[FieldOption(value="1"), FieldOption(value="2")]),
            TemplateFieldInfo(id="layers", label="分层", default="0",
                              options=[FieldOption(value="0", label="不分层"),
                                       FieldOption(value="3", label="3 层"),
                                       FieldOption(value="4", label="4 层")])],
    steps=["generate", "upscale", "layer_encode", "layer", "layers"],
    step_details=[
        TemplateStepInfo(id="generate", type="image.generate"),
        TemplateStepInfo(id="upscale", type="image.upscale", deliverable=True,
                         inputs=["generate"]),
        TemplateStepInfo(id="layer_encode", type="image.layer_encode", when="layers"),
        TemplateStepInfo(id="layer", type="image.layer", when="layers",
                         inputs=["generate", "layer_encode"]),
        TemplateStepInfo(id="layers", type="image.upscale", deliverable=True, when="layers",
                         inputs=["layer"]),
    ])


def _scene_pack(layers: str = "", layered: bool = False) -> PackResponse:
    fields = {"layers": layers} if layers else {}
    steps = {"generate": StepState(status="done", outputs=["01/generate.png"]),
             "upscale": StepState(status="done", outputs=["01/upscale.png"])}
    if layered:
        steps["layer_encode"] = StepState(status="done", outputs=["01/layer_encode.pt"])
        steps["layer"] = StepState(status="done", outputs=["01/layer_1.png", "01/layer_2.png"])
        steps["layers"] = StepState(status="done", outputs=[
            "01/layers_1.png", "01/layers_2.png", "01/layers_3.png"])
    elif not layers:
        steps.update({s: StepState(status="skipped") for s in ("layer_encode", "layer", "layers")})
    item = CollectionItem(id="01", prompt="forest", fields=fields, steps=steps)
    manifest = Manifest(id="s1", name="森林", type="scene", template="scene_background",
                        items=[item])
    return PackResponse(manifest=manifest, media_base="/media/library/s1")


def test_skipped_steps_count_as_finished():
    pack = _scene_pack()
    item = pack.manifest.items[0]
    layers_step = SCENE.step_details[4]
    assert step_skipped(item, layers_step, SCENE)
    progress = pack_progress(pack.manifest, SCENE)
    # "不分层"的包做完整图就是完成，不能一直停在"待执行"
    assert (progress.done, progress.total, progress.status) == (1, 1, "done")
    assert step_fraction(pack.manifest, SCENE) == 100


def test_condition_is_checked_before_the_pack_ever_ran():
    # 刚建好还没执行：状态都是 pending，按字段默认值现算，也不该冒出图层卡片
    item = CollectionItem(id="01", prompt="forest")
    assert step_skipped(item, SCENE.step_details[4], SCENE)
    item.fields["layers"] = "4"
    assert not step_skipped(item, SCENE.step_details[4], SCENE)


PROPS_EITHER = TemplateInfo(
    id="scene_props", type="scene", name="场景道具", cover="trim",
    fields=[TemplateFieldInfo(id="sheet", default="0",
                              options=[FieldOption(value="0"), FieldOption(value="6")])],
    steps=["trim", "sheet_upscale"],
    step_details=[
        TemplateStepInfo(id="trim", type="image.trim", deliverable=True, when="!sheet"),
        TemplateStepInfo(id="sheet_upscale", type="image.upscale", deliverable=True,
                         when="sheet"),
    ])


def test_negated_condition_skips_the_other_chain():
    single, sheet = PROPS_EITHER.step_details
    item = CollectionItem(id="01", prompt="mushroom")
    assert not step_skipped(item, single, PROPS_EITHER) and step_skipped(item, sheet, PROPS_EITHER)
    item.fields["sheet"] = "6"
    assert step_skipped(item, single, PROPS_EITHER) and not step_skipped(item, sheet, PROPS_EITHER)


def test_cover_falls_back_when_the_cover_step_was_skipped():
    item = CollectionItem(id="01", prompt="mushrooms", fields={"sheet": "6"}, steps={
        "trim": StepState(status="skipped"),
        "sheet_upscale": StepState(status="done", outputs=["01/sheet_upscale_1.png",
                                                           "01/sheet_upscale_2.png"])})
    manifest = Manifest(id="p", type="scene", template="scene_props", items=[item])
    assert pack_cover(manifest, PROPS_EITHER) == "01/sheet_upscale_1.png"


def test_layer_thumbnail_is_the_bottom_layer():
    item = _scene_pack(layers="3", layered=True).manifest.items[0]
    assert latest_image(item, step_chain(SCENE, "layers")) == "01/layers_1.png"


def test_scene_detail_shows_layer_cards_only_when_layering(qapp):
    view = PackDetail()
    view.show_pack(_scene_pack(), SCENE, busy=False)
    assert set(view._cards) == {("01", "upscale")}
    view.show_pack(_scene_pack(layers="3", layered=True), SCENE, busy=False)
    assert set(view._cards) == {("01", "upscale"), ("01", "layers")}


def test_cover_offers_layering_afterwards(qapp):
    panel = DetailPanel()
    pack = _scene_pack()
    item = pack.manifest.items[0]
    requested = []
    panel.fields_update_requested.connect(
        lambda item_id, fields: requested.append((item_id, fields)))
    panel.show_asset(pack, SCENE, item, SCENE.step_details[1], busy=False)
    [(apply_btn, combo, current)] = panel._option_rows
    assert current == "0" and not apply_btn.isEnabled()  # 没改就不能点
    combo.setCurrentIndex(combo.findData("4"))
    assert apply_btn.isEnabled()
    # 轮询刷新同一个素材时不重建下拉，用户正在选的值不会被冲掉
    panel.show_asset(pack, SCENE, item, SCENE.step_details[1], busy=False)
    assert panel._option_rows[0][1] is combo and combo.currentData() == "4"
    panel.set_busy(True)
    assert not apply_btn.isEnabled()
    panel.set_busy(False)
    apply_btn.click()
    assert requested == [("01", {"layers": "4"})]


def test_layer_assets_have_no_layering_option(qapp):
    panel = DetailPanel()
    pack = _scene_pack(layers="3", layered=True)
    panel.show_asset(pack, SCENE, pack.manifest.items[0], SCENE.step_details[4], busy=False)
    assert panel._option_rows == []


def test_multi_image_preview_lists_layers_bottom_to_top(qapp):
    panel = DetailPanel()
    pack = _scene_pack(layers="3", layered=True)
    panel.show_asset(pack, SCENE, pack.manifest.items[0], SCENE.step_details[4], busy=False)
    panel.set_images(panel.key, [QPixmap(20, 10), QPixmap(20, 10), QPixmap(20, 10)])
    texts = [panel._preview.itemAt(i).widget().text() for i in range(panel._preview.count())
             if isinstance(panel._preview.itemAt(i).widget(), QLabel)
             and panel._preview.itemAt(i).widget().text()]
    assert texts[0] == "1 (bottom)" and "3 (top)" in texts and "2" in texts


def test_page_updates_fields_then_runs(qapp):
    page = _page(qapp)
    page._template_list = [SCENE]
    page._on_packs(PackListResponse(packs=[_scene_pack()]))
    page._open_pack("s1")
    started = []
    page._start = lambda fn, *args, **kwargs: started.append((fn, args, kwargs))
    page.detail.detail.fields_update_requested.emit("01", {"layers": "3"})
    fn, (pack_id, request), kwargs = started[0]
    assert fn == page._client.update_item_fields and pack_id == "s1"
    assert request.item_id == "01" and request.fields == {"layers": "3"}
    ran = []
    page._run_pack = ran.append
    kwargs["on_ok"](_scene_pack(layers="3"))
    assert ran == ["s1"]
