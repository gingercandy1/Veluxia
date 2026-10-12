import hashlib
import json
import re
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QMessageBox, QStackedWidget, QWidget

from src.app.client import ApiClient
from src.app.ui.base.anim import page_enter
from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.cards import ASSET_THUMB, PACK_THUMB, Card
from src.app.ui.library.detail_panel import PREVIEW_WIDTH, AssetKey
from src.app.ui.library.garden import GardenView
from src.app.ui.library.cut_dialog import CutDialog
from src.app.ui.library.pack_detail import RING_SOURCE_STEP, PackDetail, screen_key, seam_key
from src.app.ui.library.pack_form import PAGE_BACKGROUND, PackForm
from src.app.ui.library.pack_status import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    deliverables,
    fallback_template,
    item_title,
    latest_image,
    media_kind,
    pack_cover,
    step_chain,
)
from src.app.ui.library.ring_cards import PICKER_THUMB
from src.app.work import LibraryTaskWorker
from src.shared.schemas import (
    AddItemsRequest,
    ApproveStepRequest,
    CollectionItem,
    CreatePackRequest,
    DraftItemsRequest,
    DraftItemsResponse,
    PackListResponse,
    PackResponse,
    ResetStepRequest,
    RingLayerProgress,
    StyleListResponse,
    StylePreset,
    TemplateInfo,
    TemplateListResponse,
    UpdateItemFieldsRequest,
)
from src.shared.settings import PROJECT_ROOT

# 执行中轮询的间隔：进度只存在 manifest 里（ADR 0004），任务状态只用来判断结束
POLL_INTERVAL_MS = 2000
# 详情预览用的本地副本：音频播放器和看图浮层都只认本地文件
LIBRARY_CACHE_DIR = Path(PROJECT_ROOT) / "cache" / "library"
# 缩略图和详情预览缩好后的副本：原图常有几 MB 到十几 MB（精灵图、长条背景），
# 缩成几十 KB 存一份，下次打开不必再下载和解码原图
THUMB_CACHE_DIR = LIBRARY_CACHE_DIR / "thumbs"
# 同时下载并解码的图片数：打开资料库会为每个包、每张素材各起一个线程，
# 全部同时解码会占满 CPU，把界面线程的绘制挤得卡顿
_DECODE_SLOTS = threading.Semaphore(2)
# 会占用显存、要能中途停止的循环操作；其余（规划、采用、合成）是秒级的纯 CPU 操作
RING_GENERATING = ("generate", "redo", "layers", "extend", "edit")
# 自动切割是单次操作，但可以中途停止（按钮显示"停止分段"）
RING_STOPPABLE = (*RING_GENERATING, "cuts")
CHARACTER_TYPE = "character"
# 切页时新页面滑入的距离：只是一点方向感，滑太远会显得拖沓
PAGE_SLIDE = 12


def state_signature(item: CollectionItem, step_ids: list[str]) -> str:
    """产物文件名重跑后不变，用状态和耗时区分新旧内容，缩略图和本地副本据此失效。"""
    parts = []
    for step_id in step_ids:
        state = item.steps.get(step_id)
        if state is not None:
            parts.append(f"{step_id}:{state.status}:{state.meta.get('elapsed', '')}")
    return "|".join(parts)


def export_file_name(name: str) -> str:
    """资源包名当默认文件名：换掉 Windows 文件名里不允许的字符。"""
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return cleaned or "pack"


class LibraryPage(BaseWidget):
    """资料库：花园（分类卡片墙）→ 打开资源包（树 + 素材卡片 + 详情），另有新建表单。
    全部请求都在 LibraryTaskWorker 里执行；执行中每两秒拉一次资源包列表，所有视图都从它刷新。"""
    ring_progress = Signal(str, float)      # 接缝 key, 0~1：worker 线程发，主线程更新接缝连接器
    layer_progress = Signal(object)         # RingLayerProgress：worker 线程发，主线程给各段上色

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library_page")
        self._client = ApiClient.instance()
        self._workers: set[LibraryTaskWorker] = set()
        self._template_list: list[TemplateInfo] = []
        self._templates: dict[str, TemplateInfo] = {}
        self._packs: list[PackResponse] = []
        self._refreshing = False
        self._run_worker: LibraryTaskWorker | None = None
        self._run_pack_id = ""
        self._run_stop = threading.Event()
        self._ring_worker: LibraryTaskWorker | None = None
        self._ring_pack_id = ""
        # 环卡片缩略图已发起加载的 (包 id, {条目 id: token})；换包时清空，同一张图不重复加载
        self._ring_thumb_pack = ""
        self._ring_thumb_tokens: dict[str, str] = {}
        # 环预览的整图缓存：token（地址 + 内容版本）→ 解码后的图；每次刷新只保留用到的
        self._ring_image_cache: dict[str, QImage] = {}
        self._ring_action = ""
        self._seam_in_progress = ""     # 生成接缝时正在做的接缝 key，换了就说明上一条做完了
        self._ring_stop = threading.Event()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._refresh_packs)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()

        self.garden = GardenView()
        self.form = PackForm()
        self.add_form = PackForm(append=True)
        self.detail = PackDetail()
        for page in (self.garden, self.form, self.add_form, self.detail):
            self.stack.addWidget(page)
        layout.addWidget(self.stack)

        self.garden.pack_opened.connect(self._open_pack)
        self.garden.new_requested.connect(self._show_form)
        self.form.back_requested.connect(self._show_garden)
        self.form.create_requested.connect(self._create_pack)
        self.form.draft_requested.connect(lambda request: self._draft_items(self.form, request))
        self.add_form.back_requested.connect(self._back_to_detail)
        self.add_form.add_requested.connect(self._add_items)
        self.add_form.draft_requested.connect(
            lambda request: self._draft_items(self.add_form, request))
        self.detail.add_items_requested.connect(self._show_add_form)
        self.form.style_save_requested.connect(self._save_style)
        self.form.style_delete_requested.connect(self._delete_style)
        self.detail.style_sync_requested.connect(self._sync_style)
        self.detail.source_refresh_requested.connect(self._refresh_source)
        self.detail.back_requested.connect(self._show_garden)
        self.detail.pack_selected.connect(self._open_pack)
        self.detail.run_requested.connect(self._run_current)
        self.detail.stop_requested.connect(self._stop_run)
        self.detail.ring_view_requested.connect(self._load_ring_preview)
        self.detail.ring_plan_requested.connect(
            lambda item_ids: self._run_ring_action("plan", item_ids=item_ids))
        self.detail.ring_order_requested.connect(
            lambda order: self._run_ring_action("order", order=order))
        self.detail.ring_extend_requested.connect(
            lambda item_ids: self._run_ring_action("extend", item_ids=item_ids))
        self.detail.ring_generate_requested.connect(
            lambda keys: self._run_ring_action("generate", keys=keys))
        self.detail.ring_candidates_requested.connect(self._load_ring_candidates)
        self.detail.ring_cuts_auto_requested.connect(lambda: self._run_ring_action("cuts"))
        self.detail.ring_layers_requested.connect(
            lambda indices: self._run_ring_action("layers", indices=indices))
        self.detail.ring_edit_requested.connect(
            lambda target, prompt: self._run_ring_action("edit", target, prompt))
        self.detail.ring_version_requested.connect(
            lambda target, index: self._run_ring_action("version", target, index=index))
        self.detail.ring_images_requested.connect(self._load_ring_images)
        self.detail.ring_layers_progressed.connect(self._refresh_layer_status)
        self.detail.ring_layers_merge_requested.connect(
            lambda segment, ids: self._run_ring_action("merge", segment=segment, layer_ids=ids))
        self.detail.ring_merged_delete_requested.connect(
            lambda segment, layer_id: self._run_ring_action(
                "delete_merged", segment=segment, layer_ids=[layer_id]))
        self.detail.ring_layer_opened.connect(self._open_ring_file)
        self.ring_progress.connect(self._on_ring_progress)
        self.layer_progress.connect(self.detail.set_layer_progress)
        self.detail.ring_adopt_requested.connect(
            lambda key, candidate_id: self._run_ring_action("adopt", key, candidate_id))
        self.detail.ring_redo_requested.connect(
            lambda key, extra: self._run_ring_action("redo", key, extra))
        self.detail.delete_requested.connect(self._delete_current)
        self.detail.export_requested.connect(self._export_current)
        self.detail.asset_selected.connect(self._on_asset_selected)
        self.detail.asset_opened.connect(self._on_asset_opened)
        self.detail.detail.reset_requested.connect(self._reset_step)
        self.detail.detail.approve_requested.connect(self._approve_step)
        self.detail.detail.fields_update_requested.connect(self._update_fields)
        self.detail.asset_delete_requested.connect(self._delete_item)
        self.detail.asset_export_requested.connect(self._export_asset)

    # ---- 对外 ----
    def activate(self):
        """每次进入页面时刷新：资源包可能在别处被改动（例如手动删了文件夹）。"""
        self.garden.set_loading(True)
        self._start(self._client.list_templates, on_ok=self._on_templates,
                    on_error=lambda message: self._on_templates(
                        TemplateListResponse.from_error(message)))
        self._start(self._client.list_styles, on_ok=self._on_styles,
                    on_error=lambda message: self._on_styles(StyleListResponse.from_error(message)))

    def shutdown(self):
        """关窗口时调用：停止执行并等 worker 退出，否则运行中的 QThread 被回收会直接崩溃。"""
        self._poll_timer.stop()
        self._run_stop.set()
        self._ring_stop.set()
        for worker in list(self._workers):
            worker.wait(3000)

    # ---- 页面切换 ----
    def _switch_to(self, page: QWidget, forward: bool):
        """切页时淡入并滑动一小段：前进从右边进来，返回从左边进来。"""
        if self.stack.currentWidget() is page:
            return
        self.stack.setCurrentWidget(page)
        page_enter(page, PAGE_BACKGROUND, PAGE_SLIDE if forward else -PAGE_SLIDE)

    def _show_garden(self):
        self._switch_to(self.garden, forward=False)

    def _show_form(self, pack_type: str = ""):
        if pack_type:
            self.form.preselect_category(pack_type)
        self.form.set_error("")
        self._switch_to(self.form, forward=True)

    def _open_pack(self, pack_id: str, forward: bool = True):
        pack = self._find_pack(pack_id)
        if pack is None:
            return
        self._switch_to(self.detail, forward=forward)
        self._show_detail(pack)

    def _show_add_form(self):
        pack = self.detail.pack
        if pack is None:
            return
        self.add_form.set_target(pack.manifest, self._templates[pack.manifest.template])
        self.add_form.set_error("")
        self._switch_to(self.add_form, forward=True)

    def _back_to_detail(self):
        target = self.add_form.target_id()
        if self._find_pack(target) is None:
            self._show_garden()  # 追加期间资源包在别处被删了
            return
        self._open_pack(target, forward=False)

    def _current_pack(self) -> PackResponse | None:
        if self.stack.currentWidget() is not self.detail or self.detail.pack is None:
            return None
        return self._find_pack(self.detail.pack.manifest.id)

    def _find_pack(self, pack_id: str) -> PackResponse | None:
        return next((p for p in self._packs if p.manifest.id == pack_id), None)

    # ---- 数据刷新 ----
    def _on_templates(self, response: TemplateListResponse):
        if not response.ok:
            self.garden.set_loading(False)
            self.garden.set_error(self.tr("Failed to load templates: {0}").format(response.error))
            return
        self._template_list = response.templates
        self.form.set_templates(response.templates)
        self._refresh_packs()

    def _refresh_packs(self):
        if self._refreshing:
            return
        self._refreshing = True
        self._start(self._client.list_packs, on_ok=self._on_packs, on_error=self._on_packs_error)

    def _on_packs_error(self, message: str):
        self._on_packs(PackListResponse.from_error(message))

    def _on_packs(self, response: PackListResponse):
        self._refreshing = False
        self.garden.set_loading(False)
        if not response.ok:
            self.garden.set_error(self.tr("Failed to load packs: {0}").format(response.error))
            self._update_polling()
            return
        self.garden.set_error("")
        self._packs = response.packs
        known = {t.id: t for t in self._template_list}
        # 模板已下线的旧资源包也要能显示，按 manifest 里的步骤拼一个
        self._templates = {p.manifest.template: known.get(p.manifest.template)
                           or fallback_template(p.manifest) for p in self._packs}
        self.garden.set_packs(self._packs, self._templates)
        self.detail.set_tree(self._packs, self._templates)
        self.form.set_characters(self._character_options())
        self._load_covers()
        current = self._current_pack()
        if current is not None:
            self._show_detail(current)
        elif self.detail.pack is not None and self._find_pack(self.detail.pack.manifest.id) is None:
            self._show_garden()  # 打开的资源包在别处被删了
        self._update_polling()

    def _character_options(self) -> list[tuple[str, str]]:
        options = []
        for pack in self._packs:
            # 动作包也归在角色分类下，但它的条目是动作，不能再被当成角色绑定
            if pack.manifest.type != CHARACTER_TYPE or pack.manifest.source:
                continue
            for item in pack.manifest.items:
                label = f"{pack.manifest.name or pack.manifest.id} / {item_title(item)}"
                options.append((label, f"{pack.manifest.id}/{item.id}"))
        return options

    def _show_detail(self, pack: PackResponse):
        template = self._templates[pack.manifest.template]
        if self.detail.show_pack(pack, template, busy=self._is_running(pack.manifest.id)):
            self._load_preview()
        self._update_run_state()
        self._load_asset_thumbnails(pack, template)
        self._load_ring_thumbnails(pack, template)

    # ---- 执行状态 ----
    def _running_pack_ids(self) -> set[str]:
        ids = {p.manifest.id for p in self._packs if p.running}
        if self._run_worker is not None:
            ids.add(self._run_pack_id)
        if self._ring_worker is not None and self._ring_action in RING_GENERATING:
            ids.add(self._ring_pack_id)
        return ids

    def _is_busy(self) -> bool:
        return bool(self._running_pack_ids())

    def _is_running(self, pack_id: str) -> bool:
        return pack_id in self._running_pack_ids()

    def _update_run_state(self):
        pack = self.detail.pack
        if pack is None:
            return
        running = self._running_pack_ids()
        here = pack.manifest.id in running
        # 单驻留下同时只跑一个资源包，另一个包提交了也只会在后端排队等模型
        self.detail.set_running(here, other_running=bool(running - {pack.manifest.id}))
        if self._run_worker is not None and self._run_stop.is_set():
            self.detail.stop_btn.setEnabled(False)

    def _update_polling(self):
        if self._is_busy():
            self._poll_timer.start()
        else:
            self._poll_timer.stop()

    def _run_current(self):
        pack = self.detail.pack
        if pack is None or self._is_busy():
            return
        self._run_pack(pack.manifest.id)

    def _run_pack(self, pack_id: str):
        self.detail.set_error("")
        self._run_pack_id = pack_id
        self._run_stop = threading.Event()
        self._run_worker = self._start(
            self._client.run_pack, pack_id, stop_event=self._run_stop,
            on_ok=self._on_run_finished, on_error=self._on_run_error)
        self._update_run_state()
        self._update_polling()
        self._refresh_packs()

    def _stop_run(self):
        if self._ring_worker is not None and self._ring_action in RING_STOPPABLE:
            self._ring_stop.set()
            self.detail.stop_btn.setEnabled(False)
            return
        self._run_stop.set()
        self.detail.stop_btn.setEnabled(False)

    def _on_run_finished(self, response: PackResponse):
        stopped = self._run_stop.is_set()
        self._run_worker = None
        if not response.ok and not stopped:
            self.detail.set_error(self.tr("Run failed: {0}").format(response.error))
        self._update_run_state()
        self._refresh_packs()

    def _on_run_error(self, message: str):
        self._on_run_finished(PackResponse.from_error(message))

    def _load_ring_preview(self):
        pack = self.detail.pack
        if pack is None or pack.manifest.type != "scene" or len(pack.manifest.items) < 2:
            return
        self._start(self._fetch_ring_preview, pack,
                    on_ok=self._on_ring_preview_loaded, on_error=self._on_worker_error)

    def _fetch_ring_preview(self, pack: PackResponse):
        status = self._client.ring_status(pack.manifest.id)
        if not status.ok:
            raise RuntimeError(status.error or self.tr("Could not load ring status."))
        items = {item.id: item for item in pack.manifest.items}
        before, after = [], []
        rendered_count = 0
        # 同一屏可以在环里出现多次：原图只取一次
        originals: dict[str, QImage] = {}
        used: set[str] = set()
        count = len(status.order)
        for index, item_id in enumerate(status.order):
            item = items.get(item_id)
            state = item.steps.get("generate") if item else None
            if state is None or state.status != "done" or not state.outputs:
                raise ValueError(self.tr("Screen {0} has no finished source image.").format(item_id))
            source_sig = state_signature(item, [RING_SOURCE_STEP])
            # 屏当前用的图可能是外扩 / 改过的版本；旧后端没给 screen_files 时用 generate 的产物
            screen_file = (status.screen_files[index] if index < len(status.screen_files)
                           and status.screen_files[index] else state.outputs[0])
            if item_id not in originals:
                originals[item_id] = self._fetch_cached_ring_image(
                    pack, screen_file, f"{source_sig}|{screen_file}", used)
            before.append(originals[item_id])
            rendered_path = status.rendered[index] if index < len(status.rendered) else ""
            if rendered_path:
                # 合成图由这一屏和左右两屏的当前图、两条接缝采用的候选决定：这些没变就是同一张图
                left_key = f"{status.order[index - 1]}>{item_id}"
                right_key = f"{item_id}>{status.order[(index + 1) % count]}"
                signature = "|".join((
                    source_sig, screen_file, status.order[index - 1],
                    self._neighbour_file(status, index - 1),
                    status.order[(index + 1) % count],
                    self._neighbour_file(status, (index + 1) % count),
                    self._adopted_id(status, left_key), self._adopted_id(status, right_key)))
                after.append(
                    self._fetch_cached_ring_image(pack, rendered_path, signature, used))
                rendered_count += 1
            else:
                after.append(originals[item_id])
        # 只留这次用到的图：换了顺序或重做后，旧的合成图不再占内存
        for token in set(self._ring_image_cache) - used:
            self._ring_image_cache.pop(token, None)

        candidate_count = sum(seam.has_candidates for seam in status.seams.values())
        stale_count = sum(seam.stale for seam in status.seams.values())
        text = self.tr("{0} screens · {1} seams with candidates · {2} stale · {3} rendered")
        status_text = text.format(len(status.order), candidate_count, stale_count, rendered_count)
        if rendered_count < len(status.order):
            status_text += "\n" + self.tr("Not rendered yet")
        return before, after, status_text, status

    def _emit_ring_progress(self, progress: dict):
        """在 worker 线程里被调用：经信号排队回主线程再更新连接器，不直接碰控件。"""
        self.ring_progress.emit(progress["key"], float(progress["fraction"]))

    def _on_ring_progress(self, key: str, fraction: float):
        if self._ring_action == "extend":
            self.detail.set_node_progress(screen_key(key), fraction)
        elif self._ring_action == "edit":
            self.detail.set_node_progress(key, fraction)
        else:
            self.detail.set_ring_progress(key, fraction)
        # 换到下一条接缝（外扩时是下一屏）说明上一条已经存盘并合成：刷新预览，不必等全部做完
        if self._seam_in_progress and key != self._seam_in_progress:
            self._load_ring_preview()
        self._seam_in_progress = key

    def _emit_layer_progress(self, progress: dict):
        """同上，分层任务的进度。"""
        self.layer_progress.emit(RingLayerProgress(**progress))

    def _redo_seam(self, pack_id: str, key: str, extra: str, stop_event: threading.Event):
        """重做一个接缝并立刻采用、合成：用户点"重做"就是想马上在滚动条里看到新结果，
        不满意再用候选下拉换回旧的。"""
        response = self._client.generate_ring(
            pack_id, stop_event=stop_event, keys=[key], extra_prompt=extra or None,
            on_progress=self._emit_ring_progress)
        if not response.ok:
            return response
        status = self._client.ring_status(pack_id)
        if not status.ok:
            return PackResponse.from_error(status.error or self.tr("Could not load ring status."))
        seam = status.seams.get(key)
        if seam is None or not seam.candidates:
            return PackResponse.from_error(self.tr("The seam produced no candidate."))
        return self._adopt_seam(pack_id, key, seam.candidates[-1].id)

    def _adopt_seam(self, pack_id: str, key: str, candidate_id: str):
        response = self._client.adopt_ring(pack_id, key, candidate_id)
        if not response.ok:
            return response
        return self._client.render_ring(pack_id)

    def _set_order_and_render(self, pack_id: str, order: list[str]):
        """改了顺序立刻合成：邻居变了的屏要换上对应的接缝补丁，否则预览里看不到新位置的效果。"""
        response = self._client.set_ring_order(pack_id, order)
        if not response.ok:
            return response
        return self._client.render_ring(pack_id)

    def _load_ring_candidates(self, key: str):
        seam = self.detail.ring_seam(key)
        if seam is None or not seam.candidates:
            return
        self._start(self._fetch_candidate_images, list(seam.candidates),
                    on_ok=lambda images: self.detail.set_ring_candidate_images(key, images),
                    on_error=self._on_worker_error)

    def _refresh_layer_status(self):
        """分层做完一段就取最新状态，让这一段的图层马上出现在分层页。"""
        pack = self.detail.pack
        if pack is None:
            return
        pack_id = pack.manifest.id

        def apply(status):
            if not status.ok:
                self.detail.set_error(self.tr("Ring operation failed: {0}").format(status.error))
            elif self.detail.pack is not None and self.detail.pack.manifest.id == pack_id:
                self.detail.apply_layer_status(status)

        self._start(self._client.ring_status, pack_id, on_ok=apply,
                    on_error=self._on_worker_error)

    def _load_ring_images(self, paths: list[str]):
        """画布和树要的段图 / 图层（相对路径）。"""
        pack = self.detail.pack
        if pack is None or not paths:
            return
        pack_id = pack.manifest.id

        def apply(images: dict[str, QImage]):
            # 取图期间用户可能已经换了包：别把别的包的图塞进当前画布
            if self.detail.pack is not None and self.detail.pack.manifest.id == pack_id:
                self.detail.set_segment_images(images)

        self._start(self._fetch_segment_images, pack, paths,
                    on_ok=apply, on_error=self._on_worker_error)

    def _fetch_segment_images(self, pack: PackResponse, paths: list[str]) -> dict[str, QImage]:
        return {relative: self._fetch_ring_image(pack, relative) for relative in paths}

    def _open_ring_file(self, relative: str):
        pack = self.detail.pack
        if pack is None:
            return
        # 段图、图层重做后文件名不变，manifest 里也没有能区分新旧的签名：每次放大都重新下载，
        # 一张图约 1MB，不值得为它记版本
        self._start(self._download, pack, relative, uuid.uuid4().hex,
                    on_ok=lambda path: self.detail.show_image(str(path)),
                    on_error=self._on_worker_error)

    def _fetch_candidate_images(self, candidates) -> dict[str, QImage]:
        images = {}
        for candidate in candidates:
            with _DECODE_SLOTS:
                images[candidate.id] = QImage.fromData(self._client.fetch_media(candidate.file))
        return images

    @staticmethod
    def _neighbour_file(status, index: int) -> str:
        return status.screen_files[index] if index < len(status.screen_files) else ""

    @staticmethod
    def _adopted_id(status, key: str) -> str:
        seam = status.seams.get(key)
        return seam.adopted if seam is not None else ""

    def _fetch_ring_image(self, pack: PackResponse, relative: str) -> QImage:
        url = f"{pack.media_base}/{relative}"
        with _DECODE_SLOTS:
            image = QImage.fromData(self._client.fetch_media(url))
        if image.isNull():
            raise ValueError(self.tr("Could not decode image: {0}").format(url))
        return image

    def _fetch_cached_ring_image(self, pack: PackResponse, relative: str, signature: str,
                                 used: set[str]) -> QImage:
        """环预览用的整图。signature 是内容版本（同名文件重写后它会变）：版本没变就直接用
        缓存，生成接缝时每做完一条都会刷新预览，不能每次都把所有屏重下一遍。"""
        token = f"{pack.media_base}/{relative}#{signature}"
        used.add(token)
        image = self._ring_image_cache.get(token)
        if image is None:
            image = self._ring_image_cache[token] = self._fetch_ring_image(pack, relative)
        return image

    def _on_ring_preview_loaded(self, result):
        self.detail.set_ring(*result)

    def _review_suggested_cuts(self):
        """自动切割做完：取最新状态，弹对话框让用户看过、调整后再确认。"""
        pack = self.detail.pack
        if pack is None:
            return
        pack_id = pack.manifest.id
        self._start(self._fetch_ring_preview, pack,
                    on_ok=lambda result: self._on_cuts_preview_loaded(pack_id, result),
                    on_error=self._on_worker_error)

    def _on_cuts_preview_loaded(self, pack_id: str, result):
        if self.detail.pack is None or self.detail.pack.manifest.id != pack_id:
            return
        self.detail.set_ring(*result)
        status = result[3]
        if not status.suggested_cuts:
            return
        dialog = CutDialog(self.detail.ring_images(), list(status.suggested_cuts),
                           self._cuts_warning(status), self.detail)
        # 确定才提交（并清空所有分层）；取消什么都不变
        if dialog.exec():
            self._run_ring_action("confirm", cuts=dialog.cuts())

    def _cuts_warning(self, status) -> str:
        """接缝缺失或过期时，合成图会在那里变化，切点可能要跟着挪。"""
        order = status.order
        keys = dict.fromkeys(seam_key(order[index], order[(index + 1) % len(order)])
                             for index in range(len(order)))
        not_ready = 0
        for key in keys:
            seam = status.seams.get(key)
            if seam is None or not (seam.candidates or seam.has_candidates) or seam.stale:
                not_ready += 1
        if not not_ready:
            return ""
        return self.tr("{0} seams are not ready; the cuts may be affected.").format(not_ready)

    def _run_ring_action(self, action: str, key: str = "", argument: str = "",
                         item_ids: list[str] | None = None, cuts: list[int] | None = None,
                         indices: list[int] | None = None, order: list[str] | None = None,
                         segment: int = 0, layer_ids: list[str] | None = None,
                         keys: list[str] | None = None, index: int = 0):
        """argument：redo 时是追加的提示词，adopt 时是候选 id，edit 时是改图指令；
        key：redo / adopt 的接缝，edit / version 的目标键；
        keys：generate 时要追加候选的接缝，空则只补缺失和过期的；
        item_ids：plan 时参与循环的条目，extend 时要外扩的屏（空 = 全部）；
        order：order 时手动排的顺序（可重复同一屏）；index：version 时的版本序号（-1 = 原图）；
        cuts：confirm 时用户确认的切点；indices：layers 时要分层的段；
        segment / layer_ids：merge 时的段和要合并的层，delete_merged 时的段和要删的合并层。"""
        pack = self.detail.pack
        if pack is None or self._ring_worker is not None:
            return
        self.detail.set_error("")
        self._ring_action = action
        self._ring_pack_id = pack.manifest.id
        self._ring_stop = threading.Event()
        self._seam_in_progress = ""
        kwargs = {}
        if action == "generate":
            fn, args, kwargs = self._client.generate_ring, (self._ring_pack_id,), {
                "stop_event": self._ring_stop, "keys": keys or None,
                "on_progress": self._emit_ring_progress}
        elif action == "extend":
            fn, args, kwargs = self._client.extend_ring, (self._ring_pack_id,), {
                "item_ids": item_ids or None, "stop_event": self._ring_stop,
                "on_progress": self._emit_ring_progress}
        elif action == "edit":
            fn, args, kwargs = self._client.edit_ring_target, (
                self._ring_pack_id, key, argument), {
                "stop_event": self._ring_stop, "on_progress": self._emit_ring_progress}
        elif action == "version":
            fn, args = self._client.select_ring_version, (self._ring_pack_id, key, index)
        elif action == "redo":
            fn, args = self._redo_seam, (self._ring_pack_id, key, argument, self._ring_stop)
        elif action == "adopt":
            fn, args = self._adopt_seam, (self._ring_pack_id, key, argument)
        elif action == "plan":
            fn, args = self._client.plan_ring, (self._ring_pack_id, item_ids)
        elif action == "order":
            fn, args = self._set_order_and_render, (self._ring_pack_id, order or [])
        elif action == "cuts":
            fn, args = self._client.auto_ring_cuts, (self._ring_pack_id, self._ring_stop)
        elif action == "confirm":
            fn, args = self._client.confirm_ring_cuts, (self._ring_pack_id, cuts or [])
        elif action == "merge":
            fn, args = self._client.merge_ring_layers, (
                self._ring_pack_id, segment, layer_ids or [])
        elif action == "delete_merged":
            fn, args = self._client.delete_merged_layer, (
                self._ring_pack_id, segment, (layer_ids or [""])[0])
        elif action == "layers":
            fn, args, kwargs = self._client.layer_ring, (self._ring_pack_id,), {
                "stop_event": self._ring_stop, "indices": indices or None,
                "on_progress": self._emit_layer_progress}
        else:
            raise ValueError(f"未知的环操作：{action}")
        if action in RING_GENERATING:
            self.detail.set_ring_running(True, action=action)
        else:
            self.detail.set_ring_operation(action)
        self._ring_worker = self._start(
            fn, *args, on_ok=self._on_ring_action_finished,
            on_error=self._on_ring_action_error, **kwargs)
        self._update_run_state()
        if action in RING_GENERATING:
            self._update_polling()
            self._refresh_packs()

    def _on_ring_action_finished(self, response: PackResponse):
        pack_id = self._ring_pack_id
        action = self._ring_action
        self._ring_worker = None
        self.detail.set_ring_running(False)
        self.detail.set_ring_operation(None)
        if not response.ok:
            if not self._ring_stop.is_set():
                self.detail.set_error(self.tr("Ring operation failed: {0}").format(response.error))
            if action in ("generate", "layers", "order", "extend", "edit") and self.detail.pack \
                    and self.detail.pack.manifest.id == pack_id:
                # 接缝 / 外扩 / 分层每做完一个就存盘：停止或中途失败后，已完成的也要显示出来；
                # 改顺序失败时卡片条要回到后端实际的环
                self.detail.clear_segment_images()
                self._load_ring_preview()
        else:
            self._packs = [
                response if pack.manifest.id == pack_id else pack for pack in self._packs
            ]
            if self.detail.pack and self.detail.pack.manifest.id == pack_id:
                self._show_detail(response)
                if action in ("layers", "confirm", "merge", "delete_merged", "extend", "edit",
                              "version"):
                    # 重做的段、重新导出的段图和改过的层沿用原文件名，删掉的合并层编号还会被
                    # 下一次合并重新用上：缓存的缩略图要作废
                    self.detail.clear_segment_images()
                if action in ("generate", "redo", "adopt", "cuts", "confirm", "order", "extend"):
                    # 刚改过接缝或切段：直接看拼接后的效果（切点就是在合成图上算的）
                    self.detail.ring_after_btn.setChecked(True)
                if action == "cuts":
                    self._review_suggested_cuts()
                else:
                    self._load_ring_preview()
        self._ring_action = ""
        self._update_run_state()
        self._update_polling()

    def _on_ring_action_error(self, message: str):
        self._ring_worker = None
        self.detail.set_ring_running(False)
        self.detail.set_ring_operation(None)
        self.detail.set_error(self.tr("Ring operation failed: {0}").format(message))
        self._ring_action = ""
        self._update_run_state()
        self._update_polling()

    # ---- 新建 / 删除 ----
    def _create_pack(self, request: CreatePackRequest):
        self.form.set_busy(True)
        self._start(self._client.create_pack, request, on_ok=self._on_created,
                    on_error=lambda message: self._on_created(PackResponse.from_error(message)))

    def _on_created(self, response: PackResponse):
        self.form.set_busy(False)
        if not response.ok:
            self.form.set_error(self.tr("Failed to create pack: {0}").format(response.error))
            return
        self.form.reset()
        self._packs = [*self._packs, response]
        self._on_packs(PackListResponse(packs=self._packs))
        self._open_pack(response.manifest.id)
        # 建包的意图就是生成：直接开始执行；别的资源包正在执行时只打开，由用户稍后再点
        if not self._is_busy():
            self._run_pack(response.manifest.id)

    def _add_items(self, pack_id: str, request: AddItemsRequest):
        self.add_form.set_busy(True)
        self._start(self._client.add_items, pack_id, request, on_ok=self._on_items_added,
                    on_error=lambda message: self._on_items_added(PackResponse.from_error(message)))

    def _on_items_added(self, response: PackResponse):
        self.add_form.set_busy(False)
        if not response.ok:
            self.add_form.set_error(self.tr("Failed to add items: {0}").format(response.error))
            return
        self.add_form.reset()
        self._packs = [response if p.manifest.id == response.manifest.id else p
                       for p in self._packs]
        self._on_packs(PackListResponse(packs=self._packs))
        self._open_pack(response.manifest.id, forward=False)
        # 和建包一样，加条目的意图就是生成；已完成的步骤执行器会跳过，只跑新条目
        if not self._is_busy():
            self._run_pack(response.manifest.id)

    def _draft_items(self, form: PackForm, request: DraftItemsRequest):
        form.set_drafting(True)
        self._start(self._client.draft_items, request,
                    on_ok=lambda response: self._on_drafted(form, request.template, response),
                    on_error=lambda message: self._on_drafted(
                        form, request.template, DraftItemsResponse.from_error(message)))

    def _on_drafted(self, form: PackForm, template_id: str, response: DraftItemsResponse):
        form.set_drafting(False)
        if not response.ok:
            form.set_error(self.tr("AI draft failed: {0}").format(response.error))
            return
        form.apply_drafts(template_id, response.items)

    def _delete_current(self):
        pack = self.detail.pack
        if pack is None:
            return
        manifest = pack.manifest
        reply = QMessageBox.question(
            self,
            self.tr("Delete pack"),
            self.tr("Delete pack \"{0}\" and all its generated files? This cannot be undone.")
            .format(manifest.name or manifest.id),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start(self._client.delete_pack, manifest.id, on_ok=self._on_deleted)

    def _delete_item(self, item_id: str):
        pack = self.detail.pack
        if pack is None:
            return
        item = next((i for i in pack.manifest.items if i.id == item_id), None)
        if item is None:
            return
        reply = QMessageBox.question(
            self,
            self.tr("Delete item"),
            self.tr("Delete item \"{0}\" with its {1} asset(s) and all generated files? "
                    "This cannot be undone.")
            .format(item_title(item), self.detail.item_asset_count(item_id)),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start(self._client.delete_item, pack.manifest.id, item_id,
                    on_ok=self._on_item_deleted,
                    on_error=lambda message: self._on_item_deleted(PackResponse.from_error(message)))

    def _on_item_deleted(self, response: PackResponse):
        if not response.ok:
            self.detail.set_error(self.tr("Failed to delete item: {0}").format(response.error))
            return
        # 和追加条目不同，删除不触发执行：剩下的条目都是用户已经看过的
        self._packs = [response if p.manifest.id == response.manifest.id else p
                       for p in self._packs]
        self._on_packs(PackListResponse(packs=self._packs))

    # ---- 导出 ----
    def _export_current(self):
        pack = self.detail.pack
        if pack is None:
            return
        manifest = pack.manifest
        folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        default = Path(folder or Path.home()) / f"{export_file_name(manifest.name or manifest.id)}.zip"
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export pack"), str(default), self.tr("Zip archive (*.zip)"))
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".zip":
            target = target.with_name(target.name + ".zip")
        self.detail.set_error("")
        self.detail.set_exporting(True)
        self._start(self._client.export_pack, manifest.id, target,
                    on_ok=lambda response: self._on_exported(response, target),
                    on_error=lambda message: self._on_exported(
                        PackResponse.from_error(message), target))

    def _export_asset(self, item_id: str, step_id: str):
        """导出一张素材卡片：一个文件按原格式另存，多个文件（如分层）打成 zip。"""
        pack = self.detail.pack
        item = next((i for i in pack.manifest.items if i.id == item_id), None) if pack else None
        state = item.steps.get(step_id) if item else None
        if state is None or not state.outputs:
            return
        step = next((s for s in self._templates[pack.manifest.template].step_details
                     if s.id == step_id), None)
        name = export_file_name(f"{item_title(item)}_{step.label or step_id}")
        suffix = Path(state.outputs[0]).suffix if len(state.outputs) == 1 else ".zip"
        folder = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DownloadLocation)
        default = Path(folder or Path.home()) / f"{name}{suffix}"
        kind = self.tr("Zip archive (*.zip)") if suffix == ".zip" \
            else self.tr("Asset (*{0})").format(suffix)
        path, _ = QFileDialog.getSaveFileName(self, self.tr("Export asset"), str(default), kind)
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != suffix:
            target = target.with_name(target.name + suffix)
        self.detail.set_error("")
        self._start(self._client.export_asset, pack.manifest.id, item_id, step_id, target,
                    on_ok=lambda response: self._on_exported(response, target),
                    on_error=lambda message: self._on_exported(
                        PackResponse.from_error(message), target))

    def _on_exported(self, response, target: Path):
        self.detail.set_exporting(False)
        if not response.ok:
            self.detail.set_error(self.tr("Export failed: {0}").format(response.error))
            return
        box = QMessageBox(QMessageBox.Icon.Information, self.tr("Export pack"),
                          self.tr("Saved to {0}").format(target), parent=self)
        open_btn = box.addButton(self.tr("Show in folder"), QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.parent)))

    def _on_deleted(self, response):
        if not response.ok:
            self.detail.set_error(self.tr("Failed to delete pack: {0}").format(response.error))
            return
        self._show_garden()
        self._refresh_packs()

    # ---- 风格预设 ----
    def _save_style(self, preset: StylePreset):
        self._start(self._client.save_style, preset,
                    on_ok=lambda response: self._on_styles(response, select_name=preset.name),
                    on_error=lambda message: self._on_styles(StyleListResponse.from_error(message)))

    def _delete_style(self, style_id: str):
        self._start(self._client.delete_style, style_id, on_ok=self._on_styles,
                    on_error=lambda message: self._on_styles(StyleListResponse.from_error(message)))

    def _on_styles(self, response: StyleListResponse, select_name: str = ""):
        if not response.ok:
            self.form.set_error(self.tr("Style presets: {0}").format(response.error))
            return
        self.form.set_styles(response.styles, select_name)
        self.detail.set_styles(response.styles)

    def _sync_style(self):
        pack = self.detail.pack
        if pack is None:
            return

        def handle(response: PackResponse):
            if not response.ok:
                self.detail.set_error(response.error or "")
                return
            self._refresh_packs()
        self._start(self._client.sync_pack_style, pack.manifest.id, on_ok=handle,
                    on_error=lambda message: handle(PackResponse.from_error(message)))

    # ---- 审阅 / 重做 ----
    def _approve_step(self, item_id: str, step_id: str, content: str | None):
        pack = self.detail.pack
        if pack is None:
            return
        request = ApproveStepRequest(item_id=item_id, step_id=step_id, content=content)
        self._start(self._client.approve_step, pack.manifest.id, request,
                    on_ok=self._after_update(pack.manifest.id))

    def _refresh_source(self):
        pack = self.detail.pack
        if pack is None:
            return
        self._start(self._client.refresh_pack_source, pack.manifest.id,
                    on_ok=self._after_update(pack.manifest.id))

    def _update_fields(self, item_id: str, fields: dict):
        """事后改选项（如补做分层）：后端只把受影响的步骤标成待重做，改完直接接着执行。"""
        pack = self.detail.pack
        if pack is None:
            return
        request = UpdateItemFieldsRequest(item_id=item_id, fields=fields)
        self._start(self._client.update_item_fields, pack.manifest.id, request,
                    on_ok=self._after_update(pack.manifest.id))

    def _reset_step(self, item_id: str, step_id: str):
        pack = self.detail.pack
        if pack is None:
            return
        request = ResetStepRequest(item_id=item_id, step_id=step_id)
        self._start(self._client.reset_step, pack.manifest.id, request,
                    on_ok=self._after_update(pack.manifest.id))

    def _after_update(self, pack_id: str) -> Callable[[PackResponse], None]:
        """确认剧本、重做某步之后直接接着执行：用户的意图就是往下跑，不必再点一次执行。"""
        def handle(response: PackResponse):
            if not response.ok:
                self.detail.set_error(response.error or "")
                return
            if self._is_busy():
                self._refresh_packs()
                return
            self._run_pack(pack_id)
        return handle

    # ---- 缩略图 ----
    def _load_covers(self):
        for pack in self._packs:
            template = self._templates[pack.manifest.template]
            relative = pack_cover(pack.manifest, template)
            if not relative:
                continue
            item_id = relative.split("/", 1)[0]
            item = next((i for i in pack.manifest.items if i.id == item_id), None)
            signature = state_signature(item, template.steps) if item else ""
            self._load_thumbnail(self.garden.card(pack.manifest.id),
                                 f"{pack.media_base}/{relative}",
                                 signature, PACK_THUMB,
                                 lambda pixmap, pack_id=pack.manifest.id:
                                 self.garden.set_thumbnail(pack_id, pixmap))

    def _load_asset_thumbnails(self, pack: PackResponse, template: TemplateInfo):
        for step in deliverables(template):
            if media_kind(step.type) != "image":
                continue
            chain = step_chain(template, step.id)
            # 附加视图（分层）的缩略图存在原图卡片里，换用独立的 token 记是否已加载
            layered = bool(step.overlay_of)
            for item in pack.manifest.items:
                relative = latest_image(item, chain)
                if not relative:
                    continue
                self._load_thumbnail(
                    self.detail.asset_card(item.id, step.id),
                    f"{pack.media_base}/{relative}",
                    state_signature(item, [s.id for s in chain]), ASSET_THUMB,
                    lambda pixmap, item_id=item.id, step_id=step.id:
                    self.detail.set_asset_thumbnail(item_id, step_id, pixmap),
                    layered)

    def _load_ring_thumbnails(self, pack: PackResponse, template: TemplateInfo):
        """环的卡片和"添加"弹窗用 generate 的原图：它不是交付物，卡片墙不会替它加载。"""
        pack_id = pack.manifest.id
        if self._ring_thumb_pack != pack_id:
            self._ring_thumb_pack, self._ring_thumb_tokens = pack_id, {}
        chain = step_chain(template, RING_SOURCE_STEP)
        for item in self.detail.ready_items():
            relative = latest_image(item, chain)
            if not relative:
                continue
            url = f"{pack.media_base}/{relative}"
            signature = state_signature(item, [s.id for s in chain])
            token = f"{url}#{signature}"
            if self._ring_thumb_tokens.get(item.id) == token:
                continue
            # 先记下再加载：失败也不会每次轮询都重试同一张坏图
            self._ring_thumb_tokens[item.id] = token
            self._start(self._fetch_image, url, PICKER_THUMB, signature,
                        on_ok=lambda image, item_id=item.id: self._apply_ring_thumbnail(
                            pack_id, item_id, QPixmap.fromImage(image)),
                        on_error=lambda message: print(f"⚠️ 环缩略图加载失败：{message}"))

    def _apply_ring_thumbnail(self, pack_id: str, item_id: str, pixmap: QPixmap):
        # 加载期间可能已经切到别的包：条目 id 会重名，不能把图贴到别的包上
        if self.detail.pack is not None and self.detail.pack.manifest.id == pack_id:
            self.detail.set_ring_thumbnail(item_id, pixmap)

    def _load_thumbnail(self, card: Card | None, url: str, signature: str, size: QSize,
                        apply: Callable[[QPixmap], None], layered: bool = False):
        token = f"{url}#{signature}"
        if card is None or (card.layer_token if layered else card.thumb_token) == token:
            return
        # 先记下再加载：失败也不会每次轮询都重试同一张坏图
        if layered:
            card.layer_token = token
        else:
            card.thumb_token = token
        # 分层缩略图在后台悄悄加载：卡片上正显示着原图，不必再扫微光
        if not layered:
            card.set_loading(True)

        def failed(message: str):
            card.set_loading(False)
            print(f"⚠️ 缩略图加载失败：{message}")
        self._start(self._fetch_image, url, size, signature,
                    on_ok=lambda image: apply(QPixmap.fromImage(image)), on_error=failed)

    def _fetch_image(self, url: str, size: QSize, signature: str) -> QImage:
        """取图并缩到 size 内。signature 是内容版本：同名产物重跑后它会变，旧缓存随之作废。"""
        cache = self._thumb_cache_path(url, size, signature)
        cached = QImage(str(cache))
        if not cached.isNull():
            return cached
        # 在 worker 线程里解码和缩放：原图可能有几千像素，放主线程会卡界面
        with _DECODE_SLOTS:
            image = QImage.fromData(self._client.fetch_media(url))
            if image.isNull():
                raise ValueError(f"无法解码图片：{url}")
            scaled = image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                  Qt.TransformationMode.SmoothTransformation)
        self._save_thumb_cache(scaled, cache)
        return scaled

    @staticmethod
    def _thumb_cache_path(url: str, size: QSize, signature: str) -> Path:
        key = f"{url}#{signature}#{size.width()}x{size.height()}"
        return THUMB_CACHE_DIR / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()[:20]}.png"

    @staticmethod
    def _save_thumb_cache(image: QImage, cache: Path):
        """先写临时文件再改名：别的线程读到写了一半的文件会当成坏图重新下载。
        缓存只是加速，写不进去不影响缩略图显示，只提示一下。"""
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_name(f"{cache.stem}.{threading.get_ident()}.tmp")
        if not image.save(str(temp), "PNG"):
            print(f"⚠️ 缩略图缓存写入失败：{cache}")
            return
        temp.replace(cache)

    # ---- 素材详情 ----
    def _on_asset_selected(self, _item_id: str, _step_id: str):
        pack = self.detail.pack
        if pack is not None and self.detail.refresh_detail(self._is_running(pack.manifest.id)):
            self._load_preview()

    def _on_asset_opened(self, item_id: str, step_id: str):
        found = self._selected_outputs(item_id, step_id)
        if found is None:
            return
        pack, outputs = found
        images = [o for o in outputs if o.lower().endswith(IMAGE_SUFFIXES)]
        if not images:
            return  # 音频、剧本单击就在详情里看，双击不另开
        key = self.detail.detail.key
        self._start(self._download, pack, images[0], key.signature,
                    on_ok=lambda path: self.detail.show_image(str(path)),
                    on_error=self.detail.set_error)

    def _selected_outputs(self, item_id: str, step_id: str) -> tuple[PackResponse, list[str]] | None:
        pack = self._current_pack()
        if pack is None:
            return None
        item = next((i for i in pack.manifest.items if i.id == item_id), None)
        state = item.steps.get(step_id) if item else None
        if state is None or state.status != "done":
            return None
        return pack, state.outputs

    def _load_preview(self):
        key = self.detail.detail.key
        found = self._selected_outputs(key.item_id, key.step_id) if key else None
        if found is None:
            return
        pack, outputs = found
        panel = self.detail.detail

        def on_error(message: str):
            panel.set_preview_error(key, message)

        images = [o for o in outputs if o.lower().endswith(IMAGE_SUFFIXES)]
        audio = [o for o in outputs if o.lower().endswith(AUDIO_SUFFIXES)]
        if len(images) > 1:
            urls = [f"{pack.media_base}/{image}" for image in images]
            size = QSize(PREVIEW_WIDTH, PREVIEW_WIDTH * 2)
            self._start(lambda: [self._fetch_image(url, size, key.signature) for url in urls],
                        on_ok=lambda loaded: panel.set_images(
                            key, [QPixmap.fromImage(image) for image in loaded]),
                        on_error=on_error)
        elif images:
            self._start(self._fetch_image, f"{pack.media_base}/{images[-1]}",
                        QSize(PREVIEW_WIDTH, PREVIEW_WIDTH * 2), key.signature,
                        on_ok=lambda image: panel.set_image(key, QPixmap.fromImage(image)),
                        on_error=on_error)
        elif audio:
            self._start(self._download_tracks, pack, outputs, key,
                        on_ok=lambda tracks: panel.set_audio(key, tracks), on_error=on_error)
        elif outputs:
            self._start(self._fetch_text, f"{pack.media_base}/{outputs[0]}",
                        on_ok=lambda text: panel.set_text(key, text), on_error=on_error)

    def _fetch_text(self, url: str) -> str:
        return self._client.fetch_media(url).decode("utf-8")

    def _download(self, pack: PackResponse, relative: str, signature: str) -> Path:
        """下载到按内容签名分目录的本地副本：同名产物重跑后签名变了，不会读到旧文件。"""
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        folder, _, name = relative.rpartition("/")
        target = LIBRARY_CACHE_DIR / pack.manifest.id / folder / digest / name
        if not target.exists():
            data = self._client.fetch_media(f"{pack.media_base}/{relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return target

    def _download_tracks(self, pack: PackResponse, outputs: list[str],
                         key: AssetKey) -> list[tuple[str, Path]]:
        """对话配音按 dialogue.json 的顺序列出每句台词和音频；其他音频直接列文件。"""
        audio = [o for o in outputs if o.lower().endswith(AUDIO_SUFFIXES)]
        dialogue = next((o for o in outputs if o.endswith("dialogue.json")), "")
        if not dialogue:
            return [("", self._download(pack, o, key.signature)) for o in audio]
        lines = json.loads(self._fetch_text(f"{pack.media_base}/{dialogue}")).get("lines", [])
        by_name = {o.rpartition("/")[2]: o for o in audio}
        tracks = []
        for line in lines:
            relative = by_name.get(line.get("audio", ""))
            if relative is None:
                raise ValueError(f"找不到台词音频：{line.get('audio', '')}")
            emotion = f"（{line['emotion']}）" if line.get("emotion") else ""
            caption = f"{line.get('speaker', '')}{emotion}：{line.get('text', '')}"
            tracks.append((caption, self._download(pack, relative, key.signature)))
        return tracks

    # ---- worker ----
    def _start(self, fn: Callable[..., Any], *args, on_ok: Callable[[Any], None],
               on_error: Callable[[str], None] | None = None, **kwargs) -> LibraryTaskWorker:
        worker = LibraryTaskWorker(fn, *args, **kwargs)
        worker.finished_ok.connect(on_ok)
        worker.error.connect(on_error or self._on_worker_error)
        worker.finished.connect(lambda: self._workers.discard(worker))
        self._workers.add(worker)
        worker.start()
        return worker

    def _on_worker_error(self, message: str):
        current = self.stack.currentWidget()
        target = current if current in (self.form, self.add_form, self.detail) else self.garden
        target.set_error(message)
        print(f"⚠️ 资料库请求失败：{message}")
