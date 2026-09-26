import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QStackedWidget

from src.app.client import ApiClient
from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.cards import ASSET_THUMB, PACK_THUMB, Card
from src.app.ui.library.detail_panel import PREVIEW_WIDTH, AssetKey
from src.app.ui.library.garden import GardenView
from src.app.ui.library.pack_detail import PackDetail
from src.app.ui.library.pack_form import PackForm
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
from src.app.work import LibraryTaskWorker
from src.shared.schemas import (
    ApproveStepRequest,
    CollectionItem,
    CreatePackRequest,
    PackListResponse,
    PackResponse,
    ResetStepRequest,
    TemplateInfo,
    TemplateListResponse,
)
from src.shared.settings import PROJECT_ROOT

# 执行中轮询的间隔：进度只存在 manifest 里（ADR 0004），任务状态只用来判断结束
POLL_INTERVAL_MS = 2000
# 详情预览用的本地副本：音频播放器和看图浮层都只认本地文件
LIBRARY_CACHE_DIR = Path(PROJECT_ROOT) / "cache" / "library"
CHARACTER_TYPE = "character"


def state_signature(item: CollectionItem, step_ids: list[str]) -> str:
    """产物文件名重跑后不变，用状态和耗时区分新旧内容，缩略图和本地副本据此失效。"""
    parts = []
    for step_id in step_ids:
        state = item.steps.get(step_id)
        if state is not None:
            parts.append(f"{step_id}:{state.status}:{state.meta.get('elapsed', '')}")
    return "|".join(parts)


class LibraryPage(BaseWidget):
    """资料库：花园（分类卡片墙）→ 打开资源包（树 + 素材卡片 + 详情），另有新建表单。
    全部请求都在 LibraryTaskWorker 里执行；执行中每两秒拉一次资源包列表，所有视图都从它刷新。"""

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

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._refresh_packs)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.garden = GardenView()
        self.form = PackForm()
        self.detail = PackDetail()
        for page in (self.garden, self.form, self.detail):
            self.stack.addWidget(page)
        layout.addWidget(self.stack)

        self.garden.pack_opened.connect(self._open_pack)
        self.garden.new_requested.connect(self._show_form)
        self.form.back_requested.connect(self._show_garden)
        self.form.create_requested.connect(self._create_pack)
        self.detail.back_requested.connect(self._show_garden)
        self.detail.pack_selected.connect(self._open_pack)
        self.detail.run_requested.connect(self._run_current)
        self.detail.stop_requested.connect(self._stop_run)
        self.detail.delete_requested.connect(self._delete_current)
        self.detail.asset_selected.connect(self._on_asset_selected)
        self.detail.asset_opened.connect(self._on_asset_opened)
        self.detail.detail.reset_requested.connect(self._reset_step)
        self.detail.detail.approve_requested.connect(self._approve_step)

    # ---- 对外 ----
    def activate(self):
        """每次进入页面时刷新：资源包可能在别处被改动（例如手动删了文件夹）。"""
        self._start(self._client.list_templates, on_ok=self._on_templates)

    def shutdown(self):
        """关窗口时调用：停止执行并等 worker 退出，否则运行中的 QThread 被回收会直接崩溃。"""
        self._poll_timer.stop()
        self._run_stop.set()
        for worker in list(self._workers):
            worker.wait(3000)

    # ---- 页面切换 ----
    def _show_garden(self):
        self.stack.setCurrentWidget(self.garden)

    def _show_form(self, pack_type: str = ""):
        if pack_type:
            self.form.preselect_category(pack_type)
        self.form.set_error("")
        self.stack.setCurrentWidget(self.form)

    def _open_pack(self, pack_id: str):
        pack = self._find_pack(pack_id)
        if pack is None:
            return
        self.stack.setCurrentWidget(self.detail)
        self._show_detail(pack)

    def _current_pack(self) -> PackResponse | None:
        if self.stack.currentWidget() is not self.detail or self.detail.pack is None:
            return None
        return self._find_pack(self.detail.pack.manifest.id)

    def _find_pack(self, pack_id: str) -> PackResponse | None:
        return next((p for p in self._packs if p.manifest.id == pack_id), None)

    # ---- 数据刷新 ----
    def _on_templates(self, response: TemplateListResponse):
        if not response.ok:
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
            if pack.manifest.type != CHARACTER_TYPE:
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

    # ---- 执行状态 ----
    def _running_pack_ids(self) -> set[str]:
        ids = {p.manifest.id for p in self._packs if p.running}
        if self._run_worker is not None:
            ids.add(self._run_pack_id)
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

    def _on_deleted(self, response):
        if not response.ok:
            self.detail.set_error(self.tr("Failed to delete pack: {0}").format(response.error))
            return
        self._show_garden()
        self._refresh_packs()

    # ---- 审阅 / 重做 ----
    def _approve_step(self, item_id: str, step_id: str, content: str | None):
        pack = self.detail.pack
        if pack is None:
            return
        request = ApproveStepRequest(item_id=item_id, step_id=step_id, content=content)
        self._start(self._client.approve_step, pack.manifest.id, request,
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
            for item in pack.manifest.items:
                relative = latest_image(item, chain)
                if not relative:
                    continue
                self._load_thumbnail(
                    self.detail.asset_card(item.id, step.id),
                    f"{pack.media_base}/{relative}",
                    state_signature(item, [s.id for s in chain]), ASSET_THUMB,
                    lambda pixmap, item_id=item.id, step_id=step.id:
                    self.detail.set_asset_thumbnail(item_id, step_id, pixmap))

    def _load_thumbnail(self, card: Card | None, url: str, signature: str, size: QSize,
                        apply: Callable[[QPixmap], None]):
        token = f"{url}#{signature}"
        if card is None or card.thumb_token == token:
            return
        # 先记下再加载：失败也不会每次轮询都重试同一张坏图
        card.thumb_token = token
        self._start(self._fetch_image, url, size,
                    on_ok=lambda image: apply(QPixmap.fromImage(image)),
                    on_error=lambda message: print(f"⚠️ 缩略图加载失败：{message}"))

    def _fetch_image(self, url: str, size: QSize) -> QImage:
        # 在 worker 线程里解码和缩放：原图可能有几千像素，放主线程会卡界面
        image = QImage.fromData(self._client.fetch_media(url))
        if image.isNull():
            raise ValueError(f"无法解码图片：{url}")
        return image.scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)

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
        self._start(self._download, pack, images[-1], key.signature,
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
        if images:
            self._start(self._fetch_image, f"{pack.media_base}/{images[-1]}",
                        QSize(PREVIEW_WIDTH, PREVIEW_WIDTH * 2),
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
        target = current if current in (self.form, self.detail) else self.garden
        target.set_error(message)
        print(f"⚠️ 资料库请求失败：{message}")
