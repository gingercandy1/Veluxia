import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.backend.core.collection import library, ring
from src.backend.core.collection.drafts import draft_items
from src.backend.core.collection.template import Template, list_templates
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.job_manager import Job
from src.backend.core.model_utils import to_media_url
from src.backend.router_base import BaseRouter
from src.shared.schemas import (
    AddItemsRequest,
    AdoptSeamRequest,
    ApproveStepRequest,
    BaseResponse,
    ConfirmCutsRequest,
    CreatePackRequest,
    DraftItemsRequest,
    DraftItemsResponse,
    FieldOption,
    Manifest,
    PackListResponse,
    PackResponse,
    ResetStepRequest,
    RingCutsRequest,
    RingGenerateRequest,
    RingLayersRequest,
    RingPlanRequest,
    RingStatusResponse,
    RunPackRequest,
    StyleListResponse,
    StylePreset,
    TemplateFieldInfo,
    TemplateInfo,
    TemplateListResponse,
    TemplatePresetInfo,
    TemplateStepInfo,
    UpdateItemFieldsRequest,
)


class LibraryRouter(BaseRouter):
    """资料库（ADR 0004）：资源包的增删查走普通接口，执行整个资源包走任务提交 + 轮询。"""
    prefix = "/library"
    tags   = ["Library"]

    def _register_routes(self) -> None:
        # 执行进度以 manifest 为准（GET /packs/{id}），任务状态只用来判断结束和取消
        self._register_async_generate_routes(self.handle_run, RunPackRequest, path="/run")
        self._register_async_generate_routes(
            self.handle_ring_generate, RingGenerateRequest, path="/ring/generate")
        # 首次要下载深度模型，耗时说不准，走任务提交 + 轮询
        self._register_async_generate_routes(
            self.handle_ring_cuts, RingCutsRequest, path="/ring/cuts/auto")
        # 每段约 18 分钟，十几段要跑几个小时
        self._register_async_generate_routes(
            self.handle_ring_layers, RingLayersRequest, path="/ring/layers")

        @self.router.get("/templates", response_model=TemplateListResponse, summary="可用模板")
        async def templates() -> TemplateListResponse:
            try:
                items = await asyncio.to_thread(list_templates)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return TemplateListResponse(templates=[self._template_info(t) for t in items])

        @self.router.get("/packs", response_model=PackListResponse, summary="列出资源包")
        async def list_packs() -> PackListResponse:
            manifests = await asyncio.to_thread(library.list_packs)
            return PackListResponse(packs=[self._pack_response(m) for m in manifests])

        @self.router.post("/packs", response_model=PackResponse, summary="新建资源包")
        async def create_pack(req: CreatePackRequest) -> PackResponse:
            try:
                manifest = await asyncio.to_thread(library.create_pack, req)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return self._pack_response(manifest)

        @self.router.post("/drafts", response_model=DraftItemsResponse, summary="AI 起草条目")
        async def draft(req: DraftItemsRequest) -> DraftItemsResponse:
            try:
                # 加载并运行 LLM 是同步阻塞的，放进线程池避免卡住事件循环
                items = await asyncio.to_thread(draft_items, req)
            except ValueError as exc:
                # 起草失败多半是模型输出解析不了，访问日志里只有 400，原因要单独打出来才好查
                print(f"⚠️ AI 起草失败（{req.template}：{req.theme.strip()}）：{exc}")
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except GeneratorBusyError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return DraftItemsResponse(items=items)

        @self.router.get("/packs/{pack_id}", response_model=PackResponse, summary="读取资源包")
        async def get_pack(pack_id: str) -> PackResponse:
            try:
                manifest = await asyncio.to_thread(library.get_pack, pack_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return self._pack_response(manifest)

        @self.router.get("/packs/{pack_id}/ring", response_model=RingStatusResponse,
                         summary="读取循环拼接状态")
        async def get_ring(pack_id: str) -> RingStatusResponse:
            try:
                status = await asyncio.to_thread(ring.ring_status, pack_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            directory = library.pack_dir(pack_id)
            for seam in status["seams"].values():
                seam.candidates = [
                    candidate.model_copy(update={
                        "file": to_media_url(directory / candidate.file)
                    })
                    for candidate in seam.candidates
                ]
            return RingStatusResponse(**status)

        @self.router.post("/packs/{pack_id}/ring/plan", response_model=PackResponse,
                          summary="规划循环拼接顺序")
        async def plan_ring(pack_id: str, req: RingPlanRequest | None = None) -> PackResponse:
            return await self._update_pack(
                ring.plan_ring, pack_id, req.item_ids if req is not None else None)

        @self.router.post("/packs/{pack_id}/ring/adopt", response_model=PackResponse,
                          summary="采用一个接缝候选")
        async def adopt_ring_seam(pack_id: str, req: AdoptSeamRequest) -> PackResponse:
            return await self._update_pack(
                ring.adopt_seam, pack_id, req.key, req.candidate_id)

        @self.router.post("/packs/{pack_id}/ring/render", response_model=PackResponse,
                          summary="把采用的接缝贴回各条目")
        async def render_ring(pack_id: str) -> PackResponse:
            return await self._update_pack(ring.render_ring, pack_id)

        @self.router.post("/packs/{pack_id}/ring/cuts", response_model=PackResponse,
                          summary="确认切段位置并导出每段图")
        async def confirm_ring_cuts(pack_id: str, req: ConfirmCutsRequest) -> PackResponse:
            return await self._update_pack(ring.confirm_cuts, pack_id, req.cuts)

        @self.router.delete("/packs/{pack_id}", response_model=BaseResponse, summary="删除资源包")
        async def delete_pack(pack_id: str) -> BaseResponse:
            try:
                await asyncio.to_thread(library.delete_pack, pack_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except library.PackBusyError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return BaseResponse(ok=True)

        @self.router.get("/packs/{pack_id}/export", summary="把成品打包成 zip 下载",
                         response_class=FileResponse)
        async def export_pack(pack_id: str) -> FileResponse:
            # 走 HTTP 下载而不是让前端直接读包目录：远程后端时前端看不到后端的文件系统
            fd, name = tempfile.mkstemp(suffix=".zip", prefix="veluxia_export_")
            os.close(fd)
            target = Path(name)
            try:
                await asyncio.to_thread(library.export_pack, pack_id, target)
            except FileNotFoundError as exc:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return FileResponse(target, media_type="application/zip", filename=f"{pack_id}.zip",
                                background=BackgroundTask(target.unlink, missing_ok=True))

        @self.router.get("/packs/{pack_id}/items/{item_id}/steps/{step_id}/export",
                         summary="导出一张素材卡片的成品：单文件原样返回，多文件打成 zip",
                         response_class=FileResponse)
        async def export_asset(pack_id: str, item_id: str, step_id: str) -> FileResponse:
            try:
                files = await asyncio.to_thread(library.asset_files, pack_id, item_id, step_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if len(files) == 1:
                return FileResponse(files[0], filename=files[0].name)
            fd, name = tempfile.mkstemp(suffix=".zip", prefix="veluxia_export_")
            os.close(fd)
            target = Path(name)
            try:
                await asyncio.to_thread(library.export_asset, pack_id, item_id, step_id, target)
            except (FileNotFoundError, ValueError) as exc:
                # 刚才取文件列表之后文件又被删了：和上面一样当作"没有可导出的内容"
                target.unlink(missing_ok=True)
                status = 404 if isinstance(exc, FileNotFoundError) else 400
                raise HTTPException(status_code=status, detail=str(exc)) from exc
            return FileResponse(target, media_type="application/zip",
                                filename=f"{pack_id}_{item_id}_{step_id}.zip",
                                background=BackgroundTask(target.unlink, missing_ok=True))

        @self.router.post("/packs/{pack_id}/items", response_model=PackResponse,
                          summary="往已有资源包追加条目")
        async def add_items(pack_id: str, req: AddItemsRequest) -> PackResponse:
            return await self._update_pack(library.add_items, pack_id, req)

        @self.router.delete("/packs/{pack_id}/items/{item_id}", response_model=PackResponse,
                            summary="删除资源包里的一个条目")
        async def delete_item(pack_id: str, item_id: str) -> PackResponse:
            return await self._update_pack(library.delete_item, pack_id, item_id)

        @self.router.post("/packs/{pack_id}/approve", response_model=PackResponse,
                          summary="确认待审阅的步骤")
        async def approve_step(pack_id: str, req: ApproveStepRequest) -> PackResponse:
            return await self._update_pack(library.approve_step, pack_id, req)

        @self.router.post("/packs/{pack_id}/reset", response_model=PackResponse,
                          summary="重做某条目的某一步")
        async def reset_step(pack_id: str, req: ResetStepRequest) -> PackResponse:
            return await self._update_pack(library.reset_step, pack_id, req)

        @self.router.post("/packs/{pack_id}/fields", response_model=PackResponse,
                          summary="事后修改条目字段（如补做分层），受影响的步骤标记为待重做")
        async def update_item_fields(pack_id: str, req: UpdateItemFieldsRequest) -> PackResponse:
            return await self._update_pack(library.update_item_fields, pack_id, req)

        @self.router.post("/packs/{pack_id}/sync_style", response_model=PackResponse,
                          summary="风格锁同步为预设的最新内容")
        async def sync_style(pack_id: str) -> PackResponse:
            return await self._update_pack(library.sync_style, pack_id)

        @self.router.post("/packs/{pack_id}/refresh_source", response_model=PackResponse,
                          summary="来源立绘重做后，把用旧立绘做的步骤标记为待重做")
        async def refresh_source(pack_id: str) -> PackResponse:
            return await self._update_pack(library.refresh_source, pack_id)

        @self.router.get("/styles", response_model=StyleListResponse, summary="风格预设")
        async def list_styles() -> StyleListResponse:
            try:
                styles = await asyncio.to_thread(library.list_styles)
            except ValueError as exc:
                raise HTTPException(status_code=500, detail=f"styles.json 无法解析：{exc}") from exc
            return StyleListResponse(styles=styles)

        @self.router.post("/styles", response_model=StyleListResponse, summary="新建或修改风格预设")
        async def save_style(req: StylePreset) -> StyleListResponse:
            try:
                styles = await asyncio.to_thread(library.save_style, req)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StyleListResponse(styles=styles)

        @self.router.delete("/styles/{style_id}", response_model=StyleListResponse,
                            summary="删除风格预设")
        async def delete_style(style_id: str) -> StyleListResponse:
            try:
                styles = await asyncio.to_thread(library.delete_style, style_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StyleListResponse(styles=styles)

    async def _update_pack(self, action, pack_id: str, *args) -> PackResponse:
        try:
            manifest = await asyncio.to_thread(action, pack_id, *args)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except library.PackBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return self._pack_response(manifest)

    async def handle_run(self, req: RunPackRequest, job: Job) -> PackResponse:
        # runner 内部用 asyncio.run 驱动生成器，必须放到没有事件循环的线程里执行
        manifest = await asyncio.to_thread(library.run_pack, req.pack_id, job.cancel_event)
        return self._pack_response(manifest)

    async def handle_ring_generate(self, req: RingGenerateRequest, job: Job) -> PackResponse:
        manifest = await asyncio.to_thread(
            ring.generate_seams, req.pack_id, req.keys, req.prompt, job.cancel_event,
            extra_prompt=req.extra_prompt)
        return self._pack_response(manifest)

    async def handle_ring_cuts(self, req: RingCutsRequest, job: Job) -> PackResponse:
        manifest = await asyncio.to_thread(ring.auto_cuts, req.pack_id, job.cancel_event)
        return self._pack_response(manifest)

    async def handle_ring_layers(self, req: RingLayersRequest, job: Job) -> PackResponse:
        # 生成器内部用 asyncio.run 驱动，必须放到没有事件循环的线程里
        manifest = await asyncio.to_thread(
            ring.layer_segments, req.pack_id, req.indices, job.cancel_event)
        return self._pack_response(manifest)

    @staticmethod
    def _template_info(template: Template) -> TemplateInfo:
        segments_of = {prompts: segments for segments, prompts in template.segment_fields()}
        return TemplateInfo(
            id=template.id,
            type=template.type,
            name=template.name,
            description=template.description,
            summary=template.summary,
            prompt_label=template.prompt_label,
            cover=template.cover,
            source=template.source,
            fields=[
                TemplateFieldInfo(
                    id=spec.id, label=spec.label, required=spec.required, default=spec.default,
                    options=[FieldOption(value=value, label=label) for value, label in spec.options],
                    segments_field=segments_of.get(spec.id, ""), kind=spec.kind,
                )
                for spec in template.fields
            ],
            presets=[TemplatePresetInfo(name=p.name, prompt=p.prompt, fields=dict(p.fields))
                     for p in template.presets],
            steps=template.step_ids(),
            step_details=[
                TemplateStepInfo(id=step.id, type=step.type, label=step.label,
                                 deliverable=step.deliverable, review=step.review,
                                 inputs=list(step.inputs), when=step.when)
                for step in template.steps
            ],
        )

    @staticmethod
    def _pack_response(manifest: Manifest) -> PackResponse:
        return PackResponse(
            manifest=manifest,
            media_base=to_media_url(library.pack_dir(manifest.id)),
            running=library.is_running(manifest.id),
            source_changed=library.source_changed(manifest),
        )
