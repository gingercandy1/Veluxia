import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from src.backend.core.collection import library
from src.backend.core.collection.drafts import draft_items
from src.backend.core.collection.template import Template, list_templates
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.job_manager import Job
from src.backend.core.model_utils import to_media_url
from src.backend.router_base import BaseRouter
from src.shared.schemas import (
    ApproveStepRequest,
    BaseResponse,
    CreatePackRequest,
    DraftItemsRequest,
    DraftItemsResponse,
    FieldOption,
    Manifest,
    PackListResponse,
    PackResponse,
    ResetStepRequest,
    RunPackRequest,
    StyleListResponse,
    StylePreset,
    TemplateFieldInfo,
    TemplateInfo,
    TemplateListResponse,
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

    @staticmethod
    def _template_info(template: Template) -> TemplateInfo:
        return TemplateInfo(
            id=template.id,
            type=template.type,
            name=template.name,
            description=template.description,
            prompt_label=template.prompt_label,
            cover=template.cover,
            source=template.source,
            fields=[
                TemplateFieldInfo(
                    id=spec.id, label=spec.label, required=spec.required, default=spec.default,
                    options=[FieldOption(value=value, label=label) for value, label in spec.options],
                )
                for spec in template.fields
            ],
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
