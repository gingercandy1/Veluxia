import asyncio

from fastapi import HTTPException

from src.backend.core.collection import library
from src.backend.core.collection.template import list_templates
from src.backend.core.job_manager import Job
from src.backend.core.model_utils import to_media_url
from src.backend.router_base import BaseRouter
from src.shared.schemas import (
    BaseResponse,
    CreatePackRequest,
    Manifest,
    PackListResponse,
    PackResponse,
    RunPackRequest,
    TemplateInfo,
    TemplateListResponse,
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
            return TemplateListResponse(templates=[
                TemplateInfo(id=t.id, type=t.type, steps=t.step_ids()) for t in items
            ])

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

    async def handle_run(self, req: RunPackRequest, job: Job) -> PackResponse:
        # runner 内部用 asyncio.run 驱动生成器，必须放到没有事件循环的线程里执行
        manifest = await asyncio.to_thread(library.run_pack, req.pack_id, job.cancel_event)
        return self._pack_response(manifest)

    @staticmethod
    def _pack_response(manifest: Manifest) -> PackResponse:
        return PackResponse(
            manifest=manifest,
            media_base=to_media_url(library.pack_dir(manifest.id)),
            running=library.is_running(manifest.id),
        )
