import asyncio

from fastapi import HTTPException

from src.backend.core import storage
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.router_base import BaseRouter
from src.shared.schemas import (
    DeleteModelStorageRequest,
    ModelStorageEntry,
    ModelStorageResponse,
)


class StorageRouter(BaseRouter):
    """模型权重的磁盘占用：走后端接口而不是前端直接读目录，远程后端时看的是 GPU 机器上的磁盘。"""
    prefix = "/storage"
    tags   = ["System"]

    def _register_routes(self) -> None:
        @self.router.get("/models", response_model=ModelStorageResponse, summary="模型文件占用")
        async def list_models() -> ModelStorageResponse:
            # 遍历几百 GB 的目录要一会儿，放线程池里别卡住事件循环
            return await asyncio.to_thread(self._response)

        @self.router.post("/models/delete", response_model=ModelStorageResponse,
                          summary="删除模型文件，下次使用时重新下载")
        async def delete_model(req: DeleteModelStorageRequest) -> ModelStorageResponse:
            try:
                await asyncio.to_thread(storage.delete_entry, req.id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except GeneratorBusyError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return await asyncio.to_thread(self._response)

    @staticmethod
    def _response() -> ModelStorageResponse:
        sized = storage.list_storage()
        entries = [ModelStorageEntry(id=e.id, root=e.root, category=e.category, name=e.name,
                                     size=size, manual=e.manual) for e, size in sized]
        return ModelStorageResponse(entries=entries, total=sum(size for _, size in sized))
