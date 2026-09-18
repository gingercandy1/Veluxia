from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict
from fastapi import APIRouter, HTTPException

from src.backend.core.job_manager import JobManager
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import BaseResponse, JobStatusResponse, JobSubmitResponse, ModelInfoResponse
from src.shared.enum_type import FactoryType


class BaseRouter(ABC):
    """
    所有模态路由的抽象基类。

    子类示例
    --------
    class ImageRouter(BaseRouter):
        prefix       = "/image"
        tags         = ["Image"]
        factory_type = FactoryType.Image

        def _register_routes(self):
            @self.router.post("/generate", response_model=ImageResponse)
            async def generate(req: ImageRequest):
                return await self.handle_generate(req)

        async def handle_generate(self, req: ImageRequest) -> ImageResponse:
            ...
    """

    prefix: str = ""
    tags: list[str] = []
    factory_type: FactoryType = None

    def __init__(self) -> None:
        self.router = APIRouter(prefix=self.prefix, tags=self.tags)
        self._register_routes()             # 注册子类自定义路由
        self._register_common_routes()      # 注册公共路由

    @abstractmethod
    def _register_routes(self) -> None:
        pass

    def _register_common_routes(self) -> None:
        @self.router.get("/models", response_model=ModelInfoResponse, summary="获取已注册模型列表")
        async def get_models() -> ModelInfoResponse:
            return self._get_model_info()

    def _get_model_info(self) -> ModelInfoResponse:
        names = GeneratorFactory.get_generator_names(self.factory_type)
        tags  = GeneratorFactory.get_model_info(self.factory_type)
        return ModelInfoResponse(
            type=str(self.factory_type.name),
            names=names,
            tags=tags,
        )

    @staticmethod
    def error_response(cls, msg: str, **extra) -> Dict[str, Any]:
        return {"ok": False, "error": msg, **extra}

    def _register_async_generate_routes(
        self,
        handler: Callable[[Any], Awaitable[BaseResponse]],
        request_cls: type,
        path: str = "/generate",
    ) -> None:
        """
        注册"提交任务 + 轮询状态"这一对通用路由，供耗时不固定的生成类接口
        （图片 / 动画 / 语音）复用，替代原来的一次性阻塞 HTTP 请求模式。

        handler: 原来 `await handler(req)` 就能拿到最终 Response 的那个协程函数，
        不需要改动内部生成逻辑，只是不再让 HTTP 请求一直挂着等它返回。
        """

        @self.router.post(f"{path}/submit", response_model=JobSubmitResponse, summary="提交生成任务，立即返回任务 ID")
        async def submit(req: request_cls) -> JobSubmitResponse:
            job = JobManager.submit(lambda job: handler(req, job))
            return JobSubmitResponse(job_id=job.id, status=job.status.value)

        @self.router.get(f"{path}/status/{{job_id}}", response_model=JobStatusResponse, summary="查询生成任务状态")
        async def status(job_id: str) -> JobStatusResponse:
            job = JobManager.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="任务不存在或已过期")
            result_dict = job.result.model_dump() if job.result is not None else None
            return JobStatusResponse(
                job_id=job.id,
                status=job.status.value,
                result=result_dict,
                error=job.error,
            )

        @self.router.post(f"{path}/cancel/{{job_id}}", response_model=BaseResponse, summary="取消正在进行的生成任务")
        async def cancel(job_id: str) -> BaseResponse:
            ok = JobManager.cancel(job_id)
            if not ok:
                job = JobManager.get(job_id)
                if job is None:
                    raise HTTPException(status_code=404, detail="任务不存在或已过期")
                return BaseResponse(ok=False, error=f"任务已处于 {job.status.value} 状态，无法取消")
            return BaseResponse(ok=True)
