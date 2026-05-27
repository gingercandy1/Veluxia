from abc import ABC, abstractmethod
from typing import Any, Dict
from fastapi import APIRouter

from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import ModelInfoResponse
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
