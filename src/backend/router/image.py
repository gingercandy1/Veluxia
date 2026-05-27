from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import BaseRequest, ImageResponse
from src.shared.enum_type import FactoryType


class ImageRouter(BaseRouter):
    prefix       = "/image"
    tags         = ["Image"]
    factory_type = FactoryType.Image

    def _register_routes(self) -> None:

        @self.router.post(
            "/generate",
            response_model=ImageResponse,
            summary="文生图 / 图生图",
        )
        async def generate(req: BaseRequest) -> ImageResponse:
            return await self._handle_generate(req)

    async def _handle_generate(self, req: BaseRequest) -> ImageResponse:
        extra = req.extra
        number = extra.get("number", 1)
        reference_image = extra.get("reference_image", None)

        generator = GeneratorFactory.build_generator(FactoryType.Image, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(extra)

        # ④ 循环生成
        paths: list[str] = []
        for _ in range(number):
            try:
                if reference_image:
                    path: Optional[Path] = await generator.generate_by_image()
                else:
                    path: Optional[Path] = await generator.generate()
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            if path:
                paths.append(str(path))

        return ImageResponse(
            ok=True,
            session_id=req.session_id,
            paths=paths,
        )