from typing import List
from pathlib import Path

from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import to_media_url
from src.shared.schemas import AnimationResponse, BaseRequest
from src.shared.enum_type import FactoryType


class ImageFrameRouter(BaseRouter):
    prefix       = "/image_frame"
    tags         = ["ImageFrame"]
    factory_type = FactoryType.ImageFrame

    def _register_routes(self) -> None:

        @self.router.post(
            "/interpolate",
            response_model=AnimationResponse,
            summary="帧插值（2 张 → 中间帧；2 张以上 → 序列）",
        )
        async def interpolate(req: BaseRequest) -> AnimationResponse:
            generator = GeneratorFactory.build_generator(FactoryType.ImageFrame, req.model_name)
            generator.ensure_model_loaded()
            generator.parse_params(req.extra)

            try:
                frame_paths: List[Path] = await generator.generate()
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            return AnimationResponse(
                ok=bool(frame_paths),
                session_id=req.session_id,
                frame_paths=[to_media_url(p) for p in frame_paths],
            )
