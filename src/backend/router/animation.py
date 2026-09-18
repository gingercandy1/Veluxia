from fastapi import HTTPException

from backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import to_media_url
from src.shared.schemas import AnimationResponse, BaseRequest
from src.shared.enum_type import FactoryType


class AnimationRouter(BaseRouter):
    prefix       = "/animation"
    tags         = ["Animation"]
    factory_type = FactoryType.Animation

    def _register_routes(self) -> None:
        self._register_async_generate_routes(self.handle_generate, BaseRequest)

    async def handle_generate(self, req: BaseRequest) -> AnimationResponse:
        generator = GeneratorFactory.build_generator(FactoryType.Animation, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)

        try:
            frame_paths, video_path = await generator.generate_animation()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not video_path:
            raise HTTPException(status_code=500, detail="动画生成失败，请查看后端日志")

        return AnimationResponse(
            ok=True,
            session_id=req.session_id,
            video_path=to_media_url(video_path),
            frame_paths=[to_media_url(p) for p in frame_paths],
        )

