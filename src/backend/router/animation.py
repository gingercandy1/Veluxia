from fastapi import HTTPException

from backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import AnimationResponse, BaseRequest
from src.shared.enum_type import FactoryType


class AnimationRouter(BaseRouter):
    prefix       = "/animation"
    tags         = ["Animation"]
    factory_type = FactoryType.Animation

    def _register_routes(self) -> None:
        @self.router.post("/generate", response_model=AnimationResponse, summary="图像转动画")
        async def generate(req: BaseRequest) -> AnimationResponse:
            return await self.handle_generate(req)

    async def handle_generate(self, req: BaseRequest) -> AnimationResponse:
        generator = GeneratorFactory.build_generator(FactoryType.Animation, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)

        try:
            frame_paths, video_path = await generator.generate_animation()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return AnimationResponse(
            ok=True,
            session_id=req.session_id,
            video_path=str(video_path) if video_path else None,
            frame_paths=[str(p) for p in frame_paths],
        )

