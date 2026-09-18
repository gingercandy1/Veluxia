from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import to_media_url
from src.shared.schemas import SpeechResponse, BaseRequest
from src.shared.enum_type import FactoryType

class SpeechRouter(BaseRouter):
    prefix       = "/speech"
    tags         = ["Speech"]
    factory_type = FactoryType.Speech

    def _register_routes(self) -> None:
        self._register_async_generate_routes(self.handle_generate, BaseRequest)

    async def handle_generate(self, req: BaseRequest) -> SpeechResponse:
        generator = GeneratorFactory.build_generator(FactoryType.Speech, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)

        try:
            audio_path = await generator.generate_music()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not audio_path:
            raise HTTPException(status_code=500, detail="语音/音乐生成失败，请查看后端日志")

        return SpeechResponse(
            ok=True,
            session_id=req.session_id,
            audio_path=to_media_url(audio_path),
        )
