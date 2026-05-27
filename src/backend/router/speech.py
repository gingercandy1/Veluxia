from fastapi import HTTPException

from backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.shared.schemas import SpeechResponse, BaseRequest
from src.shared.enum_type import FactoryType

class SpeechRouter(BaseRouter):
    prefix       = "/speech"
    tags         = ["Speech"]
    factory_type = FactoryType.Speech

    def _register_routes(self) -> None:
        @self.router.post("/generate", response_model=SpeechResponse, summary="文本转语音/音乐")
        async def generate(req: BaseRequest) -> SpeechResponse:
            return await self.handle_generate(req)

    async def handle_generate(self, req: BaseRequest) -> SpeechResponse:
        generator = GeneratorFactory.build_generator(FactoryType.Speech, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)

        try:
            audio_path = await generator.generate_music()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return SpeechResponse(
            ok=True,
            session_id=req.session_id,
            audio_path=str(audio_path) if audio_path else None,
        )
