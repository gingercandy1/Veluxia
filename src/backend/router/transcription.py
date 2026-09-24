from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.exceptions import GenerationCancelled, GeneratorBusyError
from src.backend.core.job_manager import Job
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import to_media_url
from src.shared.schemas import BaseRequest, TranscriptionResponse
from src.shared.enum_type import FactoryType


class TranscriptionRouter(BaseRouter):
    prefix       = "/transcription"
    tags         = ["Transcription"]
    factory_type = FactoryType.Transcription

    def _register_routes(self) -> None:
        self._register_async_generate_routes(self.handle_generate, BaseRequest)

    async def handle_generate(self, req: BaseRequest, job: Job) -> TranscriptionResponse:
        # 识别器纯 CPU，acquire 只按类串行，不会卸载显存里的图片/动画模型（ADR 0003）。
        # Auto 要按语言挑模型，所以先 parse_params 再加载。
        try:
            with GeneratorFactory.acquire(FactoryType.Transcription, req.model_name) as generator:
                generator.cancel_event = job.cancel_event
                generator.parse_params(req.extra)
                generator.ensure_model_loaded()
                result = await generator.transcribe()
        except (GenerationCancelled, GeneratorBusyError):
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        srt_path = result.pop("srt_path")
        return TranscriptionResponse(
            ok=True,
            session_id=req.session_id,
            srt_path=to_media_url(srt_path) if srt_path else None,
            **result,
        )
