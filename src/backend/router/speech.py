from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.exceptions import GenerationCancelled, GeneratorBusyError
from src.backend.core.job_manager import Job
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

    async def handle_generate(self, req: BaseRequest, job: Job) -> SpeechResponse:
        # 注意：语音生成器（ACE-Step / Qwen3-TTS）内部调用的是不支持中途打断
        # 一旦点了停止，job 会被标记为 cancelled、前端立刻停止等待，
        # 但后端这次推理仍会在线程里跑完（结果直接丢弃），不会真正省下算力；
        # 租约也要等它跑完才释放，期间新的生成请求会被拒绝而不是抢占权重。
        try:
            with GeneratorFactory.acquire(FactoryType.Speech, req.model_name) as generator:
                generator.ensure_model_loaded()
                generator.cancel_event = job.cancel_event
                generator.parse_params(req.extra)
                audio_path = await generator.generate_music()
        except (GenerationCancelled, GeneratorBusyError):
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not audio_path:
            raise HTTPException(status_code=500, detail="语音/音乐生成失败，请查看后端日志")

        return SpeechResponse(
            ok=True,
            session_id=req.session_id,
            audio_path=to_media_url(audio_path),
        )
