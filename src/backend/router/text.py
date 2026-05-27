import json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.shared.schemas import BaseRequest, TextResponse
from src.shared.enum_type import FactoryType
from src.backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.text.index_memory import SessionStore, EmbedModel

class TextRouter(BaseRouter):
    prefix       = "/text"
    tags         = ["Text"]
    factory_type = FactoryType.Text

    def _register_routes(self) -> None:
        @self.router.post("/generate", response_model=TextResponse, summary="文本生成")
        async def generate(req: BaseRequest) -> TextResponse:
            return await self.handle_generate(req)

        @self.router.post("/stream", response_model=TextResponse, summary="文本生成")
        async def generate_stream(req: BaseRequest) -> StreamingResponse:
            return await self.handle_generate_stream(req)

        @self.router.delete("/memory/session/{session_id}")
        async def clear_session(session_id: str, user_id: str = "default") -> dict:
            return await self.clear_session_memory(session_id)

    async def handle_generate(self, req: BaseRequest) -> TextResponse:
        generator = GeneratorFactory.build_generator(FactoryType.Text, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)
        generator.swtich_memory(req.session_id, )

        try:
            content = await generator.generate()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return TextResponse(ok=True, session_id=req.session_id, content=content or "")

    async def handle_generate_stream(self, req: BaseRequest):
        generator = GeneratorFactory.build_generator(FactoryType.Text, req.model_name)
        generator.ensure_model_loaded()
        generator.parse_params(req.extra)
        generator.switch_memory(req.user_id, req.session_id)
    
        async def event_stream():
            async for chunk in generator.generate_stream():
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                })

    async def clear_session_memory(self, session_id: str, user_id: str = "default"):
        embed_model = EmbedModel()
        memory = SessionStore.get_instance(embed_model=embed_model)
        memory.delete_session(user_id, session_id)
        return {"ok": True, "session_id": session_id}
