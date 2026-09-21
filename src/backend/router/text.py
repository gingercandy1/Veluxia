import asyncio
import json
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.shared.schemas import BaseRequest, TextResponse
from src.shared.enum_type import FactoryType
from src.backend.router_base import BaseRouter
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.text.index_memory import SessionStore, EmbedModel

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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
        try:
            with GeneratorFactory.acquire(FactoryType.Text, req.model_name) as generator:
                generator.ensure_model_loaded()
                generator.parse_params(req.extra)
                generator.swtich_memory(req.session_id, )
                content = await generator.generate()
        except GeneratorBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return TextResponse(ok=True, session_id=req.session_id, content=content or "")

    async def _stream_events(self, generator, req: BaseRequest):
        # 模型就绪前先把"下载 / 加载"这两个阶段推给前端：加载是同步阻塞的，
        # 放进线程池跑，这边一边等一边按当前进度发事件，界面才不会干等一片空白。
        if generator.pipe is None:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(None, generator.ensure_model_loaded)
            last = None
            while not future.done():
                stage = generator.load_stage
                if stage != last:
                    last = stage
                    yield sse({"type": "stage", **stage})
                await asyncio.sleep(0.2)
            try:
                await future
            except Exception as exc:
                yield sse({"type": "error", "text": f"模型加载失败: {exc}"})
                yield "data: [DONE]\n\n"
                return
            yield sse({"type": "stage", "stage": "ready", "progress": 1.0, "detail": ""})

        generator.parse_params(req.extra)
        generator.switch_memory(req.user_id, req.session_id)

        async for chunk in generator.generate_stream():
            yield sse(chunk)
        yield "data: [DONE]\n\n"

    async def handle_generate_stream(self, req: BaseRequest):
        async def event_stream():
            # 租约要覆盖整段流：客户端断开时生成器被关闭，finally 会释放租约
            try:
                with GeneratorFactory.acquire(FactoryType.Text, req.model_name) as generator:
                    async for event in self._stream_events(generator, req):
                        yield event
            except GeneratorBusyError as exc:
                yield sse({"type": "error", "text": str(exc)})
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
