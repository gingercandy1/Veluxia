import asyncio

from fastapi import HTTPException

from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.prompt_refiner import PromptRefiner
from src.backend.router_base import BaseRouter
from src.shared.schemas import BaseRequest, RefineResponse


class PromptRouter(BaseRouter):
    prefix = "/prompt"
    tags   = ["Prompt"]

    def _register_routes(self) -> None:
        @self.router.post("/refine", response_model=RefineResponse, summary="优化生成提示词")
        async def refine(req: BaseRequest) -> RefineResponse:
            return await self.handle_refine(req)

    async def handle_refine(self, req: BaseRequest) -> RefineResponse:
        # 这里的 model_name 不是生成器名，而是要优化的目标模态（image / animation）
        mode = req.extra.get("mode", "")
        text = req.extra.get("content", "")
        try:
            # 加载并运行模型是同步阻塞的，放进线程池避免卡住事件循环
            refined = await asyncio.to_thread(PromptRefiner().refine, text, mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GeneratorBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return RefineResponse(ok=True, session_id=req.session_id, refined=refined)
