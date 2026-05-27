from core.translate_pipeline import translate
from shared.schemas import BaseRequest, TranslateResponse
from src.backend.router_base import BaseRouter
from src.shared.enum_type import FactoryType


class TranslateRouter(BaseRouter):
    prefix       = "/translate"
    factory_type = FactoryType.Translation

    def _register_routes(self) -> None:
        @self.router.post("/generate", response_model=TranslateResponse, summary="文本生成")
        async def generate(req: BaseRequest) -> TranslateResponse:
            return await self.handle_generate(req)

        @self.router.post("/default", response_model=TranslateResponse, summary="文本生成")
        async def generate_default(req: BaseRequest) -> TranslateResponse:
            return await self.handle_default_generate(req)

    async def handle_generate(self, req: BaseRequest) -> TranslateResponse:
        text = req.extra.get("content", "")
        source = req.setting.get("translation", {}).get("source")
        target = req.setting.get("translation", {}).get("target")

        result = translate(text, target, source)
        return TranslateResponse(ok=True, session_id=req.session_id, translate_result=result or "")

    async def handle_default_generate(self, req: BaseRequest) -> TranslateResponse:
        text = req.extra.get("content", "")
        result = translate(text)
        return TranslateResponse(ok=True, session_id=req.session_id, translate_result=result or "")
