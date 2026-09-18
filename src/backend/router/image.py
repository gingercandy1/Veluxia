from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.model_utils import to_media_url
from src.shared.schemas import BaseRequest, ImageResponse
from src.shared.enum_type import FactoryType


class ImageRouter(BaseRouter):
    prefix       = "/image"
    tags         = ["Image"]
    factory_type = FactoryType.Image

    def _register_routes(self) -> None:
        self._register_async_generate_routes(self._handle_generate, BaseRequest)

        @self.router.post(
            "/remove-background",
            response_model=ImageResponse,
            summary="去背景（输入图 → 透明底 PNG）",
        )
        async def remove_background(req: BaseRequest) -> ImageResponse:
            generator = self._resolve_generator(req)
            generator.parse_params(req.extra)
            try:
                path: Optional[Path] = await generator.generate()
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return ImageResponse(
                ok=path is not None,
                session_id=req.session_id,
                paths=[to_media_url(path)] if path else [],
            )

    def _resolve_generator(self, req: BaseRequest):
        """两条路由共用：取单例生成器（不加载参数，由调用方按需装载）。"""
        generator = GeneratorFactory.build_generator(FactoryType.Image, req.model_name)
        generator.ensure_model_loaded()
        return generator

    async def _handle_generate(self, req: BaseRequest) -> ImageResponse:
        extra = req.extra
        number = extra.get("number", 1)
        reference_image = extra.get("reference_image", None)

        generator = self._resolve_generator(req)

        # ④ 循环生成
        # parse_params() 每次都要重新调用：它会生成新的 save_path（带 uuid）和新的随机种子，
        # 放在循环外只调一次会导致 batch 里每张图都写到同一个文件、用同一个种子，
        # 结果就是"生成的文件互相覆盖、内容还一模一样"。
        paths: list[str] = []
        for _ in range(number):
            generator.parse_params(req.extra)
            try:
                if reference_image:
                    path: Optional[Path] = await generator.generate_by_image()
                else:
                    path: Optional[Path] = await generator.generate()
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            if path:
                paths.append(to_media_url(path))

        if number > 0 and not paths:
            # generator.generate() 内部吞掉了异常，只返回 None；不能再把这种失败
            # 当成 ok=True 返回给前端，否则界面会显示"生成完成"但没有任何图片。
            raise HTTPException(status_code=500, detail="图片生成失败，请查看后端日志")

        return ImageResponse(
            ok=True,
            session_id=req.session_id,
            paths=paths,
        )