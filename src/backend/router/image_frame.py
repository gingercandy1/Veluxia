import asyncio
import uuid
from typing import List
from pathlib import Path

from fastapi import HTTPException

from src.backend.router_base import BaseRouter
from src.backend.core.exceptions import GeneratorBusyError
from src.backend.core.model_base import GeneratorFactory
from src.backend.core.image_frame.sprite_export import export_frames, export_sprite_sheet
from src.backend.core.model_utils import get_temp_dir, to_media_url
from src.shared.schemas import AnimationResponse, BaseRequest, SpriteSheetResponse
from src.shared.enum_type import FactoryType


class ImageFrameRouter(BaseRouter):
    prefix       = "/image_frame"
    tags         = ["ImageFrame"]
    factory_type = FactoryType.ImageFrame

    def _register_routes(self) -> None:

        @self.router.post(
            "/interpolate",
            response_model=AnimationResponse,
            summary="帧插值（2 张 → 中间帧；2 张以上 → 序列）",
        )
        async def interpolate(req: BaseRequest) -> AnimationResponse:
            try:
                with GeneratorFactory.acquire(FactoryType.ImageFrame, req.model_name) as generator:
                    generator.ensure_model_loaded()
                    generator.parse_params(req.extra)
                    frame_paths: List[Path] = await generator.generate()
            except GeneratorBusyError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

            return AnimationResponse(
                ok=bool(frame_paths),
                session_id=req.session_id,
                frame_paths=[to_media_url(p) for p in frame_paths],
            )

        @self.router.post(
            "/export",
            response_model=SpriteSheetResponse,
            summary="序列帧导出：精灵图 + atlas JSON + 编号 PNG",
        )
        async def export(req: BaseRequest) -> SpriteSheetResponse:
            # 只用 Pillow、不加载模型，所以不走生成器租约，也不会卸载正在驻留的模型
            try:
                result = await asyncio.to_thread(self._export, req.extra)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            sheet, frame_paths = result
            return SpriteSheetResponse(
                ok=True,
                session_id=req.session_id,
                sheet_path=to_media_url(sheet.sheet_path),
                atlas_path=to_media_url(sheet.atlas_path),
                frame_paths=[to_media_url(p) for p in frame_paths],
                columns=sheet.columns,
                rows=sheet.rows,
            )

    @staticmethod
    def _export(extra: dict):
        name = extra.get("name") or "sprite"
        output_dir = Path(get_temp_dir(extra.get("output_dir") or f"sprites/{name}_{uuid.uuid4().hex[:8]}"))
        frame_paths = extra.get("frame_paths", [])
        trim = bool(extra.get("trim", False))
        sheet = export_sprite_sheet(
            frame_paths, output_dir, name, columns=int(extra.get("columns", 0)),
            padding=int(extra.get("padding", 0)), trim=trim, fps=int(extra.get("fps", 12)))
        frames = export_frames(frame_paths, output_dir / "frames", name, trim=trim)
        return sheet, frames
