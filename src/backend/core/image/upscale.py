import os
import urllib.request
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from src.backend.core.image.tileable import validate_tile_mode
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import get_temp_dir
from src.shared.settings import PROJECT_ROOT

# 输出像素上限：整张图的张量都在内存里，超过后 CPU 内存先爆
_MAX_OUTPUT_PIXELS = 8192 * 8192


def tiled_upscale(model: Callable, tensor, scale: int, tile: int, pad: int,
                  on_tile: Optional[Callable[[], None]] = None):
    """分块超分：8GB 显存放不下整张大图，按 tile 切块推理，每块外扩 pad 像素再裁掉，避免接缝。

    tensor: (1, 3, H, W)，值域 0~1；返回 (1, 3, H*scale, W*scale)。
    on_tile 每块推理前调用，用于响应取消。
    """
    import torch

    _, _, height, width = tensor.shape
    out = tensor.new_zeros((1, 3, height * scale, width * scale))
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            if on_tile is not None:
                on_tile()
            y0, x0 = max(y - pad, 0), max(x - pad, 0)
            y1, x1 = min(y + tile + pad, height), min(x + tile + pad, width)
            with torch.no_grad():
                part = model(tensor[:, :, y0:y1, x0:x1])
            tile_h, tile_w = min(tile, height - y), min(tile, width - x)
            crop_y, crop_x = (y - y0) * scale, (x - x0) * scale
            out[:, :, y * scale:(y + tile_h) * scale, x * scale:(x + tile_w) * scale] = \
                part[:, :, crop_y:crop_y + tile_h * scale, crop_x:crop_x + tile_w * scale]
    return out


class UpscaleGenerator(BaseImageGenerator):
    """超分放大基类：spandrel 加载 Real-ESRGAN 系列权重，输入图 → 放大后的 PNG。

    单例按类区分，所以每个权重对应一个子类，只需填 weight_url / weight_filename。
    """
    weight_url = ""
    weight_filename = ""
    TILE = 256
    PAD = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_path = Path(PROJECT_ROOT) / "models" / "upscale" / self.weight_filename

    def _check_model_file(self):
        if self.weight_path.exists():
            return
        print(f"⏬ 正在下载超分权重 {self.weight_filename} ...")
        self.weight_path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.weight_path.with_suffix(".part")
        with urllib.request.urlopen(self.weight_url, timeout=60) as response, open(partial, "wb") as f:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                self.report_load_stage(
                    "download", progress=done / total if total else 0.0, detail=self.weight_filename)
        # 下载到一半失败不能留下残缺的 .pth，否则下次会当成已下载好的权重去加载
        os.replace(partial, self.weight_path)
        print("✅ 超分权重下载完成")

    def _load_model(self):
        if self.pipe is not None:
            return
        from spandrel import ModelLoader
        descriptor = ModelLoader().load_from_file(str(self.weight_path))
        descriptor.to(self.device).eval()
        if self.device == "cuda" and descriptor.supports_half:
            descriptor.half()
        self.pipe = descriptor
        print(f"✅ 超分模型已加载: {self.weight_filename}（x{descriptor.scale}）")

    def _upscale_rgb(self, image: Image.Image) -> Image.Image:
        import numpy as np
        torch = self.torch

        scale = self.pipe.scale
        dtype = next(self.pipe.model.parameters()).dtype
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        # 无缝图的边缘在模型眼里是图像边界（零填充），放大后首尾会对不上；
        # 先用对侧内容循环补边再放大，裁掉补的部分，首尾仍然相接
        pad_h = self.PAD if self.tile_mode in ("both", "vertical") else 0
        pad_w = self.PAD if self.tile_mode in ("both", "horizontal") else 0
        array = np.pad(array, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="wrap")
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype)
        result = tiled_upscale(self.pipe, tensor, scale, self.TILE, self.PAD, self.check_cancelled)
        height, width = result.shape[-2:]
        result = result[..., pad_h * scale:height - pad_h * scale, pad_w * scale:width - pad_w * scale]
        result = (result.squeeze(0).permute(1, 2, 0).float().clamp(0, 1) * 255).round().byte().cpu().numpy()
        return Image.fromarray(result)

    def _upscale(self, image: Image.Image) -> Image.Image:
        scale = self.pipe.scale
        if image.width * image.height * scale * scale > _MAX_OUTPUT_PIXELS:
            raise ValueError(f"输出超过 {_MAX_OUTPUT_PIXELS // 1_000_000} 百万像素上限，请先缩小输入图")

        result = self._upscale_rgb(image)
        # 先缩 RGB 再合 alpha：PIL 缩放 RGBA 会预乘 alpha，全透明像素的 RGB 被清成黑色，
        # 引擎双线性采样时边缘会混进黑色、出现暗边
        if self.outscale and self.outscale != scale:
            size = (round(image.width * self.outscale), round(image.height * self.outscale))
            result = result.resize(size, Image.LANCZOS)
        if image.mode in ("RGBA", "LA"):
            # 游戏素材常带透明通道；模型只吃 RGB，alpha 单独放大再合回去
            alpha = image.getchannel("A").resize(result.size, Image.LANCZOS)
            result.putalpha(alpha)
        return result

    async def generate(self) -> Optional[Path]:
        self.ensure_model_loaded()
        if not self.input_path:
            raise ValueError("超分需要输入图，请先添加一张图片附件")
        with Image.open(self.input_path) as image:
            result = self._upscale(image)
        result.save(self.save_path)
        print(f"✅ 超分完成: {self.save_path.name}（{result.width}x{result.height}）")
        return self.save_path

    async def generate_by_image(self) -> Optional[Path]:
        return await self.generate()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = Path(self.output_dir) / f"upscale_{uuid.uuid4()}.png"
        self.input_path = raw.get("input_path", "") or raw.get("reference_image", "")
        # 0 表示保持模型原生倍率
        self.outscale = float(raw.get("outscale", 0))
        self.tile_mode = validate_tile_mode(raw.get("tile_mode", "off"))


class RealEsrganX4PlusGenerator(UpscaleGenerator):
    """Real-ESRGAN x4plus（BSD-3）：通用照片 / 写实画面 4 倍放大"""
    weight_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    weight_filename = "RealESRGAN_x4plus.pth"


class RealEsrganAnimeGenerator(UpscaleGenerator):
    """Real-ESRGAN anime 6B（BSD-3）：插画 / 手绘 / 像素以外的 2D 游戏素材，体积小、线条干净"""
    weight_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    weight_filename = "RealESRGAN_x4plus_anime_6B.pth"
