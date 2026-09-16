import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import get_temp_dir


class BgRemovalGenerator(BaseImageGenerator):
    """去背景：rembg + u2net（MIT，可商用），输入图 → 透明底 PNG"""
    names = {"rembg-u2net": {"tag": "edit"}}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None

    def _check_model_file(self):
        # u2net.onnx 由 rembg 首次运行时自动下载到 ~/.u2net，无需预处理
        pass

    def _load_model(self):
        if self.session is not None:
            return
        from rembg import new_session
        print("🔧 正在加载 rembg-u2net（首次会自动下载 onnx 权重）...")
        self.session = new_session("u2net")

    def unload_model(self):
        self.session = None
        import gc
        gc.collect()
        print("✅ rembg-u2net 会话已释放")

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()
        from rembg import remove

        try:
            img = Image.open(self.input_path)
            out = remove(img, session=self.session)
            out.save(self.save_path)
            print(f"✅ 去背景完成: {self.save_path.name}")
            return self.save_path
        except Exception as e:
            print(f"❌ 去背景失败: {e}")
        return None

    async def generate_by_image(self) -> Optional[Path | None]:
        # 去背景语义上就是图→图：reference_image 可作为输入源
        if self.ref_image_path and not self.input_path:
            self.input_path = self.ref_image_path
        return await self.generate()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = self.get_output_dir(self.output_dir)

        self.input_path = raw.get("input_path", "") or raw.get("reference_image", "")
        self.ref_image_path = raw.get("reference_image", "")

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"nobg_{str(uuid.uuid4())}.png"
