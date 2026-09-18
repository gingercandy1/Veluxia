import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class SDXLGenerator(BaseImageGenerator):
    """SDXL-base 图片生成器（Stability Community License，可商用）"""
    model_dir = "sdxl-base-1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_img2img = None
        self.base_local = Path(PROJECT_ROOT) / "models" / "image" / self.model_dir
        if not self.model_id:
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    def _check_model_file(self):
        from huggingface_hub import snapshot_download
        if not self.base_local.exists():
            print("⏬ 正在下载 SDXL-base 权重...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.base_local),
                token=huggingface_token,
            )
            print("✅ SDXL-base 下载完成")

    def _load_model(self):
        if self.pipe is not None:
            return

        from diffusers import StableDiffusionXLPipeline

        print("🔧 正在加载 SDXL（首次较慢）...")
        dtype = self.torch.bfloat16
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.torch.cuda.empty_cache()

    def _load_img2img(self):
        if self.pipe_img2img is not None:
            return
        from diffusers import StableDiffusionXLImg2ImgPipeline
        self.ensure_model_loaded()
        self.pipe_img2img = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe)

    def unload_model(self):
        for attr in ("pipe_img2img", "pipe"):
            if getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self.torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✅ SDXL 已卸载，显存已释放")

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()

        with self.torch.inference_mode():
            try:
                image = self.pipe(width=self.width,
                                  height=self.height,
                                  prompt=self.prompt,
                                  num_inference_steps=self.num_inference_steps,
                                  guidance_scale=self.guidance_scale,
                                  generator=self.generator,
                                  callback_on_step_end=self.make_cancel_callback(),
                                  ).images[0]
                image.save(self.save_path)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
            except GenerationCancelled:
                print("🛑 生成已被用户取消")
                raise
            except Exception as e:
                print(f"❌ 生成失败: {e}")
            finally:
                self.torch.cuda.empty_cache()
        return None

    async def generate_by_image(self) -> Optional[Path | None]:
        if not self.ref_image_path:
            return Path()
        self._load_img2img()

        ref_img = Image.open(self.ref_image_path).convert("RGB")
        with self.torch.inference_mode():
            try:
                image = self.pipe_img2img(prompt=self.prompt,
                                          image=ref_img,
                                          strength=self.strength,
                                          num_inference_steps=self.num_inference_steps,
                                          guidance_scale=self.guidance_scale,
                                          generator=self.generator,
                                          callback_on_step_end=self.make_cancel_callback(),
                                          ).images[0]
                image.save(self.save_path)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
            except GenerationCancelled:
                print("🛑 生成已被用户取消")
                raise
            except Exception as e:
                print(f"❌ 生成失败: {e}")
            finally:
                self.torch.cuda.empty_cache()
        return None

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = self.get_output_dir(self.output_dir)

        self.ref_image_path = raw.get("reference_image", "")
        self.width = raw.get("width", 1024)
        self.height = raw.get("height", 1024)
        self.prompt = raw.get("content", "")
        self.guidance_scale = raw.get("guidance_scale", 5.0)
        self.num_inference_steps = raw.get("num_inference_steps", 30)
        self.strength = raw.get("strength", 0.6)

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"sdxl_{str(uuid.uuid4())}.png"
