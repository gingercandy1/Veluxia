import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image
from PIL.Image import Resampling

from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class FluxSchnellGenerator(BaseImageGenerator):
    """Flux.1-schnell 图片生成器"""
    names = { "Flux.1-schnell":{ "tag":"image"}}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.model_id:
            self.nvfp4_local = Path(PROJECT_ROOT)/ "models" / "image" / "flux2-klein-4b-nvfp4"
            self.nvfp4_path   = self.nvfp4_local / "flux-2-klein-4b-nvfp4.safetensors"
            self.base_local = Path(PROJECT_ROOT) / "models" / "image" / "flux2-klein-4b-base"
        else:
            self.base_local = Path(PROJECT_ROOT) / "models" / "image" / "flux2-klein-4b-base"
            self.nvfp4_path = self.base_local / "flux-2-klein-4b-nvfp4.safetensors"

    def _check_model_file(self):
        from huggingface_hub import hf_hub_download, snapshot_download
        if self.model_id:
            if not self.base_local.exists():
                print("⏬ 正在下载 base pipeline 组件（VAE / 文本编码器）...")
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(self.base_local),
                    token=huggingface_token,
                )
                print("✅ base 组件下载完成")

        else:
            if not self.nvfp4_local.exists():
                print("⏬ 正在下载 NVFP4 transformer 权重...")
                hf_hub_download(
                    repo_id="black-forest-labs/FLUX.2-klein-4b-nvfp4",
                    local_dir=str(self.nvfp4_local),
                    filename="flux-2-klein-4b-nvfp4.safetensors",
                    token=huggingface_token,
                )
                print("✅ NVFP4 transformer 下载完成")

            if not self.base_local.exists():
                print("⏬ 正在下载 base pipeline 组件（VAE / 文本编码器）...")
                snapshot_download(
                    repo_id="black-forest-labs/FLUX.2-klein-4B",
                    local_dir=str(self.base_local),
                    token=huggingface_token,
                )
                print("✅ base 组件下载完成")

    def _load_model(self):
        if self.pipe is not None:
            return

        from diffusers import Flux2KleinPipeline
        from safetensors.torch import load_file

        print("🔧 正在加载模型（首次较慢）...")
        dtype = self.torch.bfloat16
        self.pipe = Flux2KleinPipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=dtype,
            local_files_only=True,
        )

        state_dict = load_file(str(self.nvfp4_path))
        self.pipe.transformer.load_state_dict(state_dict, strict=False)
        self.pipe.transformer.to(dtype)

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.torch.cuda.empty_cache()

    def unload_model(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("✅ 模型已卸载，显存已释放")

    @staticmethod
    def _prepare_ref_image(
            image_path: str,
            width: int,
            height: int,
            preprocess: str = "resize",
    ) -> Image.Image:
        """
        将参考图缩放到目标分辨率。
        preprocess:
          "resize"     — 直接缩放（可能改变比例）
          "crop"       — 等比缩放后中心裁剪
          "pad"        — 等比缩放后两侧填黑
        """
        img = Image.open(image_path).convert("RGB")

        if preprocess == "resize":
            img = img.resize((width, height), Resampling.LANCZOS)

        elif preprocess == "crop":
            scale = max(width / img.width, height / img.height)
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Resampling.LANCZOS)
            left = (nw - width) // 2
            top = (nh - height) // 2
            img = img.crop((left, top, left + width, top + height))

        elif preprocess == "pad":
            scale = min(width / img.width, height / img.height)
            nw, nh = int(img.width * scale), int(img.height * scale)
            img = img.resize((nw, nh), Resampling.LANCZOS)
            canvas = Image.new("RGB", (width, height), (0, 0, 0))
            canvas.paste(img, ((width - nw) // 2, (height - nh) // 2))
            img = canvas

        return img

    async def generate_by_image(self) -> Optional[Path|None]:
        if not self.ref_image_path: return Path()
        self.ensure_model_loaded()

        ref_img = self._prepare_ref_image(self.ref_image_path, self.width, self.height)
        with self.torch.inference_mode():
            try:
                image = self.pipe(image=ref_img,
                                  width=self.width,
                                  height=self.height,
                                  prompt=self.prompt,
                                  num_inference_steps=self.num_inference_steps,
                                  guidance_scale=self.guidance_scale,
                                  generator=self.generator
                                  ).images[0]
                image.save(self.save_path)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
            except Exception as e:
                print(f"❌ 生成失败: {e}")
            finally:
                self.torch.cuda.empty_cache()
        return None

    async def generate(self) ->  Optional[Path | None]:
        self.ensure_model_loaded()

        with self.torch.inference_mode():
            try:
                image = self.pipe(width=self.width,
                                  height=self.height,
                                  prompt=self.prompt,
                                  num_inference_steps=self.num_inference_steps,
                                  guidance_scale=self.guidance_scale,
                                  generator=self.generator).images[0]
                image.save(self.save_path)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
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
        self.guidance_scale = raw.get("guidance_scale", 1.0)
        self.num_inference_steps = raw.get("num_inference_steps", 4)

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"flux_{str(uuid.uuid4())}.png"
