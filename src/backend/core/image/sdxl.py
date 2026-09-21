import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.image.tileable import apply_tile_mode, validate_tile_mode
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class SDXLGenerator(BaseImageGenerator):
    """SDXL-base 图片生成器（Stability Community License，可商用）"""
    model_dir = "sdxl-base-1.0"

    _model_attrs = ("pipe_img2img", "pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_img2img = None
        self.base_local = Path(PROJECT_ROOT) / "models" / "image" / self.model_dir
        if not self.model_id:
            self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    # 官方仓库还带 fp32 权重、两个单文件 checkpoint、ONNX 等，全下要几十 GB；
    # 管线只用 fp16 变体（约 7GB），8GB 显存也只能用这一档。
    _DOWNLOAD_PATTERNS = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "text_encoder/config.json",
        "text_encoder/*.fp16.safetensors",
        "text_encoder_2/config.json",
        "text_encoder_2/*.fp16.safetensors",
        "unet/config.json",
        "unet/*.fp16.safetensors",
        "vae/config.json",
        "vae/*.fp16.safetensors",
    ]
    _MARKER_FILE = "unet/diffusion_pytorch_model.fp16.safetensors"

    def _check_model_file(self):
        from huggingface_hub import snapshot_download
        # 不能只判断目录是否存在：之前完整仓库下载中断留下的目录不代表 fp16 权重齐全
        if not (self.base_local / self._MARKER_FILE).exists():
            print("⏬ 正在下载 SDXL-base fp16 权重（约 7GB）...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.base_local),
                allow_patterns=self._DOWNLOAD_PATTERNS,
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
            variant="fp16",
            use_safetensors=True,
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

    def _apply_tile_mode(self):
        apply_tile_mode((self.pipe.unet, self.pipe.vae), self.tile_mode)
        if self.tile_mode == "off":
            self.pipe.vae.enable_tiling()
        else:
            self.pipe.vae.disable_tiling()

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()
        self._apply_tile_mode()

        with self.torch.inference_mode():
            try:
                image = self.pipe(width=self.width,
                                  height=self.height,
                                  prompt=self.prompt,
                                  negative_prompt=self.negative_prompt,
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
        self._apply_tile_mode()

        ref_img = Image.open(self.ref_image_path).convert("RGB")
        with self.torch.inference_mode():
            try:
                image = self.pipe_img2img(prompt=self.prompt,
                                          image=ref_img,
                                          negative_prompt=self.negative_prompt,
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
        # 空串按“不使用”处理；SDXL-base 很依赖负面词压噪点和杂乱，默认值由参数面板提供
        self.negative_prompt = raw.get("negative_prompt", "").strip() or None
        self.guidance_scale = raw.get("guidance_scale", 5.0)
        self.num_inference_steps = raw.get("num_inference_steps", 30)
        self.strength = raw.get("strength", 0.6)
        self.tile_mode = validate_tile_mode(raw.get("tile_mode", "off"))

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"sdxl_{str(uuid.uuid4())}.png"
