import gc
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from huggingface_hub import snapshot_download

from core import preloader
from src.shared.settings import PROJECT_ROOT
from src.backend.core.model_base import BaseAnimationGenerator
from src.backend.core.model_utils import huggingface_token, print_vram_usage, get_temp_dir

class Wan2VideoGenerator(BaseAnimationGenerator):
    names = {
                "Wan2.2-TI2V":{ "tag":"video"}
            }

    _BASE_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Wan2.x 帧数要求：4N+1（17, 25, 33, 49, 65, 81）
    _VALID_FRAME_COUNTS = [17, 25, 33, 49, 65, 81]

    # 480P 标准分辨率（Blackwell 8GB 安全上限）
    _DEFAULT_WIDTH = 320
    _DEFAULT_HEIGHT = 320

    # 固定负面提示词
    _NEGATIVE_PROMPT = (
        "static, no motion, worst quality, inconsistent motion, "
        "blurry, jittery, distorted, low resolution, choppy, artifacts, watermark, "
        "artifacts, watermark, text, logo"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.width  = self._DEFAULT_WIDTH
        self.height = self._DEFAULT_HEIGHT
        # 本地存储目录
        self.model_dir  = Path(PROJECT_ROOT) / "models" / "animation" / "wan2"

    def _check_model_file(self):
        if self.model_dir.exists():
            return

        print(f"📥 未找到基础组件，开始下载...")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if self.model_id:
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.model_dir),
                token=huggingface_token,
            )
        else:
            snapshot_download(
                repo_id=self._BASE_REPO_ID,
                local_dir=str(self.model_dir),
                token=huggingface_token,
            )
        print(f"✅ 基础组件下载完成")

    def _load_model(self):
        print(f"🔄 正在加载 Wan2.2-TI2V...")
        print(f"   ② 组装 Pipeline（VAE / T5 / CLIP）...")
        from diffusers import WanImageToVideoPipeline
        torch = preloader.get("torch")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16,
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        print("✅ 加载完成")
        print_vram_usage()

    def _nearest_valid_frames(self, num_frames: int) -> int:
        """Wan2.x 要求帧数为 4N+1，找最接近的合法值。"""
        return min(self._VALID_FRAME_COUNTS, key=lambda x: abs(x - num_frames))

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """
        将输入图调整到目标分辨率（宽高均为 16 的倍数），保持宽高比，不足处填黑。
        同时自动裁掉纯黑边框，避免大面积黑边压制运动生成。
        """
        padding_info = {}
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.width / orig_w, self.height / orig_h)
        new_w = max(int(orig_w * scale) // 16 * 16, 16)
        new_h = max(int(orig_h * scale) // 16 * 16, 16)

        image = image.resize((new_w, new_h), Resampling.LANCZOS)

        # 居中填充到目标分辨率
        if new_w != self.width or new_h != self.height:
            canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            x_off = (self.width  - new_w) // 2
            y_off = (self.height - new_h) // 2
            canvas.paste(image, (x_off, y_off))
            image = canvas

            padding_info = {
                "original_size": (orig_w, orig_h),
                "resized_size": (new_w, new_h),
                "x_offset": x_off,
                "y_offset": y_off,
                "padded_size": (self.width, self.height)
            }

        return image, padding_info

    def _postprocess_image(self, image: Image.Image, padding_info) -> Image.Image:
        if not padding_info: return image

        x_offset = padding_info["x_offset"]
        y_offset = padding_info["y_offset"]
        resized_w, resized_h = padding_info["resized_size"]

        # 裁掉黑色边框，只保留有内容的区域
        cropped = image.crop((x_offset, y_offset, x_offset + resized_w, y_offset + resized_h))

        # 重新缩放到原始输入分辨率
        # original_w, original_h = padding_info["original_size"]
        # cropped = cropped.resize((original_w, original_h), Resampling.LANCZOS)

        return cropped

    def _encode_first_frame(self, image: Image.Image, height: int, width: int) -> "torch.Tensor":
        """将参考图编码为 VAE latent，作为第一帧条件"""
        img = image.convert("RGB").resize((width, height), Resampling.LANCZOS)
        img_tensor = self.torch.from_numpy(
            np.array(img).astype(np.float32) / 127.5 - 1.0
        ).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)

        img_tensor = img_tensor.to(device="cuda", dtype=self.torch.float32)

        with self.torch.inference_mode():
            latent = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            latent = latent * self.pipe.vae.config.scaling_factor
        return latent

    async def generate_animation(self):
        """-
        reference_image     : 参考图（PIL）
        prompt              : 文字提示，描述期望的运动
        num_frames          : 期望帧数（自动对齐到 4N+1，最多 81）
        output_dir          : 帧保存目录
        seed                : 随机种子（复现用）
        num_inference_steps : 去噪步数（推荐 20–30）
        guidance_scale      : CFG 强度（推荐 3.0–7.0，越高运动越明显）
        """
        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_frames = self._nearest_valid_frames(self.num_frames)

        reference_image = Image.open(self.reference_image_path).convert("RGB")
        image, padding_info = self._preprocess_image(reference_image)
        print(f"🎬 开始生成：{num_frames} 帧 @ {self.width}×{self.height}, "
              f"steps={self.num_inference_steps}, cfg={self.guidance_scale}")

        with self.torch.inference_mode():
            output = self.pipe(
                image=image,
                prompt=self.prompt,
                negative_prompt=self._NEGATIVE_PROMPT,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
            )

        frames: List[Image.Image] = []
        for f in output.frames[0]:
            if isinstance(f, Image.Image):
                frames.append(f)
            else:
                if f.dtype != np.uint8:
                    f = (f * 255).clip(0, 255).astype(np.uint8)
                frames.append(Image.fromarray(f))

        cropped_frames = []
        for frame in frames:
            frame = self._postprocess_image(frame, padding_info)
            if frame:
                cropped_frames.append(frame)

        # 保存帧
        image_id = str(uuid.uuid4())
        frame_paths: List[Path] = []
        for i, frame in enumerate(cropped_frames):
            frame_path = output_dir / f"{image_id}_{i:04d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # GIF 预览
        gif_path = output_dir / f"{image_id}.gif"
        cropped_frames[0].save(
            gif_path,
            save_all=True,
            append_images=cropped_frames[1:],
            duration=int(1000 / 16),  # 16 FPS
            loop=0,
        )
        print(f"✅ 已保存 {len(frames)} 帧到 {output_dir}")
        print(f"   GIF 预览：{gif_path}")

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
        return frame_paths, gif_path

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.reference_image_path = raw.get("reference_image_path", "")

        self.num_frames = raw.get("num_frames", 25)
        self.prompt = raw.get("content")

        self.num_inference_steps = raw.get("num_inference_steps", 25)
        self.guidance_scale = raw.get("guidance_scale", 5.0)

        self.seed = raw.get("seed", 42)
        self.generator = self.torch.Generator(device=self.device).manual_seed(self.seed)


