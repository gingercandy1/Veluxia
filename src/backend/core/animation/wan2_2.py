import gc
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from src.shared.settings import PROJECT_ROOT
from src.backend.core.diffusers_utils import apply_offload, gguf_filename, load_gguf_transformer
from src.backend.core.model_base import BaseAnimationGenerator
from src.backend.core.model_utils import ensure_file, ensure_snapshot, print_vram_usage, get_temp_dir

class Wan2VideoGenerator(BaseAnimationGenerator):
    _BASE_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

    # Wan2.x 帧数要求：4N+1（17, 25, 33, 49, 65, 81）
    _VALID_FRAME_COUNTS = [17, 25, 33, 49, 65, 81]

    # 480P 标准分辨率（Blackwell 8GB 安全上限）
    _DEFAULT_WIDTH = 320
    _DEFAULT_HEIGHT = 320
    # TI2V-5B 原生训练在 704p 附近，320 离分布太远时会乱画背景、肢体跑偏；
    # 宽高需是 32 的倍数（VAE 16 倍下采样 × patch 2）
    _RESOLUTIONS = {
        "low": (320, 320),
        "medium": (480, 480),
        "high": (704, 704),
    }

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
        # models.json 给了 .gguf 文件名才走量化链路：官方 transformer 是 fp32（约 19GB），
        # 加载时连同文本编码器一起会撑爆 32GB 内存，GGUF 在内存和显存里都保持压缩
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = (getattr(self, "model_extra", None) or {}).get("gguf_repo_id", "")
        self.gguf_local = Path(PROJECT_ROOT) / "models" / "animation" / "wan2-gguf"
        if not self.model_id:
            self.model_id = self._BASE_REPO_ID

    def _check_model_file(self):
        if self.gguf_filename:
            ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)
        ensure_snapshot(self.model_id, self.model_dir)

    def _load_model(self):
        print(f"🔄 正在加载 Wan2.2-TI2V...")
        print(f"   ② 组装 Pipeline（VAE / T5 / CLIP）...")
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        import torch  # 后台注册线程早已 import 过，这里只是拿缓存，不会重新触发加载
        # 与官方示例一致，VAE 固定 fp32：bf16 解码有精度损失，fp32 只多占约 1GB 内存
        pipe_kwargs = {"vae": AutoencoderKLWan.from_pretrained(
            str(self.model_dir), subfolder="vae", torch_dtype=torch.float32)}
        if self.gguf_filename:
            from diffusers import WanTransformer3DModel
            pipe_kwargs["transformer"] = load_gguf_transformer(
                WanTransformer3DModel, self.gguf_local / self.gguf_filename, self.model_dir)
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16,
            **pipe_kwargs,
        )
        apply_offload(self.pipe, "model", self.device)

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
                num_frames=num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
                callback_on_step_end=self.make_cancel_callback(),
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
        self.reference_image_path = raw.get("reference_image_path", raw.get("reference_image", ""))

        self.width, self.height = self._RESOLUTIONS.get(
            raw.get("resolution", ""), (self._DEFAULT_WIDTH, self._DEFAULT_HEIGHT))
        # 资源包按素材比例直接指定尺寸（如竖版角色动作），优先于预设档位
        if "width" in raw and "height" in raw:
            self.width, self.height = int(raw["width"]), int(raw["height"])
            if self.width % 32 or self.height % 32:
                raise ValueError(f"Wan2.2 的宽高必须是 32 的倍数：{self.width}×{self.height}")
        self.num_frames = raw.get("num_frames", 25)
        self.prompt = raw.get("content")

        self.num_inference_steps = raw.get("num_inference_steps", 25)
        self.guidance_scale = raw.get("guidance_scale", 5.0)

        self.seed = raw.get("seed", 42)
        self.generator = self.torch.Generator(device=self.device).manual_seed(self.seed)


