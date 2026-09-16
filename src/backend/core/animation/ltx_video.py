import gc
import os
import uuid
from pathlib import Path
from typing import List

from PIL import Image
from PIL.Image import Resampling
from huggingface_hub import snapshot_download, hf_hub_download

from src.shared.settings import PROJECT_ROOT
from src.backend.core.model_base import BaseAnimationGenerator
from src.backend.core.model_utils import huggingface_token, print_vram_usage, get_temp_dir


class LTXVideoGenerator(BaseAnimationGenerator):
    """
    基于 LTX-Video 的图像驱动动画生成器。
    """

    names = { "LTX-Video":{ "tag":"video"} }

    _DEFAULT_MODEL_ID = "Lightricks/LTX-Video-0.9.7-distilled"
    MODEL_NAME = "ltxv-2b-0.9.8-distilled-fp8.safetensors"

    _VALID_FRAME_COUNTS = [25, 33, 41, 49, 57, 65, 73, 81, 97, 121]
    _SAFE_RESOLUTIONS = {
        "low": (512, 320),
        "medium": (608, 384),
        "high": (704, 480),
    }

    _NEGATIVE_PROMPT = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "low resolution, static, no motion, choppy, artifacts, watermark, "
        "ghosting, double image, motion smear, flickering, morphing limbs, "
        "extra limbs, color bleeding, background movement, realistic photo, 3D render"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_dir = os.path.join(PROJECT_ROOT, "models", "animation", "ltx")
        self.model_path  = os.path.join(self.model_dir, self.MODEL_NAME)
        self.text_encoder_path = os.path.join(self.model_dir, "text_encoder")
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer")

    def _check_model_file(self):
        p = Path(self.model_dir)
        os.makedirs(str(p), exist_ok=True)

        if not p.exists():
            if self.model_id:
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(self.model_dir),
                    token=huggingface_token,
                    allow_patterns=["tokenizer/*",
                                    "text_encoder/*"
                                    ],
                )
            else:
                hf_hub_download(
                    repo_id="Lightricks/LTX-Video",
                    local_dir=self.model_dir,
                    filename=self.MODEL_NAME,
                    token=huggingface_token
                )
                snapshot_download(
                    repo_id=self._DEFAULT_MODEL_ID,
                    local_dir=str(self.model_dir),
                    token=huggingface_token,
                    allow_patterns=["tokenizer/*",
                                    "text_encoder/*"
                                    ],
                )

        print(f"✅ 模型文件：{p.name}（{p.stat().st_size/1024**3:.1f} GB）")

    def _load_model(self):
        # ① 先把 T5 文本编码器加载好
        text_encoder, tokenizer = self._load_text_encoder()
        self._load_fp16_offload(text_encoder, tokenizer)

        print("✅ 加载完成")
        print_vram_usage()

    def _load_text_encoder(self):
        from transformers import T5EncoderModel, T5Tokenizer
        print(f"   ① 加载 T5 文本编码器...")
        load_kwargs = dict(torch_dtype=self.torch.float16,
                           device_map="cpu")
        text_encoder = T5EncoderModel.from_pretrained(self.text_encoder_path, **load_kwargs)
        tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)
        return text_encoder, tokenizer

    def _load_int8_torchao(self, text_encoder, tokenizer):
        from diffusers import LTXImageToVideoPipeline

        print("   ② 从单文件加载 Transformer + VAE...")
        self.pipe = LTXImageToVideoPipeline.from_single_file(
            self.model_path,
            text_encoder=text_encoder,  # ← 注入已量化的 T5
            tokenizer=tokenizer,
            torch_dtype=self.torch.bfloat16,
        )
        self.pipe.vae = self.pipe.vae.to(dtype=self.torch.float32)
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

    def _load_fp16_offload(self, text_encoder, tokenizer):
        from diffusers import LTXImageToVideoPipeline

        print("   ② 从单文件加载 Transformer + VAE（float16）...")
        self.pipe = LTXImageToVideoPipeline.from_single_file(
            self.model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=self.torch.float16,
        )
        self.pipe.vae = self.pipe.vae.to(dtype=self.torch.float32)

        _orig_encode = self.pipe.vae.encode
        _orig_decode = self.pipe.vae.decode

        def _encode_f32(x, *args, **kwargs):
            return _orig_encode(x.to(dtype=self.torch.float32), *args, **kwargs)

        def _decode_f32(x, *args, **kwargs):
            return _orig_decode(x.to(dtype=self.torch.float32), *args, **kwargs)

        self.pipe.vae.encode = _encode_f32
        self.pipe.vae.decode = _decode_f32

        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

    def _load_full_precision(self, text_encoder, tokenizer):
        from diffusers import LTXImageToVideoPipeline

        self.pipe = LTXImageToVideoPipeline.from_single_file(
            self.model_path,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=self.torch.bfloat16,
        ).to(self.device)

        self.pipe.vae.enable_tiling()
        self.pipe.vae.to(self.torch.float32)
        self.pipe.vae.enable_slicing()
        self._try_enable_xformers()

    def _try_enable_xformers(self):
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("   ✔ xformers 注意力优化已启用")
        except Exception:
            pass

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """
        将输入图调整到目标分辨率（宽高均为 32 的倍数）。
        保持原始宽高比（不足处填充黑色），避免变形。
        """
        padding_info = {}
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.width / orig_w, self.height / orig_h)
        new_w = int(orig_w * scale) // 32 * 32
        new_h = int(orig_h * scale) // 32 * 32
        new_w = max(new_w, 32)
        new_h = max(new_h, 32)

        # 缩放
        image = image.resize((new_w, new_h), Resampling.LANCZOS)

        # 若缩放后尺寸与目标不一致，居中粘贴到目标画布
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

    def _nearest_valid_frames(self, num_frames: int) -> int:
        """LTX-Video 要求帧数为 8N+1，找最接近的合法值。"""
        valid = min(self._VALID_FRAME_COUNTS, key=lambda x: abs(x - num_frames))
        return valid

    async def generate_animation(self):
        """
        LTX-Video 核心推理，返回输出帧路径列表。

        参数
        ----
        reference_image      : 参考图（PIL）
        prompt               : 文字提示，描述期望的运动
        num_frames           : 期望帧数（自动对齐到 8N+1）
        output_dir           : 帧保存目录
        action_tag           : 输出文件前缀
        seed                 : 随机种子（复现用）
        num_inference_steps  : 去噪步数（distilled 版推荐 4–8，标准版推荐 25–50）
        guidance_scale       : CFG 强度（distilled 版推荐 1.0，标准版推荐 3.0–5.0）
        decode_timestep      : VAE 解码时步（默认 0.025）
        decode_noise_scale   : VAE 解码噪声（默认 0.0125）
        """
        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 帧数对齐
        num_frames = self._nearest_valid_frames(self.num_frames)
        # 图像预处理
        reference_image = Image.open(self.reference_image_path).convert("RGB")
        image, padding_info = self._preprocess_image(reference_image)
        print(f"🎬 开始生成：{num_frames} 帧 @ {self.width}×{self.height}, "
              f"steps={self.num_inference_steps}, cfg={self.guidance_scale}")

        with self.torch.inference_mode():
            output = self.pipe(
                image=image,
                prompt=self.prompt,
                negative_prompt=self._NEGATIVE_PROMPT,
                width=self.width,
                height=self.height,
                num_frames=self.num_frames,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                decode_timestep=self.decode_timestep,
                decode_noise_scale=self.decode_noise_scale,
                generator=self.generator,
            )

        frames: List[Image.Image] = output.frames[0]
        cropped_frames = []
        for frame in frames:
            frame = self._postprocess_image(frame, padding_info)
            if frame:
                cropped_frames.append(frame)

        # 保存帧
        frame_paths: List[Path] = []
        image_id = str(uuid.uuid4())
        for i, frame in enumerate(cropped_frames):
            frame_path = output_dir / f"{image_id}_{i:04d}.png"
            frame.save(frame_path)
            frame_paths.append(frame_path)

        # 保存为 GIF
        gif_path = output_dir / f"{image_id}.gif"
        cropped_frames[0].save(
            gif_path,
            save_all=True,
            append_images=cropped_frames[1:],
            duration=int(1000 / 24),
            loop=0,
        )
        print(f"✅ 已保存 {len(frames)} 帧到 {output_dir}")
        print(f"   GIF 预览：{gif_path}")

        # 主动释放推理缓存
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
        return frame_paths, gif_path

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.reference_image_path = raw.get("reference_image_path", "")

        self.width, self.height = self._SAFE_RESOLUTIONS.get(raw.get("resolution", ""), (512, 320))
        self.num_frames = raw.get("num_frames", 121)
        self.prompt = raw.get("content")
        
        self.num_inference_steps = raw.get("num_inference_steps", 8)
        self.guidance_scale = raw.get("guidance_scale", 1.0)

        self.seed = raw.get("seed", 42)
        self.decode_timestep = raw.get("decode_timestep", 0.05)
        self.decode_noise_scale = raw.get("decode_timestep", 0.025)

        self.generator = self.torch.Generator(device=self.device).manual_seed(self.seed)
