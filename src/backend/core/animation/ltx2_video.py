import gc
import uuid
from pathlib import Path
from typing import List

from PIL import Image
from PIL.Image import Resampling
from huggingface_hub import snapshot_download

from src.shared.settings import PROJECT_ROOT
from src.backend.core.model_base import BaseAnimationGenerator
from src.backend.core.model_utils import huggingface_token, print_vram_usage, get_temp_dir


class LTX2VideoGenerator(BaseAnimationGenerator):
    """
    基于 LTX-2 系列（Lightricks 新一代音画联合生成模型）的图像驱动动画生成器。
    与旧版 LTX-Video（0.9.x）不同，LTX-2 使用 from_pretrained 整仓加载，
    并同步生成视频与音轨，输出为带音频的 mp4。
    """

    _REPO_IDS = {
        "LTX-2.3": "Lightricks/LTX-2.3-Diffusers",
        "LTX-2.5": "Lightricks/LTX-2.5-Diffusers",
    }

    _VALID_FRAME_COUNTS = [25, 33, 41, 49, 57, 65, 73, 81, 97, 121]
    _SAFE_RESOLUTIONS = {
        "low": (512, 320),
        "medium": (768, 512),
        "high": (960, 640),
    }

    _NEGATIVE_PROMPT = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "low resolution, static, no motion, choppy, artifacts, watermark, "
        "ghosting, double image, motion smear, flickering, morphing limbs, "
        "extra limbs, color bleeding, background movement"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        repo_dir = "ltx2_3" if self.model_name == "LTX-2.3" else "ltx2_5"
        self.model_dir = Path(PROJECT_ROOT) / "models" / "animation" / repo_dir

    def _check_model_file(self):
        if self.model_dir.exists() and any(self.model_dir.iterdir()):
            print(f"✅ 模型目录已存在：{self.model_dir}")
            return

        self.model_dir.mkdir(parents=True, exist_ok=True)
        repo_id = self.model_id or self._REPO_IDS[self.model_name]
        print(f"📥 未找到 {self.model_name} 权重，开始下载 {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.model_dir),
            token=huggingface_token,
        )
        print(f"✅ {self.model_name} 权重下载完成")

    def _load_model(self):
        from diffusers import LTX2ImageToVideoPipeline

        print(f"🔄 正在加载 {self.model_name}（{self._REPO_IDS[self.model_name]}）...")
        self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
            str(self.model_dir),
            dtype=self.torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_tiling()

        print("✅ 加载完成")
        print_vram_usage()

    def _preprocess_image(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """将输入图缩放到目标分辨率（宽高均为 32 的倍数），保持宽高比，不足处填黑。"""
        padding_info = {}
        image = image.convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.width / orig_w, self.height / orig_h)
        new_w = max(int(orig_w * scale) // 32 * 32, 32)
        new_h = max(int(orig_h * scale) // 32 * 32, 32)

        image = image.resize((new_w, new_h), Resampling.LANCZOS)

        if new_w != self.width or new_h != self.height:
            canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            x_off = (self.width - new_w) // 2
            y_off = (self.height - new_h) // 2
            canvas.paste(image, (x_off, y_off))
            image = canvas

            padding_info = {
                "resized_size": (new_w, new_h),
                "x_offset": x_off,
                "y_offset": y_off,
            }
        return image, padding_info

    def _postprocess_image(self, image: Image.Image, padding_info) -> Image.Image:
        if not padding_info:
            return image
        x_offset = padding_info["x_offset"]
        y_offset = padding_info["y_offset"]
        resized_w, resized_h = padding_info["resized_size"]
        return image.crop((x_offset, y_offset, x_offset + resized_w, y_offset + resized_h))

    def _nearest_valid_frames(self, num_frames: int) -> int:
        """LTX-2 要求帧数为 8N+1，找最接近的合法值。"""
        return min(self._VALID_FRAME_COUNTS, key=lambda x: abs(x - num_frames))

    async def generate_animation(self):
        """
        LTX-2 图生视频（含音频）核心推理，返回输出帧路径列表 + 带音频的 mp4 路径。

        参数
        ----
        reference_image      : 参考图（PIL）
        prompt               : 文字提示，描述期望的画面与音效
        num_frames           : 期望帧数（自动对齐到 8N+1）
        output_dir           : 帧/视频保存目录
        seed                 : 随机种子（复现用）
        num_inference_steps  : 去噪步数
        guidance_scale       : 视频 CFG 强度（推荐 3.0）
        """
        from diffusers.utils import encode_video

        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        num_frames = self._nearest_valid_frames(self.num_frames)
        reference_image = Image.open(self.reference_image_path).convert("RGB")
        image, padding_info = self._preprocess_image(reference_image)
        print(f"🎬 开始生成（{self.model_name}）：{num_frames} 帧 @ {self.width}×{self.height}, "
              f"steps={self.num_inference_steps}, cfg={self.guidance_scale}")

        with self.torch.inference_mode():
            video, audio = self.pipe(
                image=image,
                prompt=self.prompt,
                negative_prompt=self._NEGATIVE_PROMPT,
                width=self.width,
                height=self.height,
                num_frames=num_frames,
                frame_rate=self.frame_rate,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=self.generator,
                output_type="np",
                return_dict=False,
                callback_on_step_end=self.make_cancel_callback(),
            )

        frames: List[Image.Image] = []
        for f in video[0]:
            arr = f if f.dtype.name == "uint8" else (f * 255).clip(0, 255).astype("uint8")
            frames.append(Image.fromarray(arr))

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

        # 合成带音频的 mp4
        video_path = output_dir / f"{image_id}.mp4"
        encode_video(
            video[0],
            fps=self.frame_rate,
            audio=audio[0].float().cpu() if audio is not None else None,
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate if audio is not None else None,
            output_path=str(video_path),
        )
        print(f"✅ 已保存 {len(frames)} 帧到 {output_dir}")
        print(f"   视频预览（含音频）：{video_path}")

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            gc.collect()
        return frame_paths, video_path

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.reference_image_path = raw.get("reference_image_path", raw.get("reference_image", ""))

        self.width, self.height = self._SAFE_RESOLUTIONS.get(raw.get("resolution", ""), self._SAFE_RESOLUTIONS["low"])
        self.num_frames = raw.get("num_frames", 81)
        self.frame_rate = raw.get("frame_rate", 24.0)
        self.prompt = raw.get("content")

        self.num_inference_steps = raw.get("num_inference_steps", 30)
        self.guidance_scale = raw.get("guidance_scale", 3.0)

        self.seed = raw.get("seed", 42)
        self.generator = self.torch.Generator(device=self.device).manual_seed(self.seed)
