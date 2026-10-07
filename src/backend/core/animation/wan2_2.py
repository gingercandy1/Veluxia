import gc
import uuid
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageFilter
from PIL.Image import Resampling
from src.backend.core.diffusers_utils import (
    apply_loras,
    apply_offload,
    ensure_loras,
    gguf_filename,
    load_gguf_transformer,
    parse_loras,
)
from src.backend.core.model_base import BaseAnimationGenerator
from src.backend.core.model_utils import (
    ensure_file,
    ensure_snapshot,
    get_temp_dir,
    models_root,
    print_vram_usage,
)

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
        extra = getattr(self, "model_extra", None) or {}
        # 本地存储目录：不同底座（5B / Wan2.1 14B）各用各的，"dir" 缺省沿用 5B 的 wan2
        local_name = extra.get("dir", "wan2")
        self.model_dir  = models_root() / "animation" / local_name
        # models.json 给了 .gguf 文件名才走量化链路：官方 transformer 是 fp32（约 19GB），
        # 加载时连同文本编码器一起会撑爆 32GB 内存，GGUF 在内存和显存里都保持压缩
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = extra.get("gguf_repo_id", "")
        self.gguf_local = models_root() / "animation" / f"{local_name}-gguf"
        # 14B 的量化 transformer 仍超过 8GB，整模块搬运放不下，要在 models.json 里写 "offload" 改按层搬运
        self.offload = extra.get("offload") or {}
        # 步数蒸馏 LoRA（如 lightx2v 的 4 步版）：14B 每步要 90 秒以上，不蒸馏跑一次近一小时
        self.loras = parse_loras(extra)
        # Wan2.1 的 VAE 8 倍下采样 × patch 2，宽高 16 的倍数即可；5B 是 32
        self.size_multiple = int(extra.get("size_multiple", 32))
        if not self.model_id:
            self.model_id = self._BASE_REPO_ID

    def _check_model_file(self):
        if self.gguf_filename:
            ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)
            # transformer 来自 GGUF，只要骨架里的 config；14B 官方权重是 fp32 分片（约 65GB），不下
            ensure_snapshot(self.model_id, self.model_dir,
                            ignore_patterns=["transformer/*.safetensors",
                                             "transformer/*.safetensors.index.json"])
        else:
            ensure_snapshot(self.model_id, self.model_dir)
        ensure_loras(self.loras)

    def _load_model(self):
        print(f"🔄 正在加载 {self.model_name}...")
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
        apply_loras(self.pipe, self.loras)
        if self.offload:
            apply_offload(self.pipe, "group", self.device, group=self.offload)
        else:
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

    def _first_last_kwargs(self, image: Image.Image) -> dict:
        """2.1 I2V 的循环：末帧也设成参考图，让动作从站姿出发、回到同一站姿。

        末帧只走管线的 last_image（潜变量条件 + mask）；CLIP 特征仍只用首帧：
        480P 的 I2V 权重按单张图训练图像交叉注意力，管线默认给的双图特征（514 个 token）
        偏出训练分布。管线不允许同时传 image 和 image_embeds，所以只能临时包一层
        encode_image，用完由调用方 del 还原。
        """
        original = self.pipe.encode_image

        def first_only(images, *args, **kwargs):
            if isinstance(images, (list, tuple)):
                images = images[0]
            return original(images, *args, **kwargs)

        self.pipe.encode_image = first_only
        return {"last_image": image}

    def _add_headroom(self, image: Image.Image) -> Image.Image:
        """把主体缩小并靠下放进同尺寸画布，头顶留出空白。

        主体占满画面时模型不敢让它出框，起跳只会变成转头、拉长（实测脚几乎不离地）；
        留白后同样的提示词才有向上的空间。底色取左上角像素，贴边羽化，避免方框接缝，
        所以只适合纯色 / 近纯色背景。
        """
        width, height = image.size
        scale = 1 - self.headroom
        small = image.resize((round(width * scale), round(height * scale)), Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), image.getpixel((0, 0)))
        feather = max(min(small.size) // 40, 2)
        mask = Image.new("L", small.size, 0)
        mask.paste(255, (feather, feather, small.width - feather, small.height - feather))
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        # 水平居中、脚离底边留 8%，上方的空白留给起跳
        canvas.paste(small, ((width - small.width) // 2,
                             height - small.height - round(height * 0.08)), mask)
        return canvas

    def _anchor_last_frame(self):
        """把最后一个潜帧也钉成参考图，首尾两头都有锚点。

        TI2V-5B 只用第一帧做条件，后面的帧没有约束，离第一帧越远形状变得越厉害
        （实测图标特效到第 30 帧边框溶解、主体断裂）。它的第一帧是靠 mask 注入的：
        mask 为 0 的潜帧直接取参考图的潜变量、时间步记为 0；把末帧的 mask 也置 0，
        中间帧就只能在两个相同的锚点之间变化，形状保住了，首尾也自然接得上。
        diffusers 在这个模式下会忽略 last_image 参数，所以只能在 prepare_latents 的结果上改。
        """
        if not self.pipe.config.expand_timesteps:
            raise ValueError(f"{self.model_name} 不支持循环模式：只有 Wan2.2-TI2V-5B 能锚定末帧")
        original = self.pipe.prepare_latents

        def anchored(*args, **kwargs):
            latents, condition, mask = original(*args, **kwargs)
            mask = mask.clone()
            mask[:, :, -1] = 0
            return latents, condition, mask

        # 实例属性遮住类上的方法，用完 del 掉就还原
        self.pipe.prepare_latents = anchored

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
        if self.headroom:
            reference_image = self._add_headroom(reference_image)
        image, padding_info = self._preprocess_image(reference_image)
        print(f"🎬 开始生成：{num_frames} 帧 @ {self.width}×{self.height}, "
              f"steps={self.num_inference_steps}, cfg={self.guidance_scale}")

        # 5B（expand_timesteps）靠改 mask 锚定末帧；2.1 I2V 走管线自带的 last_image
        anchored = self.loop and self.pipe.config.expand_timesteps
        first_last = self.loop and not anchored
        pipe_kwargs = {}
        if anchored:
            self._anchor_last_frame()
        try:
            with self.torch.inference_mode():
                if first_last:
                    pipe_kwargs = self._first_last_kwargs(image)
                output = self.pipe(
                    **pipe_kwargs,
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
        finally:
            # 生成器是单例，不还原的话之后不循环的生成也会被钉住末帧
            if anchored:
                del self.pipe.prepare_latents
            if first_last:
                del self.pipe.encode_image

        frames: List[Image.Image] = []
        for f in output.frames[0]:
            if isinstance(f, Image.Image):
                frames.append(f)
            else:
                if f.dtype != np.uint8:
                    f = (f * 255).clip(0, 255).astype(np.uint8)
                frames.append(Image.fromarray(f))
        if self.loop:
            # 末尾这个潜帧是按"单张图"编码的，放在因果 VAE 的序列末尾解码出来的几帧带噪：
            # 5B 发灰，2.1 14B 实测过曝（最后 4 帧近饱和像素从约 1% 跳到 20% 以上，正好是最后一个潜帧）。
            # 丢掉它们，循环播放时再由交叉淡化抹平接缝
            frames = frames[:-self.pipe.vae_scale_factor_temporal]

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
            if self.width % self.size_multiple or self.height % self.size_multiple:
                raise ValueError(f"{self.model_name} 的宽高必须是 {self.size_multiple} 的倍数："
                                 f"{self.width}×{self.height}")
        self.num_frames = raw.get("num_frames", 25)
        self.prompt = raw.get("content")
        # 循环动画：末帧也设成参考图。5B 见 _anchor_last_frame，2.1 14B 见 _first_last_kwargs
        self.loop = bool(raw.get("loop", False))
        # 主体缩小的比例（0 = 不留白），给起跳 / 伸展这类出框的动作留空间
        self.headroom = float(raw.get("headroom", 0))
        if not 0 <= self.headroom < 0.7:
            raise ValueError(f"headroom 必须在 0 到 0.7 之间：{self.headroom}")

        self.num_inference_steps = raw.get("num_inference_steps", 25)
        self.guidance_scale = raw.get("guidance_scale", 5.0)

        self.seed = raw.get("seed", 42)
        self.generator = self.torch.Generator(device=self.device).manual_seed(self.seed)


