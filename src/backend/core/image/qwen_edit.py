"""Qwen-Image-Edit-2509 按指令改图（Apache-2.0，可商用）：转面图等（ADR 0006）。

8GB 显存 / 32GB 内存下的做法（实机测试 2026-09-26）：
- transformer 用 GGUF Q4_K_M，文本编码器 Qwen2.5-VL 用 bf16，两者都按叶子层分组卸载，显存各约 2GB；
- 瓶颈是内存：编码器和 transformer 各占约 17GB，同时驻留会把内存挤满、开始换页。
  所以分两段加载，编码完先释放编码器再加载 transformer，内存峰值只取两者较大的一个；
- 编码结果（参考图 + 指令）可以存成文件：资源包先把所有视角集中编码、再集中去噪，
  一批条目只切换一次；单张出图时缓存在内存里，同一张图同一句指令多出几张也不用切回编码器。
"""
import gc
import hashlib
import random
import uuid
from pathlib import Path

from PIL import Image

from src.backend.core.diffusers_utils import (
    apply_loras,
    apply_offload,
    ensure_loras,
    gguf_filename,
    load_gguf_transformer,
    parse_loras,
)
from src.backend.core.image.qwen_image import lightning_scheduler
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import ensure_file, ensure_snapshot, get_temp_dir
from src.shared.settings import PROJECT_ROOT

# 与管线内部一致：编码器看的参考图缩到约 384×384 的面积
CONDITION_IMAGE_SIZE = 384 * 384
# 内存缓存的编码结果每条约几 MB，超过这个数就清空重来
_EMBEDS_CACHE_LIMIT = 32


class QwenImageEditPlusGenerator(BaseImageGenerator):
    """编码器管线（encoder_pipe）和去噪管线（pipe）互斥驻留，按用到的那一段加载。"""
    model_dir = "qwen-image-edit-2509"

    _model_attrs = ("encoder_pipe", "pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_pipe = None
        if not self.model_id:
            self.model_id = "Qwen/Qwen-Image-Edit-2509"
        extra = getattr(self, "model_extra", None) or {}
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = extra.get("gguf_repo_id", "QuantStack/Qwen-Image-Edit-2509-GGUF")
        self.base_local = Path(PROJECT_ROOT) / "models" / "image" / self.model_dir
        self.gguf_local = self.base_local / "gguf"
        # Lightning 4 步和镜头转换 LoRA 都只给 2509 编辑底座用，和底座放在一起
        self.loras = parse_loras(extra, local_dir=self.base_local / "lora")
        # (参考图摘要, 指令) → (prompt_embeds, prompt_embeds_mask)，都在 CPU 上
        self._embeds_cache: dict[tuple[str, str], tuple] = {}

    def _check_model_file(self):
        if not self.gguf_filename:
            raise ValueError(f"{self.model_name} 需要在 models.json 里配置 .gguf 文件名："
                             "全量 transformer 约 40GB，8GB 显存跑不动")
        # transformer 走 GGUF 单文件，跳过官方 transformer 权重以省磁盘
        ensure_snapshot(self.model_id, self.base_local, ignore_patterns=["transformer/*"])
        ensure_file(self.model_id, "transformer/config.json", self.base_local)
        ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)
        ensure_loras(self.loras)

    def _load_model(self):
        """分两段加载，用到哪段才加载哪段（_use_encoder / _use_denoiser）；
        这里先不加载，免得一上来就载入这次用不到、还要马上释放的那一半。"""

    def _release(self, attr: str):
        if getattr(self, attr) is None:
            return
        setattr(self, attr, None)
        gc.collect()
        self.torch.cuda.empty_cache()

    def _use_encoder(self):
        if self.encoder_pipe is not None:
            return
        self._release("pipe")
        from diffusers import QwenImageEditPlusPipeline

        self.report_load_stage("loading", detail=f"{self.model_name} 文本编码器")
        print("🔧 正在加载 Qwen2.5-VL 文本编码器（约 3 分钟）...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(self.base_local),
            transformer=None,
            vae=None,
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )
        apply_offload(pipe, "group", self.device)
        self.encoder_pipe = pipe
        self.report_load_stage("ready", 1.0)

    def _use_denoiser(self):
        if self.pipe is not None:
            return
        self._release("encoder_pipe")
        from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        self.report_load_stage("loading", detail=f"{self.model_name} transformer")
        print("🔧 正在加载 Qwen-Image-Edit-2509 transformer + LoRA（约 3 分钟）...")
        transformer = load_gguf_transformer(
            QwenImageTransformer2DModel, self.gguf_local / self.gguf_filename, self.base_local)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(self.base_local),
            transformer=transformer,
            text_encoder=None,
            scheduler=lightning_scheduler(),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )
        apply_loras(pipe, self.loras)
        # GGUF Q4 的 transformer 也有十几 GB，整模块搬不进 8GB，只能按层卸载
        apply_offload(pipe, "group", self.device)
        self.pipe = pipe
        self.report_load_stage("ready", 1.0)

    def _prompt_embeds(self, reference: Image.Image, instruction: str) -> tuple:
        key = (hashlib.sha1(reference.tobytes()).hexdigest(), instruction)
        if key in self._embeds_cache:
            return self._embeds_cache[key]
        self.check_cancelled()
        self._use_encoder()
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            calculate_dimensions,
        )

        width, height = calculate_dimensions(CONDITION_IMAGE_SIZE,
                                             reference.width / reference.height)
        condition = self.encoder_pipe.image_processor.resize(reference, height, width)
        with self.torch.inference_mode():
            # 参考图以列表传入，和管线内部拼出的 "Picture 1: ..." 模板一致
            embeds, mask = self.encoder_pipe.encode_prompt(
                prompt=instruction, image=[condition], device=self.torch.device(self.device))
        result = (embeds.cpu(), mask.cpu() if mask is not None else None)
        if len(self._embeds_cache) >= _EMBEDS_CACHE_LIMIT:
            self._embeds_cache.clear()
        self._embeds_cache[key] = result
        return result

    def save_prompt_embeds(self) -> Path:
        """只编码不去噪：把参考图 + 指令的编码结果存成文件，交给后面的去噪步骤。"""
        reference = self._reference()
        embeds, mask = self._prompt_embeds(reference, self.prompt)
        path = self.save_path.with_suffix(".pt")
        self.torch.save({"instruction": self.prompt, "prompt_embeds": embeds,
                         "prompt_embeds_mask": mask}, path)
        print(f"✅ 编码完成: {path.name}")
        return path

    def _load_embeds(self) -> tuple:
        data = self.torch.load(self.prompt_embeds_path, weights_only=True)
        return data["prompt_embeds"], data["prompt_embeds_mask"]

    async def generate(self) -> Path | None:
        if not self.ref_image_path:
            raise ValueError(f"{self.model_name} 是改图模型，需要先附上一张参考图")
        return await self.generate_by_image()

    async def generate_by_image(self) -> Path | None:
        reference = self._reference()
        if self.prompt_embeds_path:
            embeds, mask = self._load_embeds()
        else:
            embeds, mask = self._prompt_embeds(reference, self.prompt)
        self._use_denoiser()

        device = self.torch.device(self.device)
        with self.torch.inference_mode():
            try:
                image = self.pipe(image=[reference],
                                  prompt_embeds=embeds.to(device),
                                  prompt_embeds_mask=mask.to(device) if mask is not None else None,
                                  true_cfg_scale=1.0,
                                  width=self.width,
                                  height=self.height,
                                  num_inference_steps=self.num_inference_steps,
                                  generator=self.generator,
                                  callback_on_step_end=self.make_cancel_callback(),
                                  ).images[0]
            finally:
                self.torch.cuda.empty_cache()
        image.save(self.save_path)
        print(f"✅ 生成完成: {self.save_path.name}")
        return self.save_path

    def _reference(self) -> Image.Image:
        """透明底的立绘直接转 RGB 会变黑底，先铺到白底上。"""
        if not self.ref_image_path:
            raise ValueError(f"{self.model_name} 缺少参考图")
        with Image.open(self.ref_image_path) as image:
            rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = self.get_output_dir(self.output_dir)

        self.ref_image_path = raw.get("reference_image", "")
        self.prompt = raw.get("content", "")
        # 由编码步骤产出；为空时现场编码
        self.prompt_embeds_path = raw.get("prompt_embeds_path", "")
        # 不给宽高时管线按参考图比例取约 1024×1024 的面积
        self.width = raw.get("width")
        self.height = raw.get("height")
        # Lightning 4 步蒸馏
        self.num_inference_steps = raw.get("num_inference_steps", 4)

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"qwenedit_{uuid.uuid4()!s}.png"
