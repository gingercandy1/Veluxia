"""Qwen-Image-2.1 文生图 / 图生图（官方 diffusers 管线 QwenImage21Pipeline）。

同一个管线：不给参考图就是文生图，给了 reference_image 就以它为条件改图（图层文字修改用这个）。

模型：约 7B 的单流 DiT + Qwen3-VL 文本编码器，原生支持 RGBA 透明输出。许可证是
Qwen Research License（仅限研究 / 评测，商用要另行授权），与本项目其他 Apache-2.0 的模型不同。
Turbo 版本 8 步、CFG 为 1，采样日程存在检查点里，所以不传 num_inference_steps。

8GB 显存 / 32GB 内存下的做法（与 Qwen-Image-Layered 一致）：
- 编码器（约 16GB）和 transformer（约 14GB）同时驻留会把内存挤满，所以分两段加载：
  先用编码器把提示词编码好，释放后再加载 transformer + VAE 去噪；
- 两段都按层分组卸载，卸载方式放在 models.json 的 offload 里，方便在实机上调速度。
"""
import functools
import gc
import hashlib
import random
import uuid
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from src.backend.core.diffusers_utils import (
    apply_offload,
    dequantize_gguf_embeddings,
    gguf_filename,
    load_gguf_transformer,
)
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import ensure_file, ensure_snapshot, get_temp_dir, models_root

# 宽高范围：边长要是 32 的倍数；上限按 8GB 显存留余量（官方示例用到 2048）
MIN_SIDE = 256
MAX_SIDE = 2048
SIDE_STEP = 32
# 编辑（有参考图）的默认 output_resolution：1024 会让指令被忽略（diffusers #14824），512/768 正常
EDIT_OUTPUT_RESOLUTION = 768
# 去噪阶段的卸载方式：transformer 按块分组，VAE 没有顶层块列表，单独按叶子层。
# 组越大越快、显存越多；这是起点，没在实机上测过，按需在 models.json 的 offload 里改
_DEFAULT_OFFLOAD = {
    "type": "block_level",
    "blocks_per_group": 8,
    "use_stream": False,
    "leaf_components": ["vae"],
}


def _image_key(image: Image.Image | None) -> str:
    """参考图的内容摘要，用来判断编码缓存能不能复用；没有参考图时为空串。"""
    return hashlib.sha1(image.tobytes()).hexdigest() if image is not None else ""


class QwenImage21Generator(BaseImageGenerator):
    """编码器管线（encoder_pipe）和去噪管线（pipe）互斥驻留，按用到的那一段加载。"""

    # 最近一次提示词的编码结果也算"持有的模型状态"：卸载时一并清掉
    _model_attrs = ("encoder_pipe", "pipe", "_embeds_cache")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_pipe = None
        # (提示词 + 参考图摘要, embeds, mask)：同一句提示词、同一张图连续出图时跳过编码器，
        # 省掉一次 16GB 的加载；参考图变了编码结果也要变，所以参考图要进缓存键
        self._embeds_cache = None
        extra = getattr(self, "model_extra", None) or {}
        self.offload = {**_DEFAULT_OFFLOAD, **(extra.get("offload") or {})}
        # models.json 的 filename 是 .gguf 时 transformer 走量化单文件（约 4GB，对比全量 14GB），
        # 其余（文本编码器、VAE、配置）仍取官方仓库；没配就用官方全量权重
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = extra.get("gguf_repo_id", "")
        # 本地目录按仓库名取，量化版另用一个目录：它的骨架不含 transformer 权重，
        # 和整仓下载的目录混用会让后者被当成"已下载"
        repo_name = (self.model_id or "").split("/")[-1].lower()
        suffix = "-gguf" if self.gguf_filename else ""
        self.base_local = models_root() / "image" / f"{repo_name}{suffix}"
        self.gguf_local = self.base_local / "gguf"

    def _check_model_file(self):
        if not self.model_id:
            raise ValueError(f"{self.model_name} 需要在 models.json 里配置 repo_id")
        if not self.gguf_filename:
            ensure_snapshot(self.model_id, self.base_local)
            return
        if not self.gguf_repo_id:
            raise ValueError(f"{self.model_name} 配了 .gguf 文件名，还需要 gguf_repo_id（GGUF 所在仓库）")
        # 官方仓库只要骨架（文本编码器、VAE、配置）；transformer 用 GGUF，省下约 10GB
        # 以 model_index.json 为准：config.json / GGUF 可能先于骨架落盘，只看目录会跳过骨架下载
        ensure_snapshot(self.model_id, self.base_local, ignore_patterns=["transformer/*"],
                        marker="model_index.json")
        ensure_file(self.model_id, "transformer/config.json", self.base_local)
        ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)

    def _load_model(self):
        """分两段加载，用到哪段才加载哪段；这里先不加载，免得一上来就载入马上要释放的那一半。"""

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
        from diffusers import QwenImage21Pipeline

        self.report_load_stage("loading", detail=f"{self.model_name} 文本编码器")
        print("🔧 正在加载 Qwen3-VL 文本编码器...")
        # transformer / VAE 不加载：管线构造时只用 processor 和 VAE 配置（VAE 缺省时取默认通道数）
        pipe = QwenImage21Pipeline.from_pretrained(
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
        from diffusers import QwenImage21Pipeline

        self.report_load_stage("loading", detail=f"{self.model_name} transformer")
        print("🔧 正在加载 Qwen-Image-2.1 transformer...")
        kwargs = {}
        if self.gguf_filename:
            from diffusers import QwenImage21Transformer2DModel

            transformer = load_gguf_transformer(
                QwenImage21Transformer2DModel, self.gguf_local / self.gguf_filename,
                self.base_local)
            fixed = dequantize_gguf_embeddings(transformer)
            if fixed:
                print(f"🔧 已解压 GGUF 量化的 Embedding：{', '.join(fixed)}")
            kwargs["transformer"] = transformer
        pipe = QwenImage21Pipeline.from_pretrained(
            str(self.base_local),
            text_encoder=None,
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
            **kwargs,
        )
        apply_offload(pipe, "group", self.device, group=self.offload)
        self.pipe = pipe
        self.report_load_stage("ready", 1.0)

    def _encode_prompt(self, reference: Image.Image | None):
        """返回 (embeds, mask, image_pad_mask)，都在 CPU 上；同一句提示词、同一张参考图直接用上次的。

        image_pad_mask 标出嵌入里哪些位置是参考图的 token：带参考图去噪时管线要用它。"""
        key = (self.prompt, _image_key(reference))
        cached = self._embeds_cache
        if cached is not None and cached[0] == key:
            return cached[1], cached[2], cached[3]
        self.check_cancelled()
        self._use_encoder()
        # 文本编码器也要看到参考图：编辑指令是"以这张图为基础"的描述，不带图编出来的嵌入不对
        # 没有参考图时不传 image，文生图的调用保持原样
        condition = {"image": [reference]} if reference is not None else {}
        with self.torch.inference_mode():
            embeds, mask, pad_mask = self.encoder_pipe.encode_prompt(
                prompt=self.prompt, device=self.torch.device(self.device), **condition)
        embeds = embeds.cpu()
        mask = mask.cpu() if mask is not None else None
        pad_mask = pad_mask.cpu() if pad_mask is not None else None
        self._embeds_cache = (key, embeds, mask, pad_mask)
        return embeds, mask, pad_mask

    @contextmanager
    def _with_image_pad_mask(self, pad_mask, device):
        """带参考图时把 image_pad_mask 递给去噪管线。

        管线 __call__ 拿到 prompt_embeds 后还会再走一遍 encode_prompt：给了参考图却没有
        image_pad_mask 就报错，而 __call__ 本身没有这个参数（diffusers 0.41.0）。两段加载时
        编码器已经释放，不能让管线自己重编，所以在这次调用期间给它的 encode_prompt 绑上这个参数。"""
        if pad_mask is None:
            yield
            return
        self.pipe.encode_prompt = functools.partial(
            self.pipe.encode_prompt, image_pad_mask=pad_mask.to(device))
        try:
            yield
        finally:
            # 删掉实例上的覆盖，恢复类方法，下一次调用不会带着旧参考图的掩码
            del self.pipe.encode_prompt

    def _reference(self) -> Image.Image | None:
        """参考图统一转成 RGB：透明底直接转会变黑底，先铺到白底上（同 qwen_edit 的做法）。"""
        if not self.ref_image_path:
            return None
        with Image.open(self.ref_image_path) as image:
            rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")

    def _resolution_kwargs(self, reference: Image.Image | None) -> dict:
        """编辑模式才传 output_resolution。

        管线里它决定参考图被缩到多大（面积 = output_resolution²）；显式给了 width/height 时
        输出尺寸以后者为准，但参考图仍按它缩放。diffusers issue #14824：1024 下编辑会忽略指令、
        输出近似原图，所以编辑默认 768。文生图没有参考图，保持管线默认。"""
        if reference is None:
            return {}
        return {"output_resolution": self.output_resolution}

    async def generate(self) -> Path | None:
        self.ensure_model_loaded()
        reference = self._reference()
        embeds, mask, pad_mask = self._encode_prompt(reference)
        self.check_cancelled()
        self._use_denoiser()

        device = self.torch.device(self.device)
        condition = {"image": [reference]} if reference is not None else {}
        # 文生图不碰 image_pad_mask：管线对纯文字嵌入会自己补全零掩码
        pad_mask = pad_mask if reference is not None else None
        with self.torch.inference_mode(), self._with_image_pad_mask(pad_mask, device):
            try:
                image = self.pipe(
                    **condition,
                    **self._resolution_kwargs(reference),
                    prompt_embeds=embeds.to(device),
                    prompt_embeds_mask=mask.to(device) if mask is not None else None,
                    width=self.width,
                    height=self.height,
                    generator=self.generator,
                    callback_on_step_end=self.make_cancel_callback(),
                ).images[0]
            finally:
                self.torch.cuda.empty_cache()
        # 带透明通道时保持 RGBA：PNG 会连 alpha 一起存下
        image.save(self.save_path)
        print(f"✅ 生成完成: {self.save_path.name}（{image.mode}）")
        return self.save_path

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = Path(self.output_dir) / f"qwenimage21_{uuid.uuid4()!s}.png"
        self.prompt = (raw.get("content") or "").strip()
        if not self.prompt:
            raise ValueError(f"{self.model_name} 需要提示词")
        # 给了参考图就是图生图，没给就是文生图
        self.ref_image_path = raw.get("reference_image", "") or ""
        # 有参考图且调用方没给时用 EDIT_OUTPUT_RESOLUTION；文生图不用这个值
        self.output_resolution = int(raw.get("output_resolution") or EDIT_OUTPUT_RESOLUTION)
        if not MIN_SIDE <= self.output_resolution <= MAX_SIDE:
            raise ValueError(
                f"output_resolution 必须在 {MIN_SIDE}～{MAX_SIDE} 之间：{self.output_resolution}")
        self.width = int(raw.get("width", 1024))
        self.height = int(raw.get("height", 1024))
        for name, side in (("宽", self.width), ("高", self.height)):
            if not MIN_SIDE <= side <= MAX_SIDE or side % SIDE_STEP:
                raise ValueError(
                    f"{name}必须是 {SIDE_STEP} 的倍数且在 {MIN_SIDE}～{MAX_SIDE} 之间：{side}")
        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))
