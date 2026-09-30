"""Qwen-Image-Layered 分层（Apache-2.0，可商用）：把一张图拆成若干张透明图层，从底到顶排列。

8GB 显存 / 32GB 内存下的做法（实机测试 2026-09-30）：
- transformer 用 GGUF Q4_K_M，文本编码器 Qwen2.5-VL 用 bf16，两者都按层分组卸载；
- 瓶颈是内存：编码器约 14GB、transformer 约 17GB，同时驻留会把内存挤满，
  所以和 Qwen-Image-Edit-2509 一样分两段加载，编码完先释放编码器再加载 transformer；
- 文本编码器的权重和 Edit-2509 完全相同（16GB），不再另下一份：models.json 的
  text_encoder 字段指向共用的目录；
- 编码结果存成文件：资源包先把所有条目集中编码、再集中去噪，一批只切换一次模型。

分层用的描述直接用生成整图时的提示词，不让模型自己看图写描述：那要多加载一次 VL 生成，
而且我们本来就知道这张图画的是什么。
"""
import gc
import random
import uuid
from pathlib import Path

from PIL import Image

from src.backend.core.diffusers_utils import (
    apply_offload,
    dequantize_gguf_embeddings,
    gguf_filename,
    load_gguf_transformer,
)
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import ensure_file, ensure_snapshot, get_temp_dir
from src.shared.settings import PROJECT_ROOT

IMAGE_MODEL_ROOT = Path(PROJECT_ROOT) / "models" / "image"
# 管线只支持这两档：输出图层的面积约为 档位²，1344×768 输入在 640 档得到 832×480 的图层
RESOLUTIONS = (640, 1024)
# 负向提示词：只用来开 CFG，内容为空格（官方示例的写法）
_NEGATIVE_PROMPT = " "
# 未在 models.json 调整时的默认值，都是实测过效果的设置
_DEFAULTS = {"num_inference_steps": 20, "true_cfg_scale": 4.0, "resolution": 640}


def _layered_pipeline_class():
    """只传预先编码好的提示词、不加载文本编码器时用的管线。

    diffusers 0.39 的 QwenImageLayeredPipeline 预处理输入图时，无论有没有传 prompt_embeds
    都会读 self.text_encoder.dtype 来转换图片精度；编码器没加载时它是 None，直接报错。
    传 latents 绕开也不行：同一个函数开头按 PIL 图取 image.size[0]，张量会报错。
    所以子类化，只改 text_encoder 这一个属性：编码器不在时给出一个只带 dtype 的占位，
    精度取 transformer 的计算精度，这正是图片该转成的精度。管线的卸载、设备推断等逻辑
    都会跳过不是 nn.Module 的组件，占位不会被当成模型处理。
    """
    from types import SimpleNamespace

    from diffusers import QwenImageLayeredPipeline

    class PromptEmbedsLayeredPipeline(QwenImageLayeredPipeline):
        @property
        def text_encoder(self):
            encoder = self.__dict__.get("_text_encoder")
            if encoder is not None:
                return encoder
            transformer = self.__dict__.get("transformer")
            return SimpleNamespace(dtype=transformer.dtype) if transformer is not None else None

        @text_encoder.setter
        def text_encoder(self, value):
            self.__dict__["_text_encoder"] = value

    return PromptEmbedsLayeredPipeline


class QwenImageLayeredGenerator(BaseImageGenerator):
    """编码器管线（encoder_pipe）和去噪管线（pipe）互斥驻留，按用到的那一段加载。

    只在资料库的场景资源包里用（models.json 标了 library_only）：一次产出多张图层，
    聊天页的图片接口只接收一张图。
    """
    model_dir = "qwen-image-layered"

    _model_attrs = ("encoder_pipe", "pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_pipe = None
        extra = getattr(self, "model_extra", None) or {}
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = extra.get("gguf_repo_id", "")
        self.base_local = IMAGE_MODEL_ROOT / self.model_dir
        self.gguf_local = self.base_local / "gguf"
        encoder = extra.get("text_encoder") or {}
        self.text_encoder_repo_id = encoder.get("repo_id", "")
        self.text_encoder_local = IMAGE_MODEL_ROOT / encoder["dir"] if encoder.get("dir") else None
        # 提速实验要在实机上反复试，所以卸载方式、步数、CFG 都放在 models.json 里调，不用改代码
        self.offload = extra.get("offload") or {}
        self.defaults = {**_DEFAULTS, **(extra.get("defaults") or {})}

    def _check_model_file(self):
        if not self.model_id or not self.gguf_filename or not self.gguf_repo_id:
            raise ValueError(f"{self.model_name} 需要在 models.json 里配置 repo_id、.gguf 文件名和 "
                             "gguf_repo_id：全量 transformer 约 40GB，8GB 显存跑不动")
        if self.text_encoder_local is None or not self.text_encoder_repo_id:
            raise ValueError(f"{self.model_name} 需要在 models.json 里配置 text_encoder"
                             "（共用的文本编码器所在仓库 repo_id 和目录 dir）")
        # 自己的仓库只要骨架：transformer 走 GGUF，文本编码器用共用目录里的，各省 40GB / 16GB
        ensure_snapshot(self.model_id, self.base_local,
                        ignore_patterns=["transformer/*", "text_encoder/*"])
        ensure_file(self.model_id, "transformer/config.json", self.base_local)
        ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)
        # 共用目录按那个模型自己的方式整仓下载（同样跳过 transformer），
        # 否则只下了编码器的目录会被那个模型当成"已下载"，它的其余文件就再也不会补上
        ensure_snapshot(self.text_encoder_repo_id, self.text_encoder_local,
                        ignore_patterns=["transformer/*"])
        if not (self.text_encoder_local / "text_encoder").is_dir():
            raise FileNotFoundError(
                f"共用的文本编码器不存在：{self.text_encoder_local / 'text_encoder'}，"
                f"删掉 {self.text_encoder_local} 后重试会重新下载")

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
        from transformers import Qwen2_5_VLForConditionalGeneration

        self.report_load_stage("loading", detail=f"{self.model_name} 文本编码器")
        print("🔧 正在加载 Qwen2.5-VL 文本编码器（与 Edit-2509 共用）...")
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.text_encoder_local / "text_encoder"),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )
        pipe = _layered_pipeline_class().from_pretrained(
            str(self.base_local),
            transformer=None,
            vae=None,
            text_encoder=text_encoder,
            # processor 只在让模型自己看图写描述时用，我们总是给出描述，不必加载
            processor=None,
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
        from diffusers import QwenImageTransformer2DModel

        self.report_load_stage("loading", detail=f"{self.model_name} transformer")
        print("🔧 正在加载 Qwen-Image-Layered transformer（约 3 分钟）...")
        transformer = load_gguf_transformer(
            QwenImageTransformer2DModel, self.gguf_local / self.gguf_filename, self.base_local)
        fixed = dequantize_gguf_embeddings(transformer)
        if fixed:
            print(f"🔧 已解压 GGUF 量化的 Embedding：{', '.join(fixed)}")
        pipe = _layered_pipeline_class().from_pretrained(
            str(self.base_local),
            transformer=transformer,
            text_encoder=None,
            processor=None,
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )
        apply_offload(pipe, "group", self.device, group=self.offload)
        self.pipe = pipe
        self.report_load_stage("ready", 1.0)

    def save_prompt_embeds(self) -> Path:
        """只编码不去噪：把描述（和 CFG 用的负向提示词）的编码结果存成文件，交给后面的去噪步骤。"""
        self.check_cancelled()
        self._use_encoder()
        device = self.torch.device(self.device)
        data = {"prompt": self.prompt}
        with self.torch.inference_mode():
            embeds, mask = self.encoder_pipe.encode_prompt(prompt=self.prompt, device=device)
            data.update(prompt_embeds=embeds.cpu(), prompt_embeds_mask=_cpu(mask))
            # 不开 CFG 时省掉负向编码：每步少一次前向，编码阶段也少算一遍
            if self.true_cfg_scale > 1:
                self.check_cancelled()
                negative, negative_mask = self.encoder_pipe.encode_prompt(
                    prompt=_NEGATIVE_PROMPT, device=device)
                data.update(negative_prompt_embeds=negative.cpu(),
                            negative_prompt_embeds_mask=_cpu(negative_mask))
        path = self.save_path.with_suffix(".pt")
        self.torch.save(data, path)
        print(f"✅ 分层编码完成: {path.name}")
        return path

    def _load_embeds(self) -> dict:
        data = self.torch.load(self.prompt_embeds_path, weights_only=True)
        if self.true_cfg_scale > 1 and data.get("negative_prompt_embeds") is None:
            raise ValueError("编码文件里没有负向提示词，但开了 CFG："
                             "改过 true_cfg_scale 的话请重做编码步骤")
        return data

    async def generate(self) -> list[Path]:
        """去噪出 N 张图层，按从底到顶的顺序返回文件路径。"""
        if not self.ref_image_path:
            raise ValueError(f"{self.model_name} 需要一张要分层的图")
        if not self.prompt_embeds_path:
            raise ValueError(f"{self.model_name} 需要编码步骤产出的编码文件")
        data = self._load_embeds()
        self._use_denoiser()
        with Image.open(self.ref_image_path) as image:
            source = image.convert("RGBA")

        device = self.torch.device(self.device)
        negative = {}
        if self.true_cfg_scale > 1:
            negative = {
                "negative_prompt_embeds": data["negative_prompt_embeds"].to(device),
                "negative_prompt_embeds_mask": _to(data["negative_prompt_embeds_mask"], device),
            }
        with self.torch.inference_mode():
            try:
                layers = self.pipe(
                    image=source,
                    # 必须非空：为空时管线会加载编码器自己看图写描述
                    prompt=data["prompt"] or self.prompt,
                    prompt_embeds=data["prompt_embeds"].to(device),
                    prompt_embeds_mask=_to(data["prompt_embeds_mask"], device),
                    layers=self.layers,
                    resolution=self.resolution,
                    num_inference_steps=self.num_inference_steps,
                    true_cfg_scale=self.true_cfg_scale,
                    cfg_normalize=True,
                    generator=self.generator,
                    callback_on_step_end=self.make_cancel_callback(),
                    **negative,
                ).images[0]
            finally:
                self.torch.cuda.empty_cache()
        paths = []
        for index, layer in enumerate(layers, start=1):
            path = self.save_path.with_name(f"{self.save_path.stem}_{index}.png")
            layer.convert("RGBA").save(path)
            paths.append(path)
        print(f"✅ 分层完成: {len(paths)} 层")
        return paths

    async def generate_by_image(self) -> list[Path]:
        return await self.generate()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = Path(self.output_dir) / f"qwenlayered_{uuid.uuid4()!s}.png"
        self.ref_image_path = raw.get("reference_image", "")
        self.prompt = raw.get("content", "")
        # 由编码步骤产出；去噪时必需
        self.prompt_embeds_path = raw.get("prompt_embeds_path", "")
        self.layers = int(raw.get("layers", 4))
        if not 2 <= self.layers <= 8:
            raise ValueError(f"图层数必须在 2 到 8 之间：{self.layers}")
        self.resolution = int(raw.get("resolution", self.defaults["resolution"]))
        if self.resolution not in RESOLUTIONS:
            raise ValueError(f"分层分辨率只能是 {RESOLUTIONS} 之一：{self.resolution}")
        self.num_inference_steps = int(raw.get("num_inference_steps",
                                               self.defaults["num_inference_steps"]))
        self.true_cfg_scale = float(raw.get("true_cfg_scale", self.defaults["true_cfg_scale"]))

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))


def _cpu(tensor):
    return tensor.cpu() if tensor is not None else None


def _to(tensor, device):
    return tensor.to(device) if tensor is not None else None
