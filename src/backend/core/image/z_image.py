import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.diffusers_utils import apply_offload, gguf_filename, load_gguf_transformer
from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import ensure_file, ensure_snapshot, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class ZImageGenerator(BaseImageGenerator):
    """Z-Image-Turbo 图片生成器（Apache-2.0，可商用；diffusers 原生 ZImagePipeline）

    默认走 GGUF 量化（models.json 指定 gguf_repo_id + filename）：transformer 以单文件
    形式加载，显存占用约等于 GGUF 文件大小，8GB 可用；其余组件（文本编码器/VAE/调度器）
    沿用 repo_id 指向的官方骨架（不下载 transformer 权重）。
    未配置 .gguf 文件名时回退到官方全量权重加载（需大显存/大磁盘）。
    """
    model_dir = "z-image-turbo"

    _model_attrs = ("pipe_img2img", "pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_img2img = None
        self.gguf_filename = gguf_filename(self.model_filename)
        self.gguf_repo_id = (getattr(self, "model_extra", None) or {}).get(
            "gguf_repo_id", "jayn7/Z-Image-Turbo-GGUF")
        if not self.model_id:
            self.model_id = "Tongyi-MAI/Z-Image-Turbo"
        self.gguf_local = Path(PROJECT_ROOT) / "models" / "image" / "z-image-turbo-gguf"
        # GGUF 链路的骨架不含 transformer 权重，和全量权重分目录存，免得误当成完整模型
        base_dir = "z-image-turbo-base" if self.gguf_filename else self.model_dir
        self.base_local = Path(PROJECT_ROOT) / "models" / "image" / base_dir

    def _check_model_file(self):
        if not self.gguf_filename:
            ensure_snapshot(self.model_id, self.base_local)
            return
        # transformer 走 GGUF 单文件，跳过官方 transformer 权重以省磁盘
        ensure_snapshot(self.model_id, self.base_local, ignore_patterns=["transformer/*"])
        # GGUF 加载要读 transformer 结构配置；按上面规则下载的骨架里没有，单独补上
        ensure_file(self.model_id, "transformer/config.json", self.base_local)
        ensure_file(self.gguf_repo_id, self.gguf_filename, self.gguf_local)

    def _fp8_supported(self) -> bool:
        """FP8 动态量化需要 Ada/Hopper/Blackwell 架构（算力 >= 8.9）才能吃到硬件加速，
        否则 torchao 只能在计算前反量化回高精度，白白多一道转换开销、没有提速。"""
        if not self.torch.cuda.is_available():
            return False
        major, minor = self.torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)

    def _build_quantized_transformer(self):
        from diffusers import TorchAoConfig, ZImageTransformer2DModel
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig

        print("⚡ 检测到 FP8 加速支持，正在以动态激活量化加载 transformer...")
        quantization_config = TorchAoConfig(quant_type=Float8DynamicActivationFloat8WeightConfig())
        return ZImageTransformer2DModel.from_pretrained(
            str(self.base_local),
            subfolder="transformer",
            quantization_config=quantization_config,
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )

    def _build_transformer(self) -> tuple[dict, str]:
        """返回 (from_pretrained 的 transformer 参数, offload 模式)，文生图与图生图共用。"""
        from diffusers import ZImageTransformer2DModel

        if self.gguf_filename:
            # GGUF 权重能随整模块搬运，用最快的 model offload
            transformer = load_gguf_transformer(
                ZImageTransformer2DModel, self.gguf_local / self.gguf_filename, self.base_local)
            return {"transformer": transformer}, "model"
        if self._fp8_supported():
            # torchao 量化 tensor 与 model offload 不兼容，只能分组卸载
            return {"transformer": self._build_quantized_transformer()}, "group"
        print("ℹ️ 当前 GPU 不支持 FP8 加速（需 Ada/Hopper 及以上架构），使用 bf16 加载")
        return {}, "model"

    def _load_model(self):
        if self.pipe is not None:
            return
        if self.pipe_img2img is not None:
            # 单驻留：文生图与图生图管道互斥，先腾显存
            del self.pipe_img2img
            self.pipe_img2img = None

        from diffusers import ZImagePipeline

        print("🔧 正在加载 Z-Image-Turbo（首次较慢）...")
        pipe_kwargs, offload = self._build_transformer()
        self.pipe = ZImagePipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **pipe_kwargs,
        )
        apply_offload(self.pipe, offload, self.device)
        self.torch.cuda.empty_cache()

    def _load_img2img(self):
        if self.pipe_img2img is not None:
            return
        from diffusers import ZImageImg2ImgPipeline
        self._check_model_file()
        if self.pipe is not None:
            # 单驻留：先卸文生图管道，再载图生图管道
            del self.pipe
            self.pipe = None

        pipe_kwargs, offload = self._build_transformer()
        self.pipe_img2img = ZImageImg2ImgPipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **pipe_kwargs,
        )
        apply_offload(self.pipe_img2img, offload, self.device)

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()

        with self.torch.inference_mode():
            try:
                image = self.pipe(prompt=self.prompt,
                                  width=self.width,
                                  height=self.height,
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

        ref_img = Image.open(self.ref_image_path).convert("RGB")
        with self.torch.inference_mode():
            try:
                image = self.pipe_img2img(prompt=self.prompt,
                                          image=ref_img,
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
        # Turbo 蒸馏模型官方要求 guidance=0
        self.guidance_scale = raw.get("guidance_scale", 0.0)
        # 官方 8 NFE ≈ 9 steps
        self.num_inference_steps = raw.get("num_inference_steps", 9)
        self.strength = raw.get("strength", 0.6)

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"zimage_{str(uuid.uuid4())}.png"
