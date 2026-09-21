import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class ZImageGenerator(BaseImageGenerator):
    """Z-Image-Turbo 图片生成器（Apache-2.0，可商用；diffusers 原生 ZImagePipeline）

    默认走 GGUF 量化（models.json 指定 filename）：transformer 以单文件形式加载，
    显存占用约等于 GGUF 文件大小，8GB 可用；其余组件（文本编码器/VAE/调度器）
    沿用官方 `Tongyi-MAI/Z-Image-Turbo` 骨架（base_id，不含 transformer 权重）。
    未配置 filename 时回退到官方全量权重加载（需大显存/大磁盘）。
    """
    model_dir = "z-image-turbo"

    _model_attrs = ("pipe_img2img", "pipe")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_img2img = None
        # GGUF 判定：models.json 给了 .gguf 文件名才走量化链路
        self.gguf_filename = self.model_filename if str(self.model_filename or "").endswith(".gguf") else None
        # 官方管线骨架（文本编码器/VAE/调度器等，不含 transformer 权重）
        self.base_model_id = (getattr(self, "model_extra", None) or {}).get("base_id", "Tongyi-MAI/Z-Image-Turbo")
        if self.gguf_filename:
            if not self.model_id:
                self.model_id = "jayn7/Z-Image-Turbo-GGUF"
            self.gguf_local = Path(PROJECT_ROOT) / "models" / "image" / "z-image-turbo-gguf"
            self.base_local = Path(PROJECT_ROOT) / "models" / "image" / "z-image-turbo-base"
        else:
            self.base_local = Path(PROJECT_ROOT) / "models" / "image" / self.model_dir
            if not self.model_id:
                self.model_id = "Tongyi-MAI/Z-Image-Turbo"

    def _check_model_file(self):
        if not self.gguf_filename:
            from huggingface_hub import snapshot_download
            if not self.base_local.exists():
                print("⏬ 正在下载 Z-Image-Turbo 权重...")
                snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(self.base_local),
                    token=huggingface_token,
                )
                print("✅ Z-Image-Turbo 下载完成")
            return
        from huggingface_hub import hf_hub_download, snapshot_download
        if not self.base_local.exists():
            print("⏬ 正在下载 Z-Image-Turbo 官方骨架（不含 transformer，约数 GB）...")
            snapshot_download(
                repo_id=self.base_model_id,
                local_dir=str(self.base_local),
                # transformer 走 GGUF 单文件，这里跳过官方 transformer 权重以省磁盘
                ignore_patterns=["transformer/*"],
                token=huggingface_token,
            )
            print("✅ Z-Image-Turbo 骨架下载完成")
        gguf_path = self.gguf_local / self.gguf_filename
        if not gguf_path.exists():
            print(f"⏬ 正在下载 GGUF 量化权重 {self.gguf_filename}（约 5GB）...")
            self.gguf_local.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=self.model_id,
                filename=self.gguf_filename,
                local_dir=str(self.gguf_local),
                token=huggingface_token,
            )
            print("✅ GGUF 权重下载完成")

    def _fp8_supported(self) -> bool:
        """FP8 动态量化需要 Ada/Hopper/Blackwell 架构（算力 >= 8.9）才能吃到硬件加速，
        否则 torchao 只能在计算前反量化回高精度，白白多一道转换开销、没有提速。"""
        if not self.torch.cuda.is_available():
            return False
        major, minor = self.torch.cuda.get_device_capability()
        return (major, minor) >= (8, 9)

    def _build_gguf_transformer(self):
        """从 GGUF 单文件构建量化 transformer（8GB 显存的关键：权重常驻约文件大小）。"""
        from diffusers import GGUFQuantizationConfig, ZImageTransformer2DModel

        print(f"⚡ 正在以 GGUF 量化加载 transformer（{self.gguf_filename}）...")
        quantization_config = GGUFQuantizationConfig(compute_dtype=self.torch.bfloat16)
        return ZImageTransformer2DModel.from_single_file(
            str(self.gguf_local / self.gguf_filename),
            quantization_config=quantization_config,
            dtype=self.torch.bfloat16,
        )

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

    def _apply_offload(self, pipe, quantized: bool):
        """
        显存卸载策略。
        `enable_model_cpu_offload()` 内部用 accelerate 的存储指针互换来搬运整模块，
        这套机制和 torchao 量化后的 tensor 子类（Float8Tensor 等）不兼容，会报
        "storage of a tensor on device cuda:0 to a storage on different device cpu"。
        量化场景改用 diffusers 的分组卸载（按参数搬运，非指针互换），专为量化权重设计。
        """
        if quantized:
            pipe.enable_group_offload(
                onload_device=self.torch.device(self.device),
                offload_device=self.torch.device("cpu"),
                offload_type="leaf_level",
            )
        else:
            pipe.enable_model_cpu_offload()

    def _load_model(self):
        if self.pipe is not None:
            return
        if self.pipe_img2img is not None:
            # 单驻留：文生图与图生图管道互斥，先腾显存
            del self.pipe_img2img
            self.pipe_img2img = None

        from diffusers import ZImagePipeline

        print("🔧 正在加载 Z-Image-Turbo（首次较慢）...")
        dtype = self.torch.bfloat16
        if self.gguf_filename:
            # GGUF 链路：量化 transformer + 官方骨架，常驻显存约 GGUF 文件大小
            self.pipe = ZImagePipeline.from_pretrained(
                str(self.base_local),
                transformer=self._build_gguf_transformer(),
                torch_dtype=dtype,
                local_files_only=True,
                low_cpu_mem_usage=False,
            )
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            self.torch.cuda.empty_cache()
            return
        quantized = self._fp8_supported()
        pipe_kwargs = {}
        if quantized:
            pipe_kwargs["transformer"] = self._build_quantized_transformer()
        else:
            print("ℹ️ 当前 GPU 不支持 FP8 加速（需 Ada/Hopper 及以上架构），使用 bf16 加载")

        self.pipe = ZImagePipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=dtype,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **pipe_kwargs,
        )
        self._apply_offload(self.pipe, quantized)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
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

        quantized = self._fp8_supported() if not self.gguf_filename else False
        pipe_kwargs = {}
        if self.gguf_filename:
            pipe_kwargs["transformer"] = self._build_gguf_transformer()
        elif quantized:
            pipe_kwargs["transformer"] = self._build_quantized_transformer()

        self.pipe_img2img = ZImageImg2ImgPipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=False,
            **pipe_kwargs,
        )
        if self.gguf_filename:
            self.pipe_img2img.enable_model_cpu_offload()
        else:
            self._apply_offload(self.pipe_img2img, quantized)

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
