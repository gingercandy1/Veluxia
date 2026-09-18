import math
import random
import uuid
from pathlib import Path
from typing import Optional

from PIL import Image

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseImageGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class QwenImageLightningGenerator(BaseImageGenerator):
    """Qwen-Image-Lightning（Apache-2.0，可商用）。

    结构：unsloth Qwen-Image-2512-FP8 底座（Apache-2.0，~20GB）+ lightx2v Lightning LoRA（8 步）。
    注意：LoRA 按 8 月底座训练，与 2512 底座的兼容性以本机实测为准；
    若翻车，回退底座为 Qwen/Qwen-Image 即可。图生图走官方 Qwen-Image-Edit，为二期链路。
    """
    model_dir = "qwen-image-2512-fp8"
    lora_id = "lightx2v/Qwen-Image-Lightning"
    lora_weight = "Qwen-Image-Lightning-8steps-V1.0.safetensors"
    edit_model_id = "Qwen/Qwen-Image-Edit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pipe_edit = None
        self.base_local = Path(PROJECT_ROOT) / "models" / "image" / self.model_dir
        self.edit_local = Path(PROJECT_ROOT) / "models" / "image" / "qwen-image-edit"
        if not self.model_id:
            self.model_id = "unsloth/Qwen-Image-2512-FP8"

    def _check_model_file(self):
        from huggingface_hub import snapshot_download
        if not self.base_local.exists():
            print("⏬ 正在下载 Qwen-Image-2512-FP8 底座（~20GB，注意磁盘）...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.base_local),
                token=huggingface_token,
            )
            print("✅ Qwen-Image 底座下载完成")
        # Lightning LoRA 较小，加载时由 diffusers 自动从 hub 获取

    @staticmethod
    def _lightning_scheduler():
        from diffusers import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        })

    def _load_model(self):
        if self.pipe is not None:
            return
        if self.pipe_edit is not None:
            # 单驻留：文生图与编辑管道互斥，先腾显存
            del self.pipe_edit
            self.pipe_edit = None

        from diffusers import QwenImagePipeline

        print("🔧 正在加载 Qwen-Image + Lightning LoRA（首次较慢）...")
        dtype = self.torch.bfloat16
        self.pipe = QwenImagePipeline.from_pretrained(
            str(self.base_local),
            scheduler=self._lightning_scheduler(),
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.pipe.load_lora_weights(self.lora_id, weight_name=self.lora_weight)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.torch.cuda.empty_cache()

    def _load_edit(self):
        # 二期链路：官方 Edit 模型做图生图/指令编辑，8G 实测待验证
        if self.pipe_edit is not None:
            return
        if self.pipe is not None:
            # 单驻留：先卸文生图管道，再载编辑管道
            del self.pipe
            self.pipe = None
        from huggingface_hub import snapshot_download
        from diffusers import QwenImageEditPipeline
        if not self.edit_local.exists():
            print("⏬ 正在下载 Qwen-Image-Edit 权重...")
            snapshot_download(
                repo_id=self.edit_model_id,
                local_dir=str(self.edit_local),
                token=huggingface_token,
            )
        self.pipe_edit = QwenImageEditPipeline.from_pretrained(
            str(self.edit_local),
            torch_dtype=self.torch.bfloat16,
            local_files_only=True,
        )
        self.pipe_edit.enable_model_cpu_offload()

    def unload_model(self):
        for attr in ("pipe_edit", "pipe"):
            if getattr(self, attr) is not None:
                delattr(self, attr)
                setattr(self, attr, None)
        self.torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✅ Qwen-Image-Lightning 已卸载，显存已释放")

    async def generate(self) -> Optional[Path | None]:
        self.ensure_model_loaded()

        with self.torch.inference_mode():
            try:
                image = self.pipe(prompt=self.prompt,
                                  width=self.width,
                                  height=self.height,
                                  num_inference_steps=self.num_inference_steps,
                                  true_cfg_scale=self.true_cfg_scale,
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
        self._load_edit()

        ref_img = Image.open(self.ref_image_path).convert("RGB")
        with self.torch.inference_mode():
            try:
                image = self.pipe_edit(image=ref_img,
                                       prompt=self.prompt,
                                       num_inference_steps=self.num_inference_steps,
                                       generator=self.generator).images[0]
                image.save(self.save_path)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
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
        # Qwen-Image 用 true_cfg_scale；兼容通用 guidance_scale 键
        self.true_cfg_scale = raw.get(
            "true_cfg_scale", raw.get("guidance_scale", 1.0))
        self.num_inference_steps = raw.get("num_inference_steps", 8)

        seed_value = raw.get("seed", 0) + random.randint(1, 100000)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))

    def get_output_dir(self, output_dir):
        return Path(output_dir) / f"qwenlight_{str(uuid.uuid4())}.png"
