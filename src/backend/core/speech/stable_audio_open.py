import uuid
from pathlib import Path
from typing import Optional

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseSpeechGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir, print_vram_usage
from src.shared.settings import PROJECT_ROOT


class StableAudioOpenGenerator(BaseSpeechGenerator):
    """Stable Audio Open 1.0 音效生成器（Stability AI Community License，年营收<=100万美元可商用）

    注意：stable-audio-open-small 官方只发布了 stable-audio-tools 的原始 checkpoint
    （model.safetensors + model_config.json），没有 model_index.json / 子目录这套
    diffusers pipeline 结构，StableAudioPipeline.from_pretrained() 加载不了它；
    只有 1.0 是标准 diffusers 仓库布局，所以这里只接 1.0。
    """

    # 只需要 diffusers pipeline 组件；仓库里还带一份 stable-audio-tools 的原始
    # model.ckpt / model.safetensors / vae_model.ckpt，不下这些能省一半体积。
    _DOWNLOAD_PATTERNS = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "text_encoder/*",
        "projection_model/*",
        "transformer/*",
        "vae/*",
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_local = Path(PROJECT_ROOT) / "models" / "speech" / "stable-audio-open-1.0"
        if not self.model_id:
            self.model_id = "stabilityai/stable-audio-open-1.0"

    def _check_model_file(self):
        from huggingface_hub import snapshot_download
        if not (self.base_local / "model_index.json").exists():
            print(f"⏬ 正在下载 {self.model_name}...")
            snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.base_local),
                allow_patterns=self._DOWNLOAD_PATTERNS,
                token=huggingface_token,
            )
            print(f"✅ {self.model_name} 下载完成")

    def _load_model(self):
        if self.pipe is not None:
            return

        from diffusers import StableAudioPipeline

        print(f"🔧 正在加载 {self.model_name}（首次较慢）...")
        self.pipe = StableAudioPipeline.from_pretrained(
            str(self.base_local),
            torch_dtype=self.torch.float16,
            local_files_only=True,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.torch.cuda.empty_cache()
        print(f"✅ {self.model_name} 加载完成")
        print_vram_usage()

    async def generate_music(self) -> Optional[Path]:
        self.ensure_model_loaded()

        with self.torch.inference_mode():
            try:
                audio = self.pipe(
                    prompt=self.prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=self.num_inference_steps,
                    audio_end_in_s=self.duration,
                    num_waveforms_per_prompt=1,
                    generator=self.generator,
                ).audios[0]

                import soundfile as sf
                waveform = audio.T.float().cpu().numpy()
                sf.write(str(self.save_path), waveform, self.pipe.vae.sampling_rate)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
            except GenerationCancelled:
                print("🛑 生成已被用户取消")
                raise
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                return None
            finally:
                self.torch.cuda.empty_cache()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = Path(self.output_dir) / f"sfx_{uuid.uuid4()}.wav"

        self.prompt = raw.get("content", "")
        self.negative_prompt = raw.get("negative_prompt", "").strip() or None
        # 47s 是模型上限，音效场景大多数几秒即可，默认给短一点
        self.duration = raw.get("duration", 5.0)
        self.num_inference_steps = raw.get("num_inference_steps", 8)

        seed_value = raw.get("seed", 0)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))
