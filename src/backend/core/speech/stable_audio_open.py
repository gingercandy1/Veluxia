import uuid
from pathlib import Path

from src.backend.core.exceptions import GenerationCancelled
from src.backend.core.model_base import BaseSpeechGenerator
from src.backend.core.model_utils import (
    get_temp_dir,
    huggingface_token,
    models_root,
    print_vram_usage,
)


def normalize_peak(waveform):
    """峰值超过 1 时整体按峰值缩小：模型输出幅度不受限，直接写 wav 会被截断成爆音
    （爆炸类音效实测有 3% 以上的采样被削）；官方 stable-audio-tools 也是按峰值归一化。"""
    peak = float(abs(waveform).max())
    return waveform / peak if peak > 1.0 else waveform


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

    _DEFAULT_STEPS = 100
    _DEFAULT_NEGATIVE = "Low quality."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_local = models_root() / "speech" / "stable-audio-open-1.0"
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

    async def generate_music(self) -> Path:
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
                    # StableAudioPipeline 只支持旧式 callback(step, timestep, latents)，
                    # 用不了 make_cancel_callback()
                    callback=lambda *_: self.check_cancelled(),
                    callback_steps=1,
                ).audios[0]

                import soundfile as sf
                waveform = normalize_peak(audio.T.float().cpu().numpy())
                sf.write(str(self.save_path), waveform, self.pipe.vae.sampling_rate)
                print(f"✅ 生成完成: {self.save_path.name}")
                return self.save_path
            except GenerationCancelled:
                print("🛑 生成已被用户取消")
                raise
            except Exception as e:
                # 必须抛出：返回 None 时前端只会收到空结果，用户看不到失败原因
                print(f"❌ 生成失败: {e}")
                raise
            finally:
                self.torch.cuda.empty_cache()

    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        self.save_path = Path(self.output_dir) / f"sfx_{uuid.uuid4()}.wav"

        self.prompt = raw.get("content", "")
        # 官方示例用 "Low quality." 做负面提示词，不给的话成品容易带底噪、发闷
        self.negative_prompt = raw.get("negative_prompt", "").strip() or self._DEFAULT_NEGATIVE
        # 47s 是模型上限，音效场景大多数几秒即可，默认给短一点
        self.duration = float(raw.get("duration", 5.0))
        # 这不是蒸馏模型，官方示例用 200 步；8 步去噪不充分，出来的声音和描述对不上
        self.num_inference_steps = int(raw.get("num_inference_steps", self._DEFAULT_STEPS))

        seed_value = raw.get("seed", 0)
        self.generator = self.torch.Generator("cpu").manual_seed(int(seed_value))
