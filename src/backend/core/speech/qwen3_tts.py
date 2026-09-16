import os
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from src.backend.core.model_base import BaseSpeechGenerator
from src.backend.core.model_utils import huggingface_token, get_temp_dir
from src.shared.settings import PROJECT_ROOT


class Qwen3TTSGenerator(BaseSpeechGenerator):
    """
    基于 Qwen3-TTS 的文字转语音生成器。
    """
    dynamic = True
    # 内置预设音色
    PRESET_VOICES = ["Chelsie", "Ethan", "Serena", "Dylan", "Ana", "Vivian", "Ryan", "Aria", "Marco"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = self.model_extra.get("mode", None)
        self.model_dir = os.path.join(PROJECT_ROOT, "models", "speech", "qwen3-tts", self.model_name)

    def _check_model_file(self):
        if not os.path.exists(self.model_dir):
            return

        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=self.model_id,
            local_dir=str(self.model_dir),
            token=huggingface_token,
        )
        print("下载完成")

    def _load_model(self):
        print(f"🔧 正在加载 {self.model_id}...")
        from qwen_tts import Qwen3TTSModel
        self.pipe = Qwen3TTSModel.from_pretrained(
            self.model_dir,
            device_map=self.device,
            dtype=self.torch.bfloat16,
        )
        print(f"✅ Qwen3-TTS ({self.model_name}) 加载完成")


    async def generate_music(
            self,
    ) -> Path:
        """
        执行 TTS 推理，返回 wav 文件路径。
        mode 说明：
            "custom"  → 使用内置预设音色（需加载 *-CustomVoice 模型）
            "design"  → 用文字描述生成音色（需加载 *-VoiceDesign 模型）
            "clone"   → 上传参考音频克隆（需加载 *-Base 模型）
        """

        self.ensure_model_loaded()

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"tts_{uuid.uuid4().hex[:8]}.wav"
        print(f"🎙️ 正在合成语音（模式={self.mode}，{len(self.text)} 字）...")

        if self.mode == "design":
            if not self.voice_prompt:
                raise ValueError("VoiceDesign 模式必须提供 voice_prompt（音色描述）")
            audio_array, sample_rate = self.pipe.generate_voice_design(
                text=self.text,
                instruct=self.voice_prompt,
                language=self.language,
            )

        elif self.mode == "custom":
            kwargs = dict(text=self.text, speaker=self.voice, language=self.language)
            if self.instruct:
                kwargs["instruct"] = self.instruct
            audio_array, sample_rate = self.pipe.generate_custom_voice(**kwargs)

        elif self.mode == "clone":
            if not self.reference_audio_path or not self.reference_text:
                raise ValueError("clone 模式必须同时提供 reference_audio_path 和 reference_text")

            voice_clone_prompt = self.pipe.create_voice_clone_prompt(
                ref_audio=self.reference_audio_path,
                ref_text=self.reference_text,
            )
            audio_array, sample_rate = self.pipe.generate_voice_clone(
                text=self.text,
                voice_clone_prompt=voice_clone_prompt,
            )

        else:
            raise ValueError(f"不支持的模式: {self.mode}，请使用 'custom' / 'design' / 'clone'")

        # 安全保存音频（推荐写法）
        if isinstance(audio_array, self.torch.Tensor):
            audio_array = audio_array.cpu().numpy()

        audio_array = np.asarray(audio_array)
        if audio_array.ndim == 2:
            if audio_array.shape[0] == 1:
                audio_array = np.ravel(audio_array)
            else:
                audio_array = audio_array[0]

        sf.write(str(save_path), audio_array, sample_rate)
        print(f"✅ 语音合成完成: {save_path.name}")
        return save_path


    def parse_params(self, raw: dict):
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))

        # ── 必填 ──────────────────────────────────────────────────────────
        self.text = raw.get("content", "")
        self.language = raw.get("language", "Chinese")

        # ── VoiceDesign 参数 ──────────────────────────────────────────────
        self.voice_prompt = raw.get("voice_prompt", "")

        # ── CustomVoice 参数 ──────────────────────────────────────────────
        self.voice = raw.get("voice", "Vivian")
        self.instruct = raw.get("instruct", None)

        # ── VoiceClone 参数 ───────────────────────────────────────────────
        self.reference_audio_path = raw.get("reference_audio_path", "")
        self.reference_text = raw.get("reference_text", "")


if __name__ == '__main__':
    generator = Qwen3TTSGenerator()
    generator.ensure_model_loaded()
    generator._run_tts(
        voice_prompt="一个20岁左右的热情有活力的女孩的声音",
        text="你好，今天下午一起吃下午茶吧",
        output_dir = os.path.join(PROJECT_ROOT, "output_music")
    )
