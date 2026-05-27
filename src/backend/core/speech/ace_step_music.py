import gc
import os
from pathlib import Path

from src.backend.core.model_base import BaseSpeechGenerator
from src.backend.core.model_utils import print_vram_usage, get_temp_dir
from src.shared.settings import PROJECT_ROOT

class AceStepMusicGenerator(BaseSpeechGenerator):
    names = { "Ace-Step1.5":{ "tag":"music"} }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ACE-Step 项目路径
        self.ace_step_root = Path(os.path.join(PROJECT_ROOT, "src/core/speech/ACE_Step"))
        self.ace_model_path = Path(os.path.join(self.ace_step_root, "checkpoints"))

        self.model_dir = self.ace_step_root / "checkpoints"
        self.output_dir = Path("output/music")

    def _check_model_file(self):
        pass

    def _load_model(self):
        if self.pipe is not None:
            return
        print(f"🔄 正在加载 ACE-Step 1.5...")

        import sys
        if str(self.ace_step_root) not in sys.path:
            sys.path.insert(0, str(self.ace_step_root))

        try:
            from ACE_Step.acestep.handler import AceStepHandler
            from ACE_Step.acestep.llm_inference import LLMHandler
            from ACE_Step.acestep.inference import GenerationParams, GenerationConfig, generate_music

            self.dit_handler = AceStepHandler()
            self.llm_handler = LLMHandler()

            self.dit_handler.initialize_service(
                project_root=str(self.ace_step_root),
                config_path="acestep-v15-turbo",
                device=self.device,
            )

            self.llm_handler.initialize(
                checkpoint_dir=str(self.ace_model_path),
                lm_model_path="acestep-5Hz-lm-0.6B",
                backend="vllm",
                device=self.device,
            )
            print("✅ ACE-Step 1.5 模型加载完成")
            print_vram_usage()

        except Exception as e:
            print(f"❌ 加载 ACE-Step 失败: {e}")
            print("请确认：")
            print("1. ACE-Step-1.5 项目已正确克隆并 uv sync")
            print("2. self.ace_step_root 路径是否正确")
            raise

    def generate_music(self) -> Path:
        self.ensure_model_loaded()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎵 开始生成音乐 | 时长: {self.duration}s")
        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        # 配置生成参数
        params = GenerationParams(
            caption=self.prompt,
            bpm=128,
            duration=self.duration,
            seed=self.seed,
        )

        # 配置生成设置
        config = GenerationConfig(
            batch_size=self.batch_size,
            audio_format="flac",
        )

        # 生成音乐
        result = generate_music(self.dit_handler,
                                self.llm_handler,
                                params,
                                config,
                                save_dir=str(output_dir))

        if result.success:
            for audio in result.audios:
                print(f"✅ 已生成：{audio['path']}")
                print(f"   Seed：{audio['params'].get('seed', 'N/A')}")
            return Path(result.audios[0]["path"])
        else:
            print(f"❌ 生成失败：{result.error}")
            return Path("")

    def parse_params(self, raw: dict):
        self.prompt = raw.get("content", "")
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))

        self.duration = raw.get("duration", 10)  # 秒
        self.batch_size = raw.get("batch_size", 4)
        self.seed = raw.get("seed", 42)


# ─────────────────────────────────────────────────────────────────────────────
#  奥日风格提示词库（Ori and the Blind Forest / Will of the Wisps）
# ─────────────────────────────────────────────────────────────────────────────
ORI_STYLE_PROMPTS = {

    "forest_exploration": (
        "orchestral cinematic game music, lush forest ambiance, "
        "soft cellos and violas as foundation, soaring violin melody, "
        "gentle harp arpeggios, ethereal female wordless vocals, "
        "light piano runs, warm French horn accents, "
        "natural reverb, peaceful yet mysterious atmosphere, "
        "Ori and the Blind Forest style, high fidelity, no percussion"
    ),

    "chase_tension": (
        "intense orchestral game music, driving string ostinatos, "
        "urgent brass stabs, rapid staccato violins, "
        "pounding tympani and taiko drums, rising tension, "
        "chromatic suspense, cinematic action, "
        "Ori and the Will of the Wisps combat style, "
        "high energy, full orchestra, dynamic range"
    ),

    "emotional_climax": (
        "deeply emotional orchestral music, swelling strings, "
        "solo violin with expressive vibrato, piano melody, "
        "choir humming softly, crescendo building to full orchestra, "
        "bittersweet and hopeful tone, tears-inducing, "
        "Gareth Coker Ori soundtrack style, "
        "cinematic emotional peak, major key resolution"
    ),

    "underwater_ruins": (
        "atmospheric ambient orchestral music, underwater reverb, "
        "slow sustained strings, ethereal synthesizer pads, "
        "distant choral whispers, haunting celesta melody, "
        "mysterious and ancient feeling, sparse texture, "
        "hollow flute motifs, soft marimba, "
        "Ori underwater dungeon style, meditative pace"
    ),

    "dawn_rebirth": (
        "uplifting orchestral game music, sunrise atmosphere, "
        "warm strings slowly building, gentle oboe melody, "
        "harp glissandos, children choir softly singing, "
        "French horns entering triumphantly, "
        "hopeful and radiant tone, major key, "
        "Ori Spirit Tree restoration scene style, "
        "emotional journey from quiet to full orchestra"
    ),

    "boss_battle": (
        "epic orchestral boss battle music, powerful brass fanfare, "
        "aggressive string tremolo, epic choir chanting, "
        "epic tympani and bass drums, intense and heroic, "
        "full symphony orchestra, battle theme, "
        "Ori Shriek boss fight style, "
        "minor key, relentless rhythm, dramatic dynamics"
    ),

    "night_meditation": (
        "gentle ambient orchestral music, quiet nighttime forest, "
        "solo piano with soft string accompaniment, "
        "slow breathing rhythm, peaceful and introspective, "
        "warm cello melody, light triangle accents, "
        "Ori night spirit style, lullaby-like, minimal arrangement"
    ),
}



if __name__ == '__main__':
    generator = AceStepMusicGenerator()
    generator.ensure_model_loaded()
    generator._run_pipeline(
        prompt=ORI_STYLE_PROMPTS["forest_exploration"],
        duration=30.0,
    )
