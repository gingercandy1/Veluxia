import asyncio
import sys
from pathlib import Path

from src.backend.core.model_base import BaseSpeechGenerator
from src.backend.core.model_utils import get_temp_dir, huggingface_token, print_vram_usage
from src.shared.settings import PROJECT_ROOT

# 第三方源码目录（gitignore，单独克隆 + 安装依赖，见 backend/pyproject.toml）
ACE_STEP_ROOT = Path(PROJECT_ROOT) / "src" / "backend" / "core" / "speech" / "ACE_Step"


class AceStepMusicGenerator(BaseSpeechGenerator):
    """ACE-Step 1.5 音乐生成：5Hz LM 先规划曲式与音频码，DiT 再合成音频。

    ACE-Step 内部以 `acestep.*` 互相导入，只能把 ACE_Step 根目录挂进 sys.path，
    不能按 `ACE_Step.acestep` 包路径导入。
    """
    # pipe 指向 DiT handler（ensure_model_loaded 靠它判断是否已加载），LM 单独持有
    _model_attrs = ("pipe", "llm_handler")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.llm_handler = None
        self.checkpoint_dir = ACE_STEP_ROOT / "checkpoints"
        extra = getattr(self, "model_extra", {})
        self.dit_model = extra.get("dit_model", "acestep-v15-turbo")
        # 官方 GPU 分档里 8GB 只推荐 0.6B LM；1.7B 需要 12GB 以上
        self.lm_model = extra.get("lm_model", "acestep-5Hz-lm-0.6B")

    @staticmethod
    def _add_import_path():
        if str(ACE_STEP_ROOT) not in sys.path:
            sys.path.insert(0, str(ACE_STEP_ROOT))

    def _check_model_file(self):
        if not (ACE_STEP_ROOT / "acestep").is_dir():
            raise FileNotFoundError(
                f"未找到 ACE-Step 源码：{ACE_STEP_ROOT}，请先把 ACE-Step-1.5 克隆到该目录并安装依赖")
        self._add_import_path()
        from acestep.model_downloader import ensure_lm_model, ensure_main_model

        # 主模型包（DiT turbo、VAE、文本编码器）缺失时由 ACE-Step 自带下载器补齐；
        # 非默认的 LM 不在主模型包里，要单独下
        ok, message = ensure_main_model(self.checkpoint_dir, token=huggingface_token)
        if not ok:
            raise RuntimeError(f"ACE-Step 主模型下载失败：{message}")
        ok, message = ensure_lm_model(self.lm_model, self.checkpoint_dir, token=huggingface_token)
        if not ok:
            raise RuntimeError(f"ACE-Step LM {self.lm_model} 下载失败：{message}")

    def _load_model(self):
        self._add_import_path()
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        print(f"🔄 正在加载 {self.model_name}（DiT {self.dit_model} + LM {self.lm_model}）...")
        dit_handler = AceStepHandler()
        # 8GB 下 DiT 与 LM 放不下同时驻留：两者都开 offload，推理时轮流上显存
        message, ok = dit_handler.initialize_service(
            project_root=str(ACE_STEP_ROOT),
            config_path=self.dit_model,
            device=self.device,
            offload_to_cpu=True,
            offload_dit_to_cpu=True,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step DiT 加载失败：{message}")

        llm_handler = LLMHandler()
        # vllm 不支持 Windows，只能用 PyTorch 后端
        message, ok = llm_handler.initialize(
            checkpoint_dir=str(self.checkpoint_dir),
            lm_model_path=self.lm_model,
            backend="pt",
            device=self.device,
            offload_to_cpu=True,
        )
        if not ok:
            raise RuntimeError(f"ACE-Step LM 加载失败：{message}")

        self.pipe = dit_handler
        self.llm_handler = llm_handler
        print(f"✅ {self.model_name} 加载完成")
        print_vram_usage()

    async def generate_music(self) -> Path:
        self.ensure_model_loaded()
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        print(f"🎵 开始生成音乐 | 时长: {self.duration}s")
        params = GenerationParams(
            caption=self.prompt,
            # 游戏 BGM 多为纯音乐：没给歌词就按纯音乐生成，避免模型自己编词哼唱
            lyrics=self.lyrics or "[Instrumental]",
            instrumental=not self.lyrics,
            duration=self.duration,
            seed=self.seed,
        )
        # 只返回一首，批量只会多占显存
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[self.seed],
            audio_format="wav",
        )

        # ACE-Step 只在阶段切换时回调进度，取消也只能在这些点生效
        def _progress(*args, **kwargs):
            self.check_cancelled()

        try:
            result = await asyncio.to_thread(
                generate_music, self.pipe, self.llm_handler, params, config,
                save_dir=self.output_dir, progress=_progress)
        finally:
            self.torch.cuda.empty_cache()

        # ACE-Step 会把回调里抛出的异常吞成失败结果，取消要在这里重新识别
        self.check_cancelled()
        if not result.success or not result.audios or not result.audios[0].get("path"):
            raise RuntimeError(f"ACE-Step 生成失败：{result.error or result.status_message}")
        path = Path(result.audios[0]["path"])
        print(f"✅ 生成完成: {path.name}")
        return path

    def parse_params(self, raw: dict):
        self.prompt = raw.get("content", "")
        self.lyrics = raw.get("lyrics", "").strip()
        self.output_dir = get_temp_dir(raw.get("output_dir", ""))
        # ACE-Step 最短 10 秒；API 和资料库模板不经过面板的滑块限制，这里兜底
        self.duration = max(10.0, float(raw.get("duration", 10)))
        self.seed = int(raw.get("seed", 42))


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
    generator = AceStepMusicGenerator(model_name="Ace-Step1.5", device="auto")
    generator.parse_params({"content": ORI_STYLE_PROMPTS["forest_exploration"], "duration": 30})
    print(asyncio.run(generator.generate_music()))
