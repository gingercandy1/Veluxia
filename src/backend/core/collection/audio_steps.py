"""音频类步骤的 runner（ADR 0004）：语音、音效、音乐都走 speech 类生成器的 generate_music()，
生成后的纯 CPU 后处理（如 BGM 无缝循环）也放在这里。

具体做什么由模板参数决定：模型名选生成器（TTS 音色设计 / Stable Audio / Ace-Step），
其余参数原样交给生成器的 parse_params。content 缺省时用条目原文——音频不拼风格锁，
否则风格词会被念出来或写进音效描述。
"""
from pathlib import Path
from typing import Any

from src.backend.core.collection.generator_runner import GeneratorStepRunner
from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.collection.template import PROMPT_FIELD
from src.shared.enum_type import FactoryType


class SpeechGenerateRunner(GeneratorStepRunner):
    type_name = "speech.generate"
    factory_type = FactoryType.Speech
    output_suffix = None

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params.setdefault("content", ctx.values.get(PROMPT_FIELD, ""))
        if not str(params["content"]).strip():
            raise ValueError(f"{self.type_name} 没有要生成的内容")
        return params

    def invoke(self, generator):
        return generator.generate_music()


class AudioLoopRunner(StepRunner):
    """把音乐首尾做成无缝循环（纯 CPU）：游戏 BGM 要循环播放，原曲首尾接不上会有明显断点。

    把结尾 crossfade 秒淡出、叠到开头的淡入上，再去掉结尾这段：
    循环播放时，末尾本来就接着"被叠进开头的那段结尾"，开头又自然过渡回原曲，接缝两侧都连续。
    """
    type_name = "audio.loop"

    def run(self, ctx: StepContext) -> list[Path]:
        import numpy as np
        import soundfile as sf

        if len(ctx.inputs) != 1:
            raise ValueError(f"{self.type_name} 需要恰好一段输入音频，实际 {len(ctx.inputs)} 段")
        audio, sample_rate = sf.read(str(ctx.inputs[0]), always_2d=True)
        crossfade = float(ctx.params.get("crossfade", 2.0))
        overlap = round(crossfade * sample_rate)
        if overlap <= 0 or len(audio) <= overlap * 2:
            raise ValueError(
                f"{self.type_name} 的过渡时长 {crossfade}s 不合适：音频只有 "
                f"{len(audio) / sample_rate:.1f}s，过渡必须大于 0 且短于一半时长")

        # 等功率曲线：线性淡入淡出在接缝中间会有明显的音量凹陷
        curve = np.linspace(0.0, np.pi / 2, overlap)[:, None]
        seam = audio[-overlap:] * np.cos(curve) + audio[:overlap] * np.sin(curve)
        looped = np.concatenate([seam, audio[overlap:-overlap]])

        path = ctx.out_dir / f"{ctx.step_id}.wav"
        sf.write(str(path), looped, sample_rate)
        ctx.meta.update({"crossfade": crossfade, "seconds": round(len(looped) / sample_rate, 2)})
        return [path]


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    SpeechGenerateRunner(),
    AudioLoopRunner(),
)
