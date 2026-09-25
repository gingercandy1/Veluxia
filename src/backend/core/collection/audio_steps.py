"""音频类步骤的 runner（ADR 0004）：语音、音效、音乐都走 speech 类生成器的 generate_music()。

具体做什么由模板参数决定：模型名选生成器（TTS 音色设计 / Stable Audio / Ace-Step），
其余参数原样交给生成器的 parse_params。content 缺省时用条目原文——音频不拼风格锁，
否则风格词会被念出来或写进音效描述。
"""
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


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    SpeechGenerateRunner(),
)
