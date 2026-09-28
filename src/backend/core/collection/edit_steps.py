"""按指令改图类步骤的 runner（ADR 0006）：转面图。

Qwen-Image-Edit-2509 的文本编码器和 transformer 各占约 17GB 内存，不能同时驻留，
所以拆成"编码"和"去噪"两个步骤：执行器按步骤外层循环，一批条目（视角）只加载一次编码器、
一次 transformer。编码结果存成文件，只重做去噪（换个种子）时也不用再加载编码器。
"""
from pathlib import Path
from typing import Any

from PIL import Image

from src.backend.core.collection.frame_steps import reference_scale, white_canvas
from src.backend.core.collection.generator_runner import GeneratorStepRunner, move_into
from src.backend.core.collection.image_steps import describe_image, single_input
from src.backend.core.collection.steps import StepContext
from src.backend.core.collection.template import PROMPT_FIELD

_EMBEDS_SUFFIX = ".pt"


class EditEncodeRunner(GeneratorStepRunner):
    """立绘铺白底，和改图指令一起编码。产物：参考图（去噪要用同一张）+ 编码文件。

    指令只用条目原文 + 后缀，不拼风格锁：画风已经由参考图决定。
    """
    type_name = "image.edit_encode"
    # 生成器分段加载，用到哪段加载哪段，open() 时不预先加载
    load_on_open = False

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items()
                  if k not in ("model_name", "prompt_suffix", "scale")}
        instruction = ctx.values.get(PROMPT_FIELD, "")
        params["content"] = ", ".join(
            p.strip() for p in (instruction, ctx.params.get("prompt_suffix", "")) if p.strip())
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        if not params["content"]:
            raise ValueError(f"{self.type_name} 缺少改图指令")
        scale = reference_scale(self.type_name, ctx, 0.85)
        reference = ctx.out_dir / f"{ctx.step_id}.png"
        with Image.open(single_input(self.type_name, ctx)) as portrait:
            white_canvas(portrait.convert("RGBA"), scale).save(reference)
        params["reference_image"] = str(reference)
        embeds = self.generate_once(params)
        ctx.meta.update(self.describe(params))
        return [reference, move_into(embeds, ctx.out_dir / f"{ctx.step_id}{_EMBEDS_SUFFIX}")]

    def invoke(self, generator):
        return generator.save_prompt_embeds()


class EditRunner(GeneratorStepRunner):
    """按编码步骤的参考图和编码文件去噪出图。"""
    type_name = "image.edit"
    load_on_open = False

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        images = [p for p in ctx.inputs if p.suffix.lower() == ".png"]
        embeds = [p for p in ctx.inputs if p.suffix.lower() == _EMBEDS_SUFFIX]
        if len(images) != 1 or len(embeds) != 1:
            raise ValueError(f"{self.type_name} 需要上游编码步骤的一张参考图和一个编码文件，"
                             f"实际是 {len(images)} 张图、{len(embeds)} 个编码文件")
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params["reference_image"] = str(images[0])
        params["prompt_embeds_path"] = str(embeds[0])
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        paths = super().run(ctx)
        ctx.meta.update(describe_image(paths[0]))
        return paths


BUILTIN_RUNNERS = (EditEncodeRunner(), EditRunner())
