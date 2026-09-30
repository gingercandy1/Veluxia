"""分层类步骤的 runner：用 Qwen-Image-Layered 把生成好的整张场景拆成透明图层。

和转面图（edit_steps）一样，文本编码器和 transformer 各占十几 GB 内存，不能同时驻留，
所以拆成"编码"和"去噪"两个步骤：执行器按步骤外层循环，一批条目只加载一次编码器、
一次 transformer。编码结果存成文件，只重做去噪时也不用再加载编码器。
"""
from pathlib import Path
from typing import Any

from src.backend.core.collection.generator_runner import (
    GeneratorStepRunner,
    call_blocking,
    move_into,
)
from src.backend.core.collection.image_steps import describe_image
from src.backend.core.collection.steps import StepContext

_EMBEDS_SUFFIX = ".pt"


class LayerEncodeRunner(GeneratorStepRunner):
    """编码分层用的描述：直接用生成整图时的提示词（条目描述 + 风格锁 + 后缀），
    不让模型看图重写，既省一个模型，也保证描述和画面对得上。"""
    type_name = "image.layer_encode"
    # 生成器分段加载，用到哪段加载哪段，open() 时不预先加载
    load_on_open = False

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        params = {k: v for k, v in ctx.params.items() if k not in ("model_name", "prompt_suffix")}
        suffix = ctx.params.get("prompt_suffix", "")
        params["content"] = ", ".join(p.strip() for p in (ctx.prompt, suffix) if p.strip())
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        if not params["content"]:
            raise ValueError(f"{self.type_name} 缺少描述")
        embeds = self.generate_once(params)
        ctx.meta.update(self.describe(params))
        return [move_into(embeds, ctx.out_dir / f"{ctx.step_id}{_EMBEDS_SUFFIX}")]

    def invoke(self, generator):
        return generator.save_prompt_embeds()


class LayerRunner(GeneratorStepRunner):
    """按整图和编码文件去噪出 N 张透明图层，从底到顶命名为 <步骤>_1.png … <步骤>_N.png。"""
    type_name = "image.layer"
    load_on_open = False

    def build_params(self, ctx: StepContext) -> dict[str, Any]:
        images = [p for p in ctx.inputs if p.suffix.lower() == ".png"]
        embeds = [p for p in ctx.inputs if p.suffix.lower() == _EMBEDS_SUFFIX]
        if len(images) != 1 or len(embeds) != 1:
            raise ValueError(f"{self.type_name} 需要一张要分层的整图和编码步骤的一个编码文件，"
                             f"实际是 {len(images)} 张图、{len(embeds)} 个编码文件")
        params = {k: v for k, v in ctx.params.items() if k != "model_name"}
        params["reference_image"] = str(images[0])
        params["prompt_embeds_path"] = str(embeds[0])
        return params

    def run(self, ctx: StepContext) -> list[Path]:
        params = self.build_params(ctx)
        generator = self.generator
        generator.check_cancelled()
        generator.ensure_model_loaded()
        generator.parse_params(params)
        layers = call_blocking(generator.generate())
        if not layers or not all(Path(p).is_file() for p in layers):
            raise RuntimeError(f"{self.type_name} 没有生成图层，请查看后端日志")
        outputs = [move_into(Path(path), ctx.out_dir / f"{ctx.step_id}_{index}.png")
                   for index, path in enumerate(layers, start=1)]
        ctx.meta.update({**self.describe(params), **describe_image(outputs[0]),
                         "layers": len(outputs)})
        return outputs


BUILTIN_RUNNERS = (LayerEncodeRunner(), LayerRunner())
