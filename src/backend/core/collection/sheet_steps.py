"""组图类步骤的 runner：先让 LLM 把一类道具展开成 N 个具体变体，再把变体逐个编号写进出图提示词。

只给出图模型一句"各种蘑菇 + 大小形状各异"时，蒸馏模型多样性很低，一张图里的元素
大小、颜色都差不多；逐个写明"矮胖的红伞""细高的白杆"后，模型会照着清单逐个画，差异才拉得开。
执行器按步骤外层循环，一批条目只加载一次 LLM，写完再换出图模型。
"""
import json
import random
from pathlib import Path

from src.backend.core.collection.dialogue_steps import find_input
from src.backend.core.collection.generator_runner import GeneratorStepRunner
from src.backend.core.collection.image_steps import ImageGenerateRunner
from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.collection.template import PROMPT_FIELD
from src.shared.enum_type import FactoryType

# 产物用固定文件名：下游按文件名找输入，不依赖模板里的步骤 id
VARIANTS_NAME = "variants.json"


def build_variant_messages(subject: str, count: int) -> list[dict]:
    system = (
        "你是游戏美术策划，要在同一张素材图里画出同一类道具的多个变体。"
        f"把用户给的道具展开成正好 {count} 个变体，每个用一句简短的英文描述外观。\n"
        "要求：\n"
        "- 全部属于用户给的这一类道具，不要写别的东西。\n"
        "- 变体之间差异要大：尺寸从很小到很大都要有，高矮胖瘦、轮廓形状、主色调、"
        "数量（单个 / 一小簇）、细节（斑点、裂纹、藤蔓等）都要拉开，不要只换颜色。\n"
        "- 每句先写尺寸，如 tiny / small / medium / large / huge，再写形状、颜色和细节。\n"
        "- 只写看得见的外观，不写画风、背景、光照。\n"
        '只输出 JSON，格式：{"variants": ["描述1", "描述2"]}'
    )
    user = f"道具：{subject}\n请写 {count} 个变体。"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_variants(text: str) -> list[str]:
    """解析变体清单（LLM 输出或用户修改后的内容），格式不对时抛 ValueError 说明原因。"""
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"变体清单不是合法的 JSON：{exc}") from exc
    raw = data.get("variants") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        # 用 ValueError 而不是 TypeError：这是内容格式错误，审阅接口按 ValueError 返回 400
        raise ValueError("变体清单里没有 variants 数组")  # noqa: TRY004
    variants = [str(value).strip() for value in raw if str(value).strip()]
    if not variants:
        raise ValueError("变体清单是空的")
    return variants


def dump_variants(variants: list[str]) -> str:
    return json.dumps({"variants": variants}, ensure_ascii=False, indent=2)


class VariantsRunner(GeneratorStepRunner):
    """LLM 写变体清单，产物是 variants.json。"""
    type_name = "text.variants"
    factory_type = FactoryType.Text

    def run(self, ctx: StepContext) -> list[Path]:
        # 用条目原文而不是拼了风格锁的出图提示词：风格词会被 LLM 当成外观写进每个变体
        subject = ctx.values.get(PROMPT_FIELD, "").strip()
        if not subject:
            raise ValueError("缺少道具描述")
        count = int(ctx.params.get("count", 6))
        if count < 2:
            raise ValueError(f"{self.type_name} 的 count 至少是 2：{count}")
        temperature = float(ctx.params.get("temperature", 0.9))

        generator = self.generator
        generator.check_cancelled()
        # 每次随机种子：重做这一步要能拿到不同的变体，而不是同一份清单
        reply = generator.complete_chat(build_variant_messages(subject, count),
                                        max_tokens=60 * count + 100, temperature=temperature,
                                        json_output=True, seed=random.randrange(2**31))
        variants = parse_variants(reply)
        if len(variants) < count:
            raise ValueError(f"模型只写了 {len(variants)} 个变体，要 {count} 个，请重做这一步")
        variants = variants[:count]

        path = ctx.out_dir / VARIANTS_NAME
        path.write_text(dump_variants(variants), encoding="utf-8")
        ctx.meta.update({"model": self._model_name, "prompt": subject, "variants": variants,
                         "params": {"count": count, "temperature": temperature}})
        return [path]

    def validate_edit(self, content: str) -> str:
        return dump_variants(parse_variants(content))


class SheetGenerateRunner(ImageGenerateRunner):
    """按变体清单出组图：条目描述（含风格锁）+ 编号清单 + 构图后缀。"""
    type_name = "image.generate_sheet"

    def build_params(self, ctx: StepContext) -> dict:
        variants = parse_variants(find_input(ctx, VARIANTS_NAME).read_text(encoding="utf-8"))
        listing = "; ".join(f"({index}) {text}" for index, text in enumerate(variants, start=1))
        params = {k: v for k, v in ctx.params.items() if k not in ("model_name", "prompt_suffix")}
        parts = [ctx.prompt, f"{len(variants)} separate objects: {listing}",
                 ctx.params.get("prompt_suffix", "")]
        params["content"] = ", ".join(p.strip() for p in parts if p.strip())
        if ctx.negative_prompt:
            params["negative_prompt"] = ctx.negative_prompt
        return params


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (VariantsRunner(), SheetGenerateRunner())
