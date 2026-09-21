import functools
import gc
import json
from pathlib import Path

from src.backend.core.model_base import GeneratorFactory
from src.shared.settings import PROJECT_ROOT

_MODEL_DIR = Path(PROJECT_ROOT) / "models" / "prompt_refiner" / "Qwen2.5-1.5B-Instruct"

_STYLES_PATH = Path(__file__).with_name("prompt_styles.json")


@functools.lru_cache(maxsize=1)
def _load_styles() -> list[dict]:
    # 文件缺失或损坏要直接暴露：静默返回空表会让"参考风格"悄悄失效
    with open(_STYLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["styles"]


def find_style_hints(text: str) -> list[str]:
    """返回提示词里命中的参考风格描述。小模型不认识具体游戏，靠这里补足视觉特征。"""
    lowered = text.lower()
    return [
        style["description"]
        for style in _load_styles()
        if any(key.lower() in lowered for key in style["keys"])
    ]


_COMMON_RULES = (
    "Rules:\n"
    "- Keep every explicit requirement from the user.\n"
    "- If the request offers several options or asks for a random pick, "
    "choose exactly ONE concrete option yourself and describe only that one.\n"
    "- If a reference game or art style is named, describe its visual traits "
    "(palette, lighting, shapes, mood) instead of only repeating the name.\n"
    "- Output only the final prompt as one English paragraph. "
    "No explanations, no quotes, no lists."
)

_SYSTEM_PROMPTS = {
    "image": (
        "You are a prompt engineer for text-to-image models that produce 2D game art assets.\n"
        "Rewrite the user's request into one detailed image prompt covering: "
        "subject, art style, color palette, lighting, composition and background. "
        "Keep it under 80 words.\n" + _COMMON_RULES
    ),
    "animation": (
        "You are a prompt engineer for image-to-video and text-to-video models.\n"
        "Rewrite the user's request into one detailed video prompt covering: "
        "subject, motion, camera movement, art style and lighting. "
        "Keep it under 70 words.\n" + _COMMON_RULES
    ),
}


# 小模型光靠规则文字管不住语言和长度，用示例对话把"英文、单段、随机选一个"钉死
_EXAMPLES = {
    "image": (
        ("画一个奥日与黑暗森林风格的植物元素，随机选一种",
         "A single glowing bioluminescent mushroom cluster with translucent teal caps, "
         "soft hand-painted 2D game art in the style of Ori and the Blind Forest, "
         "luminous rim light, deep indigo forest palette, centered composition, "
         "isolated on a plain dark background"),
        ("像素风格的木剑图标",
         "A wooden sword icon, pixel art, 32-bit retro game style, warm brown tones, "
         "clean black outline, centered, isolated on a transparent-looking flat background"),
    ),
    "animation": (
        ("战士挥剑",
         "A side-view armored warrior swings a longsword in a smooth downward slash, "
         "cape fluttering, static camera, hand-painted 2D game animation style, "
         "warm rim lighting"),
    ),
}


class PromptRefiner:
    """用小型指令模型把用户的简短需求改写成生图/生视频更有效的英文提示词。

    模型只在调用期间驻留：8GB 显存放不下"优化器 + 生图模型"，用完立即释放，
    让位给后面的生成模型。
    """

    @staticmethod
    def supported_modes() -> tuple[str, ...]:
        return tuple(_SYSTEM_PROMPTS)

    def refine(self, text: str, mode: str) -> str:
        if mode not in _SYSTEM_PROMPTS:
            raise ValueError(f"不支持的提示词优化类型: {mode}")
        if not text.strip():
            return text
        if not _MODEL_DIR.exists():
            raise FileNotFoundError(f"提示词优化模型不存在: {_MODEL_DIR}")

        # 优化器占约 3GB 显存，和驻留的生图模型放不到一起：exclusive() 会先卸载它们，
        # 并在优化期间挡住新的生成请求；同时也串行化了并发的优化请求
        with GeneratorFactory.exclusive():
            return self._run(text.strip(), mode)

    def _run(self, text: str, mode: str) -> str:
        system_prompt, examples = _SYSTEM_PROMPTS[mode], _EXAMPLES[mode]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(str(_MODEL_DIR), torch_dtype=dtype).to(device)
        try:
            messages = [{"role": "system", "content": system_prompt}]
            for question, answer in examples:
                messages += [{"role": "user", "content": question},
                             {"role": "assistant", "content": answer}]
            hints = find_style_hints(text)
            user_message = text
            if hints:
                style_lines = "\n".join(hints)
                user_message += f"\n\nStyle reference (use these traits, do not mention names):\n{style_lines}"
            messages.append({"role": "user", "content": user_message})
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=120,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            refined = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().strip('"')
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        if not refined:
            raise RuntimeError("提示词优化结果为空")
        return refined
