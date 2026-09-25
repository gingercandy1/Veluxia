"""对话类步骤的 runner（ADR 0004）：写剧本 → 准备出场角色声线 → 逐句配音。

一个条目是一段完整对话，句数由 AI 决定，所以条目在新建时就能确定，不需要执行中动态增加条目。
三步各用一个模型（LLM / 音色设计 / 声音克隆），按步骤分组执行时每个模型只加载一次。
全部台词都用克隆模型配音：同一角色的每句话以同一份声线样本为基准，声音才一致。
克隆模式不接受语气指令，所以情绪只写进剧本，供表情头像等后续环节使用。
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.backend.core.collection.audio_steps import SpeechGenerateRunner
from src.backend.core.collection.generator_runner import GeneratorStepRunner, move_into
from src.backend.core.collection.steps import CastVoice, StepContext, StepRunner
from src.backend.core.collection.template import PROMPT_FIELD
from src.shared.enum_type import FactoryType
from src.shared.schemas import DIALOGUE_EMOTIONS

# 产物用固定文件名：下游按文件名找输入，不依赖模板里的步骤 id
SCRIPT_NAME = "script.json"
CAST_VOICES_NAME = "cast_voices.json"
DIALOGUE_NAME = "dialogue.json"
DEFAULT_SAMPLE_LINE = "你好，很高兴认识你，接下来请多关照。"


@dataclass(frozen=True)
class ScriptLine:
    speaker: str
    text: str
    emotion: str = ""


def parse_script(text: str) -> list[ScriptLine]:
    """解析剧本 JSON（AI 输出或用户修改后的内容），格式不对时抛 ValueError 说明原因。"""
    body = text.strip()
    # 模型偶尔会把 JSON 包在 markdown 代码块里
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"剧本不是合法的 JSON：{exc}") from exc
    raw_lines = data.get("lines") if isinstance(data, dict) else data
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("剧本里没有台词（需要 lines 数组）")

    lines = []
    for index, raw in enumerate(raw_lines, start=1):
        if not isinstance(raw, dict):
            # 用 ValueError 而不是 TypeError：这是内容格式错误，审阅接口按 ValueError 返回 400
            raise ValueError(f"第 {index} 句格式不对：{raw!r}")  # noqa: TRY004
        speaker = str(raw.get("speaker", "")).strip()
        line_text = str(raw.get("text", "")).strip()
        if not speaker or not line_text:
            raise ValueError(f"第 {index} 句缺少说话人或台词")
        lines.append(ScriptLine(speaker, line_text, str(raw.get("emotion", "")).strip()))
    return lines


def dump_script(lines: list[ScriptLine]) -> str:
    return json.dumps({"lines": [asdict(line) for line in lines]}, ensure_ascii=False, indent=2)


def build_script_messages(story: str, cast: list[CastVoice], max_lines: int) -> list[dict]:
    roster = "\n".join(f"- {member.name}：{member.description or '（无设定）'}" for member in cast)
    system = (
        "你是游戏编剧，根据剧情写一段角色对话。只输出 JSON，格式："
        '{"lines": [{"speaker": "角色名", "emotion": "情绪", "text": "台词"}]}。\n'
        f"要求：speaker 只能用给定的角色名；emotion 从 {'、'.join(DIALOGUE_EMOTIONS)} 中选；"
        "台词简短口语化、适合配音，不要旁白和动作描写；"
        f"不超过 {max_lines} 句。"
    )
    user = f"出场角色：\n{roster}\n\n剧情：{story}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class DialogueScriptRunner(GeneratorStepRunner):
    """LLM 写剧本。模板里通常标 review：剧本确认后才配音，配音是整条流程里最慢的一步。"""
    type_name = "text.dialogue"
    factory_type = FactoryType.Text

    def run(self, ctx: StepContext) -> list[Path]:
        if not ctx.cast:
            raise ValueError("对话包没有出场角色")
        story = ctx.values.get(PROMPT_FIELD, "").strip()
        if not story:
            raise ValueError("缺少剧情提示")
        max_lines = int(ctx.params.get("max_lines", 8))
        temperature = float(ctx.params.get("temperature", 0.8))

        generator = self.generator
        generator.check_cancelled()
        reply = generator.complete_chat(
            build_script_messages(story, ctx.cast, max_lines),
            max_tokens=int(ctx.params.get("max_tokens", 1024)),
            temperature=temperature,
            json_output=True,
        )
        lines = parse_script(reply)[:max_lines]
        check_speakers(lines, ctx.cast)

        path = ctx.out_dir / SCRIPT_NAME
        path.write_text(dump_script(lines), encoding="utf-8")
        ctx.meta.update({"model": self._model_name, "prompt": story, "lines": len(lines),
                         "params": {"max_lines": max_lines, "temperature": temperature}})
        return [path]

    def validate_edit(self, content: str) -> str:
        return dump_script(parse_script(content))


class CastVoiceRunner(SpeechGenerateRunner):
    """给剧本里的每个说话人准备声线样本：绑定了角色包的直接用，没有的按描述用音色设计模型生成。"""
    type_name = "speech.cast_voice"
    # 全员都绑定了角色包时不需要音色设计模型，用到时才加载
    load_on_open = False

    def run(self, ctx: StepContext) -> list[Path]:
        lines = parse_script(find_input(ctx, SCRIPT_NAME).read_text(encoding="utf-8"))
        members = check_speakers(lines, ctx.cast)
        sample_line = ctx.params.get("sample_line") or DEFAULT_SAMPLE_LINE
        language = ctx.params.get("language", "Chinese")

        voices: dict[str, dict[str, str]] = {}
        outputs: list[Path] = []
        designed = []
        for index, name in enumerate(dict.fromkeys(line.speaker for line in lines), start=1):
            member = members[name]
            if member.sample_audio is not None:
                voices[name] = {"audio": str(member.sample_audio), "text": member.sample_text}
                continue
            voice_prompt = member.voice_prompt or member.description
            if not voice_prompt:
                raise ValueError(f"角色 {name} 没有声线，也没有可用来设计声音的描述")
            path = self.generate_once({"content": sample_line, "voice_prompt": voice_prompt,
                                       "language": language})
            target = move_into(path, ctx.out_dir / f"voice_{index:02d}{path.suffix}")
            voices[name] = {"audio": str(target), "text": sample_line}
            outputs.append(target)
            designed.append(name)

        cast_path = ctx.out_dir / CAST_VOICES_NAME
        cast_path.write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")
        ctx.meta.update({"model": self._model_name if designed else "",
                         "designed": designed,
                         "reused": [name for name in voices if name not in designed]})
        return [cast_path, *outputs]


class DialogueSpeakRunner(SpeechGenerateRunner):
    """逐句克隆配音，最后写出 dialogue.json（剧本 + 每句对应的音频文件），引擎按顺序播放即可。"""
    type_name = "speech.dialogue"

    def run(self, ctx: StepContext) -> list[Path]:
        lines = parse_script(find_input(ctx, SCRIPT_NAME).read_text(encoding="utf-8"))
        voices = json.loads(find_input(ctx, CAST_VOICES_NAME).read_text(encoding="utf-8"))
        language = ctx.params.get("language", "Chinese")

        audio_paths: list[Path] = []
        entries: list[dict[str, Any]] = []
        for index, line in enumerate(lines, start=1):
            voice = voices.get(line.speaker)
            if voice is None:
                raise ValueError(f"说话人 {line.speaker} 没有声线，请重新执行声线步骤")
            path = self.generate_once({"content": line.text, "language": language,
                                       "reference_audio_path": voice["audio"],
                                       "reference_text": voice["text"]})
            target = move_into(path, ctx.out_dir / f"line_{index:02d}{path.suffix}")
            audio_paths.append(target)
            entries.append({**asdict(line), "audio": target.name})

        dialogue_path = ctx.out_dir / DIALOGUE_NAME
        dialogue_path.write_text(json.dumps({"lines": entries}, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        ctx.meta.update({"model": self._model_name, "lines": len(entries),
                         "params": {"language": language}})
        return [dialogue_path, *audio_paths]


def check_speakers(lines: list[ScriptLine], cast: list[CastVoice]) -> dict[str, CastVoice]:
    members = {member.name: member for member in cast}
    unknown = sorted({line.speaker for line in lines} - set(members))
    if unknown:
        raise ValueError(f"剧本里有不在出场角色中的说话人：{', '.join(unknown)}")
    return members


def find_input(ctx: StepContext, name: str) -> Path:
    for path in ctx.inputs:
        if path.name == name:
            return path
    raise ValueError(f"{ctx.step_id} 缺少上游产物 {name}，请检查模板的 inputs")


BUILTIN_RUNNERS: tuple[StepRunner, ...] = (
    DialogueScriptRunner(),
    CastVoiceRunner(),
    DialogueSpeakRunner(),
)
