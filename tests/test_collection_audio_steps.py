"""音频、对话类步骤 runner 契约（ADR 0004）：参数拼装、剧本解析、声线复用与设计、逐句配音。

用假的生成器，无权重、无 GPU 也可跑。
"""
import json
import threading
from typing import ClassVar

import pytest
from PIL import Image

from src.backend.core.collection.audio_steps import SpeechGenerateRunner
from src.backend.core.collection.dialogue_steps import (
    CAST_VOICES_NAME,
    DIALOGUE_NAME,
    SCRIPT_NAME,
    CastVoiceRunner,
    DialogueScriptRunner,
    DialogueSpeakRunner,
    ScriptLine,
    dump_script,
    parse_script,
)
from src.backend.core.collection.image_steps import ResizeRunner
from src.backend.core.collection.steps import CastVoice, StepContext
from src.backend.core.model_base import (
    BaseSpeechGenerator,
    BaseTextGenerator,
    GeneratorFactory,
    SingletonMeta,
)
from src.shared.enum_type import FactoryType
from src.shared.schemas import CollectionItem


class _FakeSpeech(BaseSpeechGenerator):
    output_dir = None
    # (模型名, 参数)：同一个类注册成两个模型名，模拟音色设计和克隆两种 TTS
    calls: ClassVar[list] = []
    loads: ClassVar[list] = []

    def _check_model_file(self):
        pass

    def _load_model(self):
        type(self).loads.append(self.model_name)
        self.pipe = object()

    def parse_params(self, raw):
        self.raw = raw

    def generate_music(self):
        calls = type(self).calls
        calls.append((self.model_name, self.raw))
        path = type(self).output_dir / f"raw_{len(calls)}.wav"
        path.write_bytes(b"RIFF")
        return path


class _FakeLlama(BaseTextGenerator):
    reply = ""
    requests: ClassVar[list] = []

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass

    async def generate(self):
        raise NotImplementedError

    def complete_chat(self, messages, max_tokens=1024, temperature=0.7, json_output=False):
        type(self).requests.append((messages, json_output))
        return type(self).reply


@pytest.fixture
def fake_models(tmp_path):
    _FakeSpeech.output_dir = tmp_path
    _FakeSpeech.calls, _FakeSpeech.loads = [], []
    _FakeLlama.requests = []
    for name in ("fake-design", "fake-clone"):
        GeneratorFactory.register_generator(FactoryType.Speech, name, _FakeSpeech)
    GeneratorFactory.register_generator(FactoryType.Text, "fake-llm", _FakeLlama)
    yield
    for name in ("fake-design", "fake-clone"):
        GeneratorFactory._generators[FactoryType.Speech].pop(name, None)
    GeneratorFactory._generators[FactoryType.Text].pop("fake-llm", None)
    SingletonMeta._instances.pop(_FakeSpeech, None)
    SingletonMeta._instances.pop(_FakeLlama, None)
    GeneratorFactory._holder = None


CAST = [
    CastVoice("骑士", "沉稳的老兵", voice_prompt="低沉男声"),
    CastVoice("公主", "活泼", sample_audio=None),
]


def _ctx(tmp_path, params, step_id, inputs=(), values=None, cast=()):
    out_dir = tmp_path / "pack" / "x"
    out_dir.mkdir(parents=True, exist_ok=True)
    return StepContext(
        item=CollectionItem(id="x", prompt="raw"), prompt="raw, style", negative_prompt="",
        inputs=list(inputs), params=params, out_dir=out_dir, step_id=step_id,
        values=values or {"prompt": "raw"}, cast=list(cast),
    )


def _run(runner, ctx):
    with runner.open(ctx.params, threading.Event()):
        return runner.run(ctx)


def _write_script(tmp_path, lines):
    path = tmp_path / SCRIPT_NAME
    path.write_text(dump_script(lines), encoding="utf-8")
    return path


def test_speech_step_uses_raw_prompt_and_records_meta(tmp_path, fake_models):
    params = {"model_name": "fake-design", "voice_prompt": "低沉"}
    ctx = _ctx(tmp_path, params, "voice")
    [path] = _run(SpeechGenerateRunner(), ctx)
    # 音频不拼风格锁
    assert _FakeSpeech.calls == [("fake-design", {"voice_prompt": "低沉", "content": "raw"})]
    assert path.name == "voice.wav" and path.is_file()
    assert ctx.meta == {"model": "fake-design", "prompt": "raw", "params": {"voice_prompt": "低沉"}}


def test_speech_step_rejects_empty_content(tmp_path, fake_models):
    ctx = _ctx(tmp_path, {"model_name": "fake-design"}, "voice", values={"prompt": " "})
    with pytest.raises(ValueError, match="没有要生成的内容"):
        _run(SpeechGenerateRunner(), ctx)


@pytest.mark.parametrize("text", [
    '{"lines": [{"speaker": "骑士", "emotion": "平静", "text": "出发"}]}',
    '```json\n{"lines": [{"speaker": "骑士", "text": "出发", "emotion": "平静"}]}\n```',
    '[{"speaker": "骑士", "emotion": "平静", "text": "出发"}]',
])
def test_parse_script_accepts_common_shapes(text):
    assert parse_script(text) == [ScriptLine("骑士", "出发", "平静")]


@pytest.mark.parametrize("text, message", [
    ("not json", "合法的 JSON"),
    ('{"lines": []}', "没有台词"),
    ('{"lines": [{"speaker": "骑士"}]}', "缺少说话人或台词"),
    ('{"lines": ["hi"]}', "格式不对"),
])
def test_parse_script_rejects_bad_scripts(text, message):
    with pytest.raises(ValueError, match=message):
        parse_script(text)


def test_script_step_asks_llm_for_json_and_trims_lines(tmp_path, fake_models):
    lines = [{"speaker": "骑士", "emotion": "平静", "text": f"第{i}句"} for i in range(5)]
    _FakeLlama.reply = json.dumps({"lines": lines}, ensure_ascii=False)
    params = {"model_name": "fake-llm", "max_lines": 3}
    ctx = _ctx(tmp_path, params, "script", values={"prompt": "城门相遇"}, cast=CAST)
    [path] = _run(DialogueScriptRunner(), ctx)

    messages, json_output = _FakeLlama.requests[0]
    assert json_output is True
    assert "城门相遇" in messages[-1]["content"] and "骑士" in messages[-1]["content"]
    assert path.name == SCRIPT_NAME
    assert [line.text for line in parse_script(path.read_text(encoding="utf-8"))] == \
        ["第0句", "第1句", "第2句"]
    assert ctx.meta["lines"] == 3 and ctx.meta["model"] == "fake-llm"


def test_script_step_rejects_unknown_speakers(tmp_path, fake_models):
    _FakeLlama.reply = '{"lines": [{"speaker": "路人", "text": "嗨"}]}'
    ctx = _ctx(tmp_path, {"model_name": "fake-llm"}, "script", cast=CAST)
    with pytest.raises(ValueError, match="路人"):
        _run(DialogueScriptRunner(), ctx)


def test_script_edit_is_validated_and_normalized():
    runner = DialogueScriptRunner()
    edited = runner.validate_edit('[{"speaker": "骑士", "text": "改过了"}]')
    assert parse_script(edited) == [ScriptLine("骑士", "改过了")]
    with pytest.raises(ValueError):
        runner.validate_edit("{}")


def test_cast_voice_reuses_samples_and_designs_the_rest(tmp_path, fake_models):
    sample = tmp_path / "hero.wav"
    sample.write_bytes(b"RIFF")
    cast = [CastVoice("骑士", "老兵", sample_audio=sample, sample_text="我是骑士"),
            CastVoice("公主", "活泼的少女", voice_prompt="清脆女声"),
            CastVoice("旁观者", "不说话")]
    script = _write_script(tmp_path, [ScriptLine("公主", "你好"), ScriptLine("骑士", "嗯"),
                                      ScriptLine("公主", "走吧")])
    ctx = _ctx(tmp_path, {"model_name": "fake-design", "sample_line": "试音"}, "voices",
               inputs=[script], cast=cast)
    outputs = _run(CastVoiceRunner(), ctx)

    # 只为没有样本的说话人设计一次声音，不出场的角色不处理
    assert _FakeSpeech.calls == [("fake-design", {"content": "试音", "voice_prompt": "清脆女声",
                                                  "language": "Chinese"})]
    voices = json.loads(outputs[0].read_text(encoding="utf-8"))
    assert outputs[0].name == CAST_VOICES_NAME
    assert voices["骑士"] == {"audio": str(sample), "text": "我是骑士"}
    assert voices["公主"] == {"audio": str(outputs[1]), "text": "试音"}
    assert ctx.meta["designed"] == ["公主"] and ctx.meta["reused"] == ["骑士"]


def test_cast_voice_skips_loading_when_everyone_has_a_sample(tmp_path, fake_models):
    sample = tmp_path / "hero.wav"
    sample.write_bytes(b"RIFF")
    script = _write_script(tmp_path, [ScriptLine("骑士", "嗯")])
    cast = [CastVoice("骑士", sample_audio=sample, sample_text="我是骑士")]
    _run(CastVoiceRunner(), _ctx(tmp_path, {"model_name": "fake-design"}, "voices",
                                 inputs=[script], cast=cast))
    assert _FakeSpeech.loads == [] and _FakeSpeech.calls == []


def test_speak_step_clones_each_line_and_writes_dialogue(tmp_path, fake_models):
    script = _write_script(tmp_path, [ScriptLine("骑士", "出发", "坚定"),
                                      ScriptLine("公主", "等等我", "惊讶")])
    voices = tmp_path / CAST_VOICES_NAME
    voices.write_text(json.dumps({"骑士": {"audio": "k.wav", "text": "我是骑士"},
                                  "公主": {"audio": "p.wav", "text": "试音"}}), encoding="utf-8")
    ctx = _ctx(tmp_path, {"model_name": "fake-clone"}, "speak", inputs=[script, voices])
    outputs = _run(DialogueSpeakRunner(), ctx)

    assert [raw["reference_audio_path"] for _, raw in _FakeSpeech.calls] == ["k.wav", "p.wav"]
    assert _FakeSpeech.calls[1][1]["content"] == "等等我"
    assert outputs[0].name == DIALOGUE_NAME
    assert [p.name for p in outputs[1:]] == ["line_01.wav", "line_02.wav"]
    dialogue = json.loads(outputs[0].read_text(encoding="utf-8"))["lines"]
    assert dialogue[1] == {"speaker": "公主", "text": "等等我", "emotion": "惊讶",
                           "audio": "line_02.wav"}


def test_speak_step_requires_voice_for_every_speaker(tmp_path, fake_models):
    script = _write_script(tmp_path, [ScriptLine("骑士", "出发")])
    voices = tmp_path / CAST_VOICES_NAME
    voices.write_text("{}", encoding="utf-8")
    ctx = _ctx(tmp_path, {"model_name": "fake-clone"}, "speak", inputs=[script, voices])
    with pytest.raises(ValueError, match="没有声线"):
        _run(DialogueSpeakRunner(), ctx)


def test_resize_fits_image_into_transparent_square(tmp_path):
    source = tmp_path / "in.png"
    Image.new("RGBA", (100, 50), (255, 0, 0, 255)).save(source)
    ctx = _ctx(tmp_path, {"size": 64}, "resize", inputs=[source])
    [path] = ResizeRunner().run(ctx)
    with Image.open(path) as result:
        assert result.size == (64, 64)
        assert result.getpixel((32, 32))[3] == 255 and result.getpixel((32, 2))[3] == 0
    assert ctx.meta == {"size": [64, 64]}
