"""语音识别的契约测试：models.json 清单自洽、按语言分派、字幕格式、纯 CPU 生成器不占显存租约。

无权重、无 GPU 也可跑；真实识别见 docs/smoke-checklist.md。
"""
import pytest

from src.backend.core.model_base import (
    BaseImageGenerator, BaseTranscriptionGenerator, GeneratorFactory, SingletonMeta,
    load_models_config,
)
from src.backend.core.transcription.sherpa_asr import (
    _ARCH_FILES, join_segments, resolve_model, to_srt,
)
from src.shared.enum_type import FactoryType

CATALOG = load_models_config()["transcription"]
CONCRETE = {name: entry for name, entry in CATALOG.items()
            if isinstance(entry, dict) and "prefer" not in entry}


def test_catalog_entries_are_complete():
    assert CONCRETE, "transcription 下至少要有一个具体模型"
    for name, entry in CONCRETE.items():
        assert entry.get("repo_id"), name
        assert entry.get("arch") in _ARCH_FILES, name
        assert entry.get("languages"), name
    auto = CATALOG["Auto"]
    assert auto["auto_detect"] in CONCRETE
    assert all(candidate in CONCRETE for candidate in auto["prefer"])


@pytest.mark.parametrize("language, expected", [
    ("auto", "SenseVoice-Small"),
    ("zh", "SenseVoice-Small"),
    ("yue", "SenseVoice-Small"),
    ("ko", "SenseVoice-Small"),
    ("en", "Parakeet-TDT-0.6B-v3"),
    ("es", "Parakeet-TDT-0.6B-v3"),
    ("ru", "Parakeet-TDT-0.6B-v3"),
    ("th", "Whisper-Turbo"),
])
def test_auto_routes_by_language(language, expected):
    assert resolve_model("Auto", language) == expected


def test_concrete_model_is_used_as_is():
    assert resolve_model("Whisper-Turbo", "zh") == "Whisper-Turbo"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        resolve_model("no-such-model", "zh")


def test_join_keeps_cjk_tight_and_spaces_other_languages():
    assert join_segments(["你好，", "世界。"]) == "你好，世界。"
    assert join_segments(["Hello there.", "How are you?"]) == "Hello there. How are you?"
    assert join_segments(["看这个", "demo", ""]) == "看这个 demo"


def test_srt_format():
    srt = to_srt([{"start": 0.0, "end": 1.5, "text": "first"},
                  {"start": 3661.25, "end": 3662.0, "text": " second "},
                  {"start": 5.0, "end": 6.0, "text": "  "}])
    assert srt == ("1\n00:00:00,000 --> 00:00:01,500\nfirst\n\n"
                   "2\n01:01:01,250 --> 01:01:02,000\nsecond\n")


class _FakeGpu(BaseImageGenerator):
    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass

    async def generate(self):
        pass


class _FakeCpu(BaseTranscriptionGenerator):
    uses_vram = False

    def _check_model_file(self):
        pass

    def _load_model(self):
        self.pipe = object()

    def parse_params(self, raw):
        pass

    async def transcribe(self):
        return {}


@pytest.fixture
def fakes():
    GeneratorFactory.register_generator(FactoryType.Image, "gpu-a", _FakeGpu)
    GeneratorFactory.register_generator(FactoryType.Transcription, "cpu-a", _FakeCpu)
    yield
    GeneratorFactory._generators[FactoryType.Image].pop("gpu-a", None)
    GeneratorFactory._generators[FactoryType.Transcription].pop("cpu-a", None)
    for cls in (_FakeGpu, _FakeCpu):
        SingletonMeta._instances.pop(cls, None)
    GeneratorFactory._holder = None


def test_cpu_generator_neither_evicts_nor_is_evicted_by_gpu_models(fakes):
    with GeneratorFactory.acquire(FactoryType.Image, "gpu-a") as gpu:
        gpu.ensure_model_loaded()
    with GeneratorFactory.acquire(FactoryType.Transcription, "cpu-a") as cpu:
        cpu.ensure_model_loaded()
    assert gpu.pipe is not None, "识别不该把显存里的模型卸掉"

    with GeneratorFactory.acquire(FactoryType.Image, "gpu-a"):
        pass
    assert cpu.pipe is not None, "图片生成不该把 CPU 识别器卸掉"


def test_cpu_generator_is_usable_while_a_gpu_job_runs(fakes):
    with GeneratorFactory.acquire(FactoryType.Image, "gpu-a"):
        with GeneratorFactory.acquire(FactoryType.Transcription, "cpu-a") as cpu:
            assert cpu is not None
