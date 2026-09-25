"""TTS 与动画参数键的单测（无权重、无 GPU 可跑）。

Seam：parse_params / _model_missing 的公开行为，__new__ 绕开权重加载。
torch 沿用 server lifespan 的 preload 方式；soundfile 缺装时打桩（仅 import 层）。
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backend.core.preloader import preloader
preloader.preload("torch", lambda: __import__("torch"))

if "soundfile" not in sys.modules:
    try:
        import soundfile  # noqa: F401
    except ImportError:
        stub = types.ModuleType("soundfile")
        stub.write = lambda *a, **k: None
        sys.modules["soundfile"] = stub


def test_tts_model_missing_detects_absent_dir(tmp_path):
    from src.backend.core.speech.qwen3_tts import Qwen3TTSGenerator
    g = Qwen3TTSGenerator.__new__(Qwen3TTSGenerator)
    g.model_dir = str(tmp_path / "nope")
    assert g._model_missing() is True
    g.model_dir = str(tmp_path)
    assert g._model_missing() is False


def test_animation_accepts_reference_image_key():
    torch = pytest.importorskip("torch", reason="动画 parse 需 torch（本机无，随后端环境跑）")
    from src.backend.core.animation.ltx_video import LTXVideoGenerator
    g = LTXVideoGenerator.__new__(LTXVideoGenerator)
    # __new__ 绕过了 __init__，parse_params 建随机数发生器要用到 device
    g.device = "cpu"
    g.parse_params({"content": "x", "reference_image": "a.png"})
    assert g.reference_image_path == "a.png"
    g.parse_params({"content": "x", "reference_image_path": "b.png"})
    assert g.reference_image_path == "b.png"


def test_animation_reads_decode_noise_scale_from_its_own_key():
    pytest.importorskip("torch", reason="动画 parse 需 torch（本机无，随后端环境跑）")
    from src.backend.core.animation.ltx_video import LTXVideoGenerator
    g = LTXVideoGenerator.__new__(LTXVideoGenerator)
    g.device = "cpu"
    g.parse_params({"content": "x", "decode_timestep": 0.05, "decode_noise_scale": 0.01})
    assert g.decode_timestep == 0.05 and g.decode_noise_scale == 0.01


def test_stable_audio_open_parse_params_defaults(tmp_path):
    pytest.importorskip("torch", reason="parse_params 需 torch（本机无，随后端环境跑）")
    from src.backend.core.speech.stable_audio_open import StableAudioOpenGenerator
    g = StableAudioOpenGenerator.__new__(StableAudioOpenGenerator)
    g.parse_params({"content": "footsteps on gravel", "output_dir": str(tmp_path)})
    assert g.prompt == "footsteps on gravel"
    assert g.negative_prompt is None
    assert g.duration == 5.0
    assert g.num_inference_steps == 8
    assert g.save_path.parent == tmp_path


def test_stable_audio_open_registered_and_resolvable():
    from src.backend.core.generator_registry import register_all
    from src.backend.core.model_base import GeneratorFactory
    from src.shared.enum_type import FactoryType
    from src.shared.settings import PROJECT_ROOT
    import json

    register_all()
    with open(f"{PROJECT_ROOT}/models.json", "r", encoding="utf-8") as f:
        speech_models = json.load(f)["speech"]
    names = GeneratorFactory.get_generator_names(FactoryType.Speech)
    for name in ("Stable-Audio-Open-1.0",):
        assert name in speech_models and name in names
