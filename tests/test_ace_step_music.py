"""Ace-Step 音乐生成器契约（无权重、无 GPU 可跑）。

Seam：__new__ 绕开 __init__ 的 models.json 读取，acestep.* 用假模块打桩，
只验证路径、加载失败与生成失败都会抛异常、参数解析和注册。
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

from src.backend.core.speech import ace_step_music
from src.backend.core.speech.ace_step_music import ACE_STEP_ROOT, AceStepMusicGenerator
from src.shared.settings import PROJECT_ROOT


def _generator(**attrs):
    generator = AceStepMusicGenerator.__new__(AceStepMusicGenerator)
    generator.model_name = "Ace-Step1.5"
    generator.device = "cpu"
    generator.cancel_event = None
    generator.load_stage = {}
    generator.pipe = None
    generator.llm_handler = None
    generator.dit_model = "acestep-v15-turbo"
    generator.lm_model = "acestep-5Hz-lm-0.6B"
    generator.checkpoint_dir = ACE_STEP_ROOT / "checkpoints"
    for name, value in attrs.items():
        setattr(generator, name, value)
    return generator


def _stub_acestep(monkeypatch, **modules):
    """把 acestep 及其子模块换成假模块，避免真的 import 第三方包和 torch 权重。"""
    monkeypatch.setitem(sys.modules, "acestep", types.ModuleType("acestep"))
    for name, attrs in modules.items():
        module = types.ModuleType(f"acestep.{name}")
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, f"acestep.{name}", module)


def test_source_root_points_at_backend_speech_dir():
    assert ACE_STEP_ROOT == Path(PROJECT_ROOT) / "src" / "backend" / "core" / "speech" / "ACE_Step"


def test_check_model_file_reports_missing_source(tmp_path, monkeypatch):
    monkeypatch.setattr(ace_step_music, "ACE_STEP_ROOT", tmp_path / "ACE_Step")
    with pytest.raises(FileNotFoundError, match="ACE-Step 源码"):
        _generator()._check_model_file()


class _FailingDit:
    def initialize_service(self, **kwargs):
        return "DiT boom", False


class _OkDit:
    def initialize_service(self, **kwargs):
        type(self).kwargs = kwargs
        return "ok", True


class _FailingLlm:
    def initialize(self, **kwargs):
        return "LM boom", False


class _OkLlm:
    def initialize(self, **kwargs):
        type(self).kwargs = kwargs
        return "ok", True


def test_load_model_raises_when_dit_fails(monkeypatch):
    _stub_acestep(monkeypatch, handler={"AceStepHandler": _FailingDit},
                  llm_inference={"LLMHandler": _OkLlm})
    generator = _generator()
    with pytest.raises(RuntimeError, match="DiT boom"):
        generator._load_model()
    assert generator.pipe is None


def test_load_model_raises_when_lm_fails(monkeypatch):
    _stub_acestep(monkeypatch, handler={"AceStepHandler": _OkDit},
                  llm_inference={"LLMHandler": _FailingLlm})
    generator = _generator()
    with pytest.raises(RuntimeError, match="LM boom"):
        generator._load_model()
    # 两个都成功才算加载完成，否则 ensure_model_loaded 会误以为已就绪
    assert generator.pipe is None


def test_load_model_uses_pt_backend_and_offload(monkeypatch):
    _stub_acestep(monkeypatch, handler={"AceStepHandler": _OkDit},
                  llm_inference={"LLMHandler": _OkLlm})
    monkeypatch.setattr(ace_step_music, "print_vram_usage", lambda: None)
    generator = _generator()
    generator._load_model()
    assert isinstance(generator.pipe, _OkDit) and isinstance(generator.llm_handler, _OkLlm)
    assert _OkDit.kwargs["project_root"] == str(ACE_STEP_ROOT)
    assert _OkDit.kwargs["offload_to_cpu"] and _OkDit.kwargs["offload_dit_to_cpu"]
    assert _OkLlm.kwargs["backend"] == "pt" and _OkLlm.kwargs["offload_to_cpu"]
    assert _OkLlm.kwargs["lm_model_path"] == "acestep-5Hz-lm-0.6B"


def test_parse_params_defaults_to_instrumental(tmp_path):
    generator = _generator()
    generator.parse_params({"content": "calm forest", "output_dir": str(tmp_path),
                            "duration": "30"})
    assert generator.prompt == "calm forest" and generator.lyrics == ""
    assert generator.duration == 30.0 and generator.seed == 42
    assert generator.output_dir == str(tmp_path)


def _stub_inference(monkeypatch, result):
    calls = []

    def generate_music(dit, llm, params, config, save_dir=None, progress=None):
        calls.append((params, config, save_dir))
        return result

    _stub_acestep(monkeypatch, inference={
        "GenerationParams": lambda **kw: types.SimpleNamespace(**kw),
        "GenerationConfig": lambda **kw: types.SimpleNamespace(**kw),
        "generate_music": generate_music,
    })
    return calls


def test_generate_music_returns_first_audio(tmp_path, monkeypatch):
    audio = tmp_path / "song.wav"
    result = types.SimpleNamespace(success=True, audios=[{"path": str(audio), "params": {}}],
                                   error=None, status_message="")
    calls = _stub_inference(monkeypatch, result)
    generator = _generator(pipe=object(), llm_handler=object())
    generator.parse_params({"content": "boss battle", "output_dir": str(tmp_path), "seed": 7})

    assert asyncio.run(generator.generate_music()) == audio
    params, config, save_dir = calls[0]
    assert params.lyrics == "[Instrumental]" and params.instrumental is True
    assert config.batch_size == 1 and config.seeds == [7] and config.use_random_seed is False
    assert save_dir == str(tmp_path)


def test_generate_music_raises_on_failure(tmp_path, monkeypatch):
    result = types.SimpleNamespace(success=False, audios=[], error="OOM", status_message="")
    _stub_inference(monkeypatch, result)
    generator = _generator(pipe=object(), llm_handler=object())
    generator.parse_params({"content": "x", "output_dir": str(tmp_path)})
    with pytest.raises(RuntimeError, match="OOM"):
        asyncio.run(generator.generate_music())


def test_ace_step_registered_and_used_by_music_template():
    from src.backend.core.collection.template import load_template
    from src.backend.core.generator_registry import register_all
    from src.backend.core.model_base import GeneratorFactory
    from src.shared.enum_type import FactoryType

    register_all()
    with open(f"{PROJECT_ROOT}/models.json", encoding="utf-8") as f:
        assert "Ace-Step1.5" in json.load(f)["speech"]
    assert "Ace-Step1.5" in GeneratorFactory.get_generator_names(FactoryType.Speech)
    [step] = load_template("audio_music").steps
    assert step.params["model_name"] == "Ace-Step1.5"
