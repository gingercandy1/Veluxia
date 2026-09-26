"""后端绿色 zip 装配脚本的单测（ADR 0002）。

Seam：assemble_backend 的纯函数（manifest 清单、启动器内容），tmp_path 隔离。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))

import assemble_backend as ab


def test_manifest_contains_runtime_sources():
    m = ab.build_manifest(ROOT)
    for rel in ("src/backend/server.py",
                "src/backend/router/image.py",
                "src/shared/schemas.py",
                "src/shared/settings.py",
                "src/shared/enum_type.py",
                "models.json"):
        assert rel in m, f"manifest 缺少 {rel}"


def test_manifest_excludes_dev_only():
    m = ab.build_manifest(ROOT)
    for rel in m:
        assert not rel.startswith("src/app"), "前端不应进后端包"
        assert "__pycache__" not in rel
    assert not any(r.startswith("src/backend/core/speech/ACE_Step") for r in m), \
        "ACE_Step 独立安装，不进包"


def test_launcher_points_to_embed_python(tmp_path):
    bat = ab.write_launcher(tmp_path)
    text = bat.read_text(encoding="utf-8")
    assert "src.backend.server" in text
    assert "8765" in text
    assert "PYTHONPATH" in text


def test_launcher_installs_on_first_run_and_forwards_args(tmp_path):
    text = ab.write_launcher(tmp_path).read_text(encoding="ascii")
    assert "install_backend.bat" in text
    assert "%*" in text


def test_installer_is_offline_first_then_torch(tmp_path):
    text = ab.write_installer(tmp_path).read_text(encoding="ascii")
    assert "--no-index" in text and "backend.requirements.txt" in text
    assert ab.TORCH_INDEX in text and "torch.requirements.txt" in text
    assert text.index("backend.requirements.txt") < text.index("torch.requirements.txt")


def test_split_requirements_moves_torch_out():
    text = (
        "# comment\n"
        "torch==2.11.0+cu128 ; sys_platform == 'win32'\n"
        "torchao==0.18.0 ; sys_platform == 'win32'\n"
        "torchvision==0.26.0+cu128 ; sys_platform == 'win32'\n"
        "torchaudio==2.11.0+cu128\n"
        "llama-cpp-python @ https://example.com/x.whl ; sys_platform == 'win32'\n"
    )
    base, torch = ab.split_requirements(text)
    assert [line.split("==")[0] for line in torch] == ["torch", "torchvision", "torchaudio"]
    assert any(line.startswith("torchao") for line in base)
    assert any(line.startswith("llama-cpp-python") for line in base)
