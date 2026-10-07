"""自定义模型存放目录（设置 → 模型存储）的契约：取目录的优先级、前端怎么传给后端、
生成器不再自己写死 models/。无权重、无 GPU。
"""
import re
from pathlib import Path

import pytest

from src.app import local_backend
from src.backend.core import storage
from src.backend.core.model_utils import ensure_models_root, models_root
from src.shared.settings import MODELS_DIR_ENV, PROJECT_ROOT, ConfigManager

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setting(monkeypatch):
    """改内存里的设置，不读写用户真实的 config/setting.json。"""
    monkeypatch.delenv(MODELS_DIR_ENV, raising=False)

    def set_dir(value: str):
        monkeypatch.setitem(ConfigManager()._config, "models", {"dir": value})

    set_dir("")
    return set_dir


def test_defaults_to_models_under_project(setting):
    assert models_root() == Path(PROJECT_ROOT) / "models"


def test_setting_and_env_override_in_order(setting, monkeypatch, tmp_path):
    setting(str(tmp_path / "from_setting"))
    assert models_root() == tmp_path / "from_setting"
    # 前端拉起后端时传的环境变量优先：安装版后端读的是它自己的 setting.json
    monkeypatch.setenv(MODELS_DIR_ENV, str(tmp_path / "from_env"))
    assert models_root() == tmp_path / "from_env"


def test_relative_dir_resolves_against_project_root(setting):
    setting("weights")
    assert models_root() == Path(PROJECT_ROOT) / "weights"


def test_ensure_creates_dir_and_reports_unusable_one(setting, tmp_path):
    setting(str(tmp_path / "new" / "models"))
    assert ensure_models_root().is_dir()
    blocker = tmp_path / "a_file"
    blocker.write_text("x")
    setting(str(blocker / "models"))
    with pytest.raises(RuntimeError, match="模型目录不可用"):
        ensure_models_root()


def test_frontend_passes_dir_only_when_set(setting):
    setting("D:/AI/models")
    assert local_backend.with_models_dir({})[MODELS_DIR_ENV] == "D:/AI/models"
    setting("")
    # 设置里清空了，就算前端自己带着旧变量启动也不能继续往下传
    assert MODELS_DIR_ENV not in local_backend.with_models_dir({MODELS_DIR_ENV: "old"})


def test_storage_page_lists_the_configured_dir(setting, tmp_path):
    setting(str(tmp_path))
    assert dict(storage._roots())["models"] == tmp_path


def test_backend_no_longer_hardcodes_the_models_dir():
    """新生成器照抄旧写法会悄悄绕过设置，下载到默认目录。"""
    hardcoded = re.compile(r'PROJECT_ROOT\)?\s*(/|,)\s*"models"')
    offenders = []
    for path in (ROOT / "src" / "backend").rglob("*.py"):
        if "ACE_Step" in path.parts or path.name == "model_utils.py":
            continue
        if hardcoded.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"请改用 models_root()：{offenders}"
