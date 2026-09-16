"""打包切分契约测试（ADR 0002）：前后端 pyproject 分离，依赖不串味。

Seam：pyproject 文件本身（tomllib 解析），无构建、无网络、无 GPU。
"""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    with open(ROOT / name, "rb") as f:
        return tomllib.load(f)


def test_workspace_members():
    root = _load("pyproject.toml")
    members = root["tool"]["uv"]["workspace"]["members"]
    assert set(members) == {"app", "backend"}


def test_member_projects_exist():
    app = _load("app/pyproject.toml")
    backend = _load("backend/pyproject.toml")
    assert app["project"]["name"] == "veluxia-app"
    assert backend["project"]["name"] == "veluxia-backend"


def test_dependency_separation():
    """torch/diffusers 不进前端，PySide 不进后端。"""
    app = _load("app/pyproject.toml")
    backend = _load("backend/pyproject.toml")
    app_deps = " ".join(app["project"]["dependencies"]).lower()
    backend_deps = " ".join(backend["project"]["dependencies"]).lower()
    for heavy in ("torch", "diffusers", "transformers"):
        assert heavy not in app_deps, f"前端不应依赖 {heavy}"
    assert "pyside6" not in backend_deps
    assert "rembg" in backend_deps


def test_gitignore_covers_build_artifacts():
    """构建产物不上库，但锁文件必须提交（ADR 0002 收尾）。"""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8", errors="ignore")
    for entry in (".venv", "dist", ".egg-info"):
        assert entry in text, f".gitignore 缺少 {entry}"
    assert "uv.lock" not in text, "uv.lock 必须提交，不可忽略"


def test_frontend_spec_exists():
    """前端 PyInstaller spec 存在且捆了入口与资源。"""
    text = (ROOT / "app" / "veluxia.spec").read_text(encoding="utf-8")
    assert "main.py" in text
    assert "resource" in text
    assert "datas" in text


def test_cuda_locked_single_variant():
    """后端只锁 cu128 一种 torch 变体（ADR 0002）。"""
    backend = _load("backend/pyproject.toml")
    deps = " ".join(backend["project"]["dependencies"]).lower()
    assert "torch" in deps
    sources = backend["tool"]["uv"]["sources"]
    assert "torch" in sources
