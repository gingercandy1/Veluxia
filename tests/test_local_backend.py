"""本机后端安装 / 启动与远程连接的契约测试（ADR 0005）：不下载、不起真实进程。"""
import zipfile
from pathlib import Path

from src.app import local_backend
from src.app.client import ApiClient, BackendStatus
from src.app.work import BackendStartupWorker
from src.shared.settings import ConfigManager


def _make_package(tmp_path: Path, marker: str = "new") -> Path:
    archive = tmp_path / "veluxia-backend.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("veluxia-backend/install_backend.bat", "@echo off\n")
        z.writestr("veluxia-backend/app/src/backend/server.py", marker)
    return archive


def test_extract_overwrites_code_but_keeps_user_data(tmp_path):
    root = tmp_path / "backend"
    (root / "app" / "models").mkdir(parents=True)
    (root / "app" / "models" / "weights.bin").write_text("keep")
    (root / "app" / "src" / "backend").mkdir(parents=True)
    (root / "app" / "src" / "backend" / "server.py").write_text("old")
    (root / local_backend.INSTALLED_MARKER).write_text("ok")

    local_backend.extract(_make_package(tmp_path), root, lambda _line: None)

    assert (root / "app" / "src" / "backend" / "server.py").read_text() == "new"
    assert (root / "app" / "models" / "weights.bin").read_text() == "keep"
    # 重新安装前去掉标记，中途失败不会被当成已安装
    assert not (root / local_backend.INSTALLED_MARKER).exists()


def test_extract_rejects_foreign_zip(tmp_path):
    archive = tmp_path / "other.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("something/readme.txt", "x")
    try:
        local_backend.extract(archive, tmp_path / "backend", lambda _line: None)
    except RuntimeError:
        return
    raise AssertionError("非后端安装包应被拒绝")


def test_install_from_local_zip_skips_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(local_backend, "download", lambda *a, **k: calls.append("download"))
    monkeypatch.setattr(local_backend, "run_installer", lambda root, _cb: calls.append("install"))
    archive = _make_package(tmp_path)

    local_backend.install(tmp_path / "backend", lambda _line: None, archive=archive)

    assert calls == ["install"]
    assert archive.exists()  # 用户选的本地 zip 不删


def test_is_installed_needs_marker_and_python(tmp_path):
    assert not local_backend.is_installed(tmp_path)
    (tmp_path / "python").mkdir()
    (tmp_path / "python" / "python.exe").write_text("")
    assert not local_backend.is_installed(tmp_path)
    (tmp_path / local_backend.INSTALLED_MARKER).write_text("ok")
    assert local_backend.is_installed(tmp_path)


def test_backend_env_drops_inherited_python_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "C:/elsewhere")
    env = local_backend._clean_env(tmp_path)
    assert "PYTHONHOME" not in env
    assert env["PYTHONPATH"] == str(tmp_path / "app")


def test_configure_sets_and_clears_token():
    client = ApiClient.instance()
    client.configure("http://10.0.0.2:8765/", "secret")
    assert client.base_url == "http://10.0.0.2:8765"
    assert client._session.headers["Authorization"] == "Bearer secret"
    client.configure("http://127.0.0.1:8765")
    assert "Authorization" not in client._session.headers


def test_backend_page_round_trips_server_settings(qapp, monkeypatch):
    from src.app.ui.setting.page.backend_page import BackendPage

    stored = {"mode": "remote", "url": "http://10.0.0.2:8765", "token": "t", "install_dir": ""}
    written = {}
    monkeypatch.setattr(ConfigManager, "get_section", lambda _m, _s: dict(stored))
    monkeypatch.setattr(ConfigManager, "set",
                        lambda _m, _s, key, value: written.__setitem__(key, value))

    page = BackendPage()
    assert page._remote_radio.isChecked()
    assert page._remote_box.isEnabled() and not page._local_box.isEnabled()

    page._local_radio.setChecked(True)
    page.collect()
    assert written == {**stored, "mode": "local"}


def _run_remote(monkeypatch, url: str, status: BackendStatus) -> list:
    monkeypatch.setattr("src.app.work.probe_backend", lambda _url, _token: status)
    monkeypatch.setattr(BackendStartupWorker, "REMOTE_CONNECT_RETRIES", 1)
    monkeypatch.setattr("src.app.work.time.sleep", lambda _s: None)
    monkeypatch.setattr(ConfigManager, "get_section", lambda _m, _s: {
        "mode": "remote", "url": url, "token": "t"})
    worker = BackendStartupWorker()
    failed = []
    worker.failed.connect(failed.append)
    worker.run()  # 直接在当前线程跑，信号同步投递
    return failed


def test_remote_unauthorized_reports_token_problem(qapp, monkeypatch):
    failed = _run_remote(monkeypatch, "http://10.0.0.2:8765", BackendStatus.UNAUTHORIZED)
    assert len(failed) == 1 and "token" in failed[0]


def test_remote_empty_url_reports_without_probing(qapp, monkeypatch):
    failed = _run_remote(monkeypatch, "", BackendStatus.OK)
    assert len(failed) == 1
