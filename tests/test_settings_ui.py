from src.app.client import ApiGuardClient
from src.app.ui.setting.page.gpu_page import GpuPage
from src.shared.settings import ConfigManager


def test_gpu_page_defaults_to_cpu_and_detects_on_load(qapp, monkeypatch):
    calls = []

    def detect_device(_client):
        calls.append(True)
        return {"cuda_available": False}

    monkeypatch.setattr(ApiGuardClient, "detect_device", detect_device)
    monkeypatch.setattr(ConfigManager, "get", lambda _manager, _section, _key, default=None: default)

    page = GpuPage()

    assert page._cpu_radio.isChecked()
    assert calls == [True]