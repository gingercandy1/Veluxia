from src.app.client import ApiGuardClient
from src.app.ui.setting.page.gpu_page import GpuPage
from src.shared.settings import ConfigManager


def _finish_detect(qapp, page):
    worker = page._detect_worker
    assert worker is not None
    assert worker.wait(5000)
    qapp.processEvents()


def test_gpu_page_detects_only_when_shown_and_off_the_ui_thread(qapp, monkeypatch):
    calls = []

    def detect_device(_client):
        calls.append(True)
        return {"cuda_available": True, "gpus": [{"name": "RTX 4060", "total_memory_gb": 8}]}

    monkeypatch.setattr(ApiGuardClient, "detect_device", detect_device)
    monkeypatch.setattr(ConfigManager, "get", lambda _manager, _section, _key, default=None: default)

    page = GpuPage()
    # 构造时不请求：主窗口建好之前守护服务还没起来，同步请求会拖慢窗口出现
    assert page._cpu_radio.isChecked() and calls == []

    page.show()
    assert not page._refresh_btn.isEnabled()
    _finish_detect(qapp, page)
    assert calls == [True]
    assert "RTX 4060" in page._status_label.text() and page._refresh_btn.isEnabled()


def test_gpu_page_does_not_report_cpu_when_service_is_not_up(qapp, monkeypatch):
    monkeypatch.setattr(ApiGuardClient, "detect_device", lambda _client: {})
    page = GpuPage()
    page.show()
    _finish_detect(qapp, page)
    text = page._status_label.text()
    assert "CPU" not in text and "Refresh" in text
