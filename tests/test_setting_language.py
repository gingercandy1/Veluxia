from src.app.client import ApiGuardClient
from src.app.ui.setting.setting import SettingPage


def test_setting_save_emits_language_change(qapp, monkeypatch):
    monkeypatch.setattr(ApiGuardClient, "detect_device", lambda _client: {})
    page = SettingPage()
    page._general_page._language_combo.setCurrentIndex(
        page._general_page._language_combo.findData("en")
    )
    monkeypatch.setattr(page._config, "save", lambda: None)

    changed = []
    page.language_changed.connect(changed.append)
    page._on_save()

    assert changed == ["en"]