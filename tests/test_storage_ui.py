"""存储空间页：表格内容、删除确认（手动准备的权重要额外提醒）、失败提示、忙碌时禁用按钮。"""
import pytest
from PySide6.QtWidgets import QMessageBox

from src.app.ui.setting.page import storage_page
from src.app.ui.setting.page.storage_page import StoragePage, format_size
from src.shared.schemas import ModelStorageEntry, ModelStorageResponse

SDXL = ModelStorageEntry(id="models/image/sdxl-base", root="models", category="image",
                         name="sdxl-base", size=7 * 1024 ** 3)
FILM = ModelStorageEntry(id="models/image_frame/film_net", root="models",
                         category="image_frame", name="film_net", size=140 * 1024 ** 2,
                         manual=True)
U2NET = ModelStorageEntry(id="rembg/u2net.onnx", root="rembg", name="u2net.onnx", size=512)


@pytest.mark.parametrize("size, text", [
    (512, "512 B"), (1536, "1.5 KB"), (140 * 1024 ** 2, "140.0 MB"), (7 * 1024 ** 3, "7.0 GB")])
def test_format_size(size, text):
    assert format_size(size) == text


@pytest.fixture
def page(qapp):
    view = StoragePage()
    calls = []
    view._start = lambda fn, *args: calls.append((fn.__name__, args))
    view.calls = calls
    return view


def _listed(page):
    total = SDXL.size + FILM.size + U2NET.size
    page._on_response(ModelStorageResponse(entries=[SDXL, FILM, U2NET], total=total))


def test_loads_once_when_first_shown_and_fills_table(page):
    page.show()
    page.hide()
    page.show()
    assert page.calls == [("list_model_storage", ())]
    assert not page.refresh_btn.isEnabled()
    _listed(page)
    assert page.refresh_btn.isEnabled()
    assert page.table.rowCount() == 3
    assert page.table.item(0, page.COL_NAME).text() == "sdxl-base"
    assert page.table.item(0, page.COL_SIZE).text() == "7.0 GB"
    assert "7.1 GB" in page.total_label.text() and "3" in page.total_label.text()
    page.show()
    assert len(page.calls) == 1  # 统计过了，再切回来不重复遍历


def test_delete_asks_and_warns_for_manual_weights(page, monkeypatch):
    _listed(page)
    asked = []

    def question(_parent, _title, message, *_args):
        asked.append(message)
        return QMessageBox.StandardButton.No
    monkeypatch.setattr(storage_page.QMessageBox, "question", question)
    page._buttons[1].click()   # 插帧权重
    assert "NOT downloaded automatically" in asked[0] and page.calls == []

    monkeypatch.setattr(storage_page.QMessageBox, "question",
                        lambda *args: QMessageBox.StandardButton.Yes)
    page._buttons[0].click()
    assert page.calls == [("delete_model_storage", ("models/image/sdxl-base",))]
    # 删除进行中：其他删除按钮和刷新都禁用，避免并发删
    assert not page._buttons[2].isEnabled()
    assert not page.refresh_btn.isEnabled()

    page._on_response(ModelStorageResponse(entries=[FILM, U2NET], total=FILM.size + U2NET.size))
    assert page.table.rowCount() == 2 and page._buttons[0].isEnabled()


def test_failed_delete_keeps_list_and_shows_reason(page, monkeypatch):
    _listed(page)
    monkeypatch.setattr(storage_page.QMessageBox, "question",
                        lambda *args: QMessageBox.StandardButton.Yes)
    page._buttons[0].click()
    page._on_response(ModelStorageResponse.from_error("正在运行 SDXL，请等它完成后再试"))
    assert not page.error_label.isHidden() and "SDXL" in page.error_label.text()
    assert page.table.rowCount() == 3 and page._buttons[0].isEnabled()
    assert "7.1 GB" in page.total_label.text()
