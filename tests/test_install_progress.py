"""设置页安装进度：没安装时不占位，安装中有流光和最新一行，结束后停下并按成败着色。"""
from src.app.ui.setting.page.backend_page import BackendPage
from src.app.ui.setting.page.install_progress import InstallProgress


def test_hidden_until_install_starts(qapp):
    progress = InstallProgress()
    assert progress.isHidden() and not progress.is_running()

    progress.start("准备安装")
    assert not progress.isHidden() and progress.is_running()
    progress.append("Downloading torch (42%)")
    assert "42%" in progress.status_label.text()
    assert progress.log.toPlainText().splitlines() == ["准备安装", "Downloading torch (42%)"]

    progress.finish(False, "网络超时")
    assert not progress.is_running() and progress.strip.isHidden()
    assert progress.status_label.property("state") == "error"
    assert progress.log.toPlainText().splitlines()[-1] == "✗ 网络超时"

    progress.start("重试")                 # 重新安装时清掉上次的日志
    assert progress.log.toPlainText() == "重试"


def test_backend_page_has_no_empty_log_box(qapp):
    page = BackendPage()
    assert page._progress.isHidden()
