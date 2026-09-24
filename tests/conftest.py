import os

import pytest


@pytest.fixture(scope="session")
def qapp():
    """全进程只能有一个 Qt 应用对象；先建了 QCoreApplication 之后控件测试就建不了 QApplication，
    所以统一在这里建 QApplication（离屏，无需显示器）。"""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
