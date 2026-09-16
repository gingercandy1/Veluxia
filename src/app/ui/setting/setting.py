from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QStackedWidget, QFrame, QMessageBox,
    QLabel, QPushButton, QSizePolicy
)

from src.app.ui.setting.page.gpu_page import GpuPage
from src.app.ui.setting.page.model_page import ModelPage
from src.app.ui.setting.page.translation_page import TranslationPage
from src.app.ui.setting.page.about_page import AboutPage
from src.app.ui.setting.page.log_page import LogPanel, log_success
from src.app.ui.base.action_button import ActionButton
from src.app.ui.window_data import WindowData
from src.shared.settings import ConfigManager

class _NavItem(ActionButton):
    def __init__(self, text: str, svg: str, page_index: int, parent=None):
        super().__init__(svg_str=svg, text=text, parent=parent,
                         width=WindowData.SettingButtonWidth,
                         height=WindowData.SettingButtonHeight,
                         )
        self.page_index = page_index
        self.setObjectName("nav_item")
        self.setChecked(True)
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class _NavBar(QWidget):
    page_changed = Signal(int)

    NAV_ITEMS = [
        ("Graphics Card", ":/svg/gpu.svg",          0),
        ("Model",         ":/svg/model.svg",        1),
        ("Translation",   ":/svg/translate.svg",    2),
        ("Journal",       ":/svg/log.svg",          3),
        ("About",         ":/svg/about.svg",        4),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nav_bar")
        self._buttons: list[_NavItem] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(4)

        for text, svg, idx in self.NAV_ITEMS:
            btn = _NavItem(text, svg, idx)
            btn.clicked.connect(lambda _, b=btn: self._select(b))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # 默认选中第一项
        self._select(self._buttons[0])

    def _select(self, btn: _NavItem):
        for b in self._buttons:
            b.setChecked(False)
        btn.setChecked(True)
        self.page_changed.emit(btn.page_index)

    def select_index(self, index: int):
        if 0 <= index < len(self._buttons):
            self._select(self._buttons[index])


class _BottomBar(QWidget):
    save_clicked = Signal()
    back_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("bottom_bar")
        self.setFixedHeight(52)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)

        self._dirty_label = QLabel()
        self._dirty_label.setObjectName("dirty_label")
        self._dirty_label.hide()
        layout.addWidget(self._dirty_label)

        layout.addStretch()

        self._save_btn = QPushButton(self.tr("Save"))
        self._save_btn.setObjectName("save_btn")
        self._save_btn.setFixedSize(80, 34)
        self._save_btn.clicked.connect(self.save_clicked)
        layout.addWidget(self._save_btn)

    def set_dirty(self, dirty: bool):
        if dirty:
            self._dirty_label.setText(self.tr("● There are unsaved changes."))
            self._dirty_label.show()
        else:
            self._dirty_label.hide()

class SettingPage(QWidget):
    """
    完整设置页面：
    左侧导航 + 右侧内容区 + 底部操作栏
    """
    back_requested = Signal()
    install_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("setting_page")
        self._config = ConfigManager()
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 主体区（导航 + 内容）
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # 左侧导航
        self._navbar = _NavBar()
        self._navbar.setFixedWidth(WindowData.SettingWidth)
        body.addWidget(self._navbar)

        # 分割线
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setObjectName("nav_separator")
        body.addWidget(sep)

        # 右侧内容 QStackedWidget
        self._stack = QStackedWidget()
        self._stack.setObjectName("setting_stack")
        body.addWidget(self._stack, 1)

        root.addLayout(body, 1)

        # 底部分割线
        bottom_sep = QFrame()
        bottom_sep.setFrameShape(QFrame.Shape.HLine)
        bottom_sep.setObjectName("bottom_separator")
        root.addWidget(bottom_sep)

        # 底部操作栏
        self._bottom_bar = _BottomBar()
        root.addWidget(self._bottom_bar)

        # 注册子页面
        self._register_pages()

    def _register_pages(self):
        """按导航顺序注册子页面"""
        self._gpu_page         = GpuPage()
        self._model_page       = ModelPage()
        self._translation_page = TranslationPage()
        self._log_panel        = LogPanel()
        self._about_page       = AboutPage()

        for page in [
            self._gpu_page,
            self._model_page,
            self._translation_page,
            self._log_panel,
            self._about_page,
        ]:
            self._stack.addWidget(page)

    def _connect_signals(self):
        self._navbar.page_changed.connect(self._stack.setCurrentIndex)
        self._bottom_bar.save_clicked.connect(self._on_save)
        self._gpu_page.install.connect(self.install_requested.emit)

        # 监听 ConfigManager dirty 变化
        # 各子页面修改数据后调用 _check_dirty 即可
        self._dirty_timer_id = self.startTimer(500)  # 每 500ms 轮询一次 dirty 状态

    def timerEvent(self, event):
        """轮询 dirty 状态更新底部提示"""
        self._bottom_bar.set_dirty(self._config.is_dirty)

    def _on_save(self):
        # 通知各子页面把控件值写入 ConfigManager
        for i in range(self._stack.count()):
            page = self._stack.widget(i)
            if hasattr(page, "collect"):
                page.collect()

        self._config.save()
        self._bottom_bar.set_dirty(False)
        log_success("设置已保存")

    def _on_back(self):
        if self._config.is_dirty:
            reply = QMessageBox.question(
                self,
                "未保存的更改",
                "有设置尚未保存，是否保存后再退出？",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if reply == QMessageBox.StandardButton.Save:
                self._on_save()
                self.back_requested.emit()
            elif reply == QMessageBox.StandardButton.Discard:
                self._config.load()  # 丢弃改动，重新加载
                self.back_requested.emit()
            # Cancel 什么都不做
        else:
            self.back_requested.emit()

    def navigate_to(self, index: int):
        """外部调用跳转到指定页"""
        self._navbar.select_index(index)