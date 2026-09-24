from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QLabel
)

from src.app.i18n import DEFAULT_LANGUAGE, LANGUAGES
from src.shared.settings import ConfigManager


class GeneralPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("general_page")
        self._config = ConfigManager()
        self._build_ui()
        self.load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel(self.tr("General Settings"))
        title.setObjectName("page_title")
        layout.addWidget(title)

        group = QGroupBox(self.tr("Interface"))
        group.setObjectName("setting_group")
        form = QFormLayout(group)
        form.setSpacing(12)

        self._language_combo = QComboBox()
        for code, (_locale, native_name) in LANGUAGES.items():
            self._language_combo.addItem(native_name, code)
        form.addRow(self.tr("Language:"), self._language_combo)

        # 各控件的文案在构造时就定了，运行中切换要重建整个界面，所以重启生效
        hint = QLabel(self.tr("The new language takes effect after restarting the app."))
        hint.setObjectName("page_hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        layout.addWidget(group)
        layout.addStretch()

        self._language_combo.currentIndexChanged.connect(self.collect)

    def load(self):
        code = self._config.get("general", "language", DEFAULT_LANGUAGE)
        index = self._language_combo.findData(code)
        if index < 0:
            index = self._language_combo.findData(DEFAULT_LANGUAGE)
        self._language_combo.setCurrentIndex(index)

    def collect(self):
        """把控件值写入 ConfigManager"""
        self._config.set("general", "language", self._language_combo.currentData())
