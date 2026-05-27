from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QGroupBox, QComboBox, QLabel
)

from src.backend.core.translate_pipeline import TranslationPipeline
from src.shared.settings import ConfigManager

class TranslationPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("translation_page")
        self._config = ConfigManager()
        self._build_ui()
        self.load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 标题
        title = QLabel("翻译设置")
        title.setObjectName("page_title")
        layout.addWidget(title)

        # 语言设置分组
        group = QGroupBox("语言配置")
        group.setObjectName("setting_group")
        form = QFormLayout(group)
        form.setSpacing(12)

        # 获取支持的语言列表
        languages = TranslationPipeline.get_supported_languages()
        lang_items = list(languages.items())  # [(code, name), ...]

        # 源语言
        self._source_combo = QComboBox()
        self._source_combo.addItem("自动检测", "auto")
        for code, name in lang_items:
            self._source_combo.addItem(f"{name}（{code}）", code)
        form.addRow("源语言：", self._source_combo)

        # 目标语言
        self._target_combo = QComboBox()
        for code, name in lang_items:
            self._target_combo.addItem(f"{name}（{code}）", code)
        form.addRow("目标语言：", self._target_combo)

        layout.addWidget(group)

        # 引擎说明
        engine_group = QGroupBox("翻译引擎")
        engine_group.setObjectName("setting_group")
        engine_layout = QVBoxLayout(engine_group)

        engine_info = QLabel(
            "主引擎：Google Translate（质量最佳）\n"
            "备用引擎：MyMemory（Google 不可用时自动切换）"
        )
        engine_info.setObjectName("engine_info")
        engine_info.setWordWrap(True)
        engine_layout.addWidget(engine_info)
        layout.addWidget(engine_group)

        layout.addStretch()

        # 监听变化标记 dirty
        self._source_combo.currentIndexChanged.connect(self._on_changed)
        self._target_combo.currentIndexChanged.connect(self._on_changed)

    def _on_changed(self):
        self.collect()

    def load(self):
        source = self._config.get("translation", "source", "auto")
        target = self._config.get("translation", "target", "en")

        idx = self._source_combo.findData(source)
        if idx >= 0:
            self._source_combo.setCurrentIndex(idx)

        idx = self._target_combo.findData(target)
        if idx >= 0:
            self._target_combo.setCurrentIndex(idx)

    def collect(self):
        """把控件值写入 ConfigManager"""
        self._config.set("translation", "source", self._source_combo.currentData())
        self._config.set("translation", "target", self._target_combo.currentData())