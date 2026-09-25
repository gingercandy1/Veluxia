from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QWidget, QFormLayout, QLabel, QSizePolicy

from src.app.ui.param.panel_base import BaseParamPanel
from src.shared.enum_type import FactoryType


class _CustomVoiceWidget(QWidget):
    dynamic = True
    PRESET_VOICES = ["Chelsie", "Ethan", "Serena", "Dylan", "Ana", "Vivian", "Ryan", "Aria", "Marco"]

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(10)
        form.setContentsMargins(0, 4, 0, 0)

        self._voice = QComboBox()
        self._voice.addItems(self.PRESET_VOICES)
        self._voice.setCurrentText("Vivian")
        form.addRow(QLabel(self.tr("voice")), self._voice)

    def get_params(self) -> dict:
        return {"voice": self._voice.currentText()}


class _DesignVoiceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QTextEdit
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(10)
        form.setContentsMargins(0, 4, 0, 0)

        self._voice_prompt = QTextEdit()
        self._voice_prompt.setPlaceholderText(
            self.tr("Example: The voice of a passionate and energetic 20-year-old girl."))
        self._voice_prompt.setFixedHeight(60)
        form.addRow(QLabel(self.tr("voice_prompt")), self._voice_prompt)

    def get_params(self) -> dict:
        return {"voice_prompt": self._voice_prompt.toPlainText().strip()}


class _CloneVoiceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLineEdit, QPushButton, QHBoxLayout, QTextEdit
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(10)
        form.setContentsMargins(0, 4, 0, 0)

        # 参考音频路径
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self._audio_path = QLineEdit()
        self._audio_path.setPlaceholderText(self.tr("reference_audio_path"))
        browse = QPushButton("...")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        h.addWidget(self._audio_path)
        h.addWidget(browse)
        form.addRow(QLabel(self.tr("reference_audio")), row)

        # 参考文本
        self._ref_text = QTextEdit()
        self._ref_text.setPlaceholderText(
            self.tr("Refer to the text content corresponding to the audio."))
        self._ref_text.setFixedHeight(52)
        form.addRow(QLabel(self.tr("reference_text")), self._ref_text)

    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("choose audio"), "", self.tr("audio file") + " (*.wav *.mp3 *.flac)")
        if path:
            self._audio_path.setText(path)

    def get_params(self) -> dict:
        return {
            "reference_audio_path": self._audio_path.text().strip(),
            "reference_text":       self._ref_text.toPlainText().strip(),
        }


class Qwen3TTSPanel(BaseParamPanel):
    # 不用 dynamic：它会把 speech 下所有模型（含 Ace-Step1.5）都注册成本面板
    names = ["Qwen3-TTS-0.6b-custom", "Qwen3-TTS-0.6b-base", "Qwen3-TTS-1.7b-custom",
             "Qwen3-TTS-1.7b-base", "Qwen3-TTS-1.7b-design"]
    type = FactoryType.Speech
    _MODES = ["custom", "design", "clone"]
    _LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish"]

    def __init__(self, parent=None):
        super().__init__(title="Qwen3-TTS panel", parent=parent)

    def _build_widgets(self):
        # 语言
        self._language = QComboBox()
        self._language.addItems(self._LANGUAGES)
        self._add_row(self.tr("language"), self._language)

        # 各模式参数区：按 _MODES 顺序各占一行，bind_model 时只显示当前模式那行。
        # 不用 QStackedWidget：它按最高的一页撑高度，短的页会上下留空
        self._custom_w = _CustomVoiceWidget()
        self._design_w = _DesignVoiceWidget()
        self._clone_w  = _CloneVoiceWidget()
        self._mode_widgets = [self._custom_w, self._design_w, self._clone_w]
        for widget in self._mode_widgets:
            # 面板被撑高时别让子页吃掉多余高度，否则输入框会被推到页中间
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self._form.addRow(widget)
        self._mode = "custom"
        self._show_mode(self._mode)

    def _show_mode(self, mode: str):
        for m, widget in zip(self._MODES, self._mode_widgets):
            self._form.setRowVisible(widget, m == mode)
        self._align_label_columns(self._mode_widgets[self._MODES.index(mode)])

    def _align_label_columns(self, page: QWidget):
        # 子页有自己的 QFormLayout，标签列宽各算各的，输入框就对不齐；统一成最宽的标签
        outer_labels = [self._form.itemAt(row, QFormLayout.LabelRole).widget()
                        for row in range(self._form.rowCount())
                        if self._form.itemAt(row, QFormLayout.LabelRole) is not None]
        labels = outer_labels + page.findChildren(QLabel)
        width = max(label.sizeHint().width() for label in labels)
        for label in labels:
            label.setMinimumWidth(width)
            # 标签被拉宽后文字默认靠左，要跟其他面板一样靠右贴着输入框
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def bind_model(self, model_name: str):
        super().bind_model(model_name)
        self._mode = self._config.get("speech", {}).get(model_name, {}).get("mode", "custom")
        self._show_mode(self._mode)

    def get_params(self) -> dict:
        sub_params = self._mode_widgets[self._MODES.index(self._mode)].get_params()
        return {
            "mode":     self._mode,
            "language": self._language.currentText(),
            **sub_params,
        }
