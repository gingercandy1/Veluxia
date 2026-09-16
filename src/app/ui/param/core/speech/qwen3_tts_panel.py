from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QStackedWidget, QWidget, QFormLayout, QLabel

from src.app.ui.param.panel_base import BaseParamPanel


class _CustomVoiceWidget(QWidget):
    dynamic = True
    PRESET_VOICES = ["Chelsie", "Ethan", "Serena", "Dylan", "Ana", "Vivian", "Ryan", "Aria", "Marco"]

    def __init__(self, parent=None):
        super().__init__(parent)
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(6)
        form.setContentsMargins(0, 4, 0, 0)

        self._voice = QComboBox()
        self._voice.addItems(self.PRESET_VOICES)
        self._voice.setCurrentText("Vivian")
        form.addRow(QLabel("voice"), self._voice)

    def get_params(self) -> dict:
        return {"voice": self._voice.currentText()}


class _DesignVoiceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QTextEdit
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(6)
        form.setContentsMargins(0, 4, 0, 0)

        self._voice_prompt = QTextEdit()
        self._voice_prompt.setPlaceholderText("Example: The voice of a passionate and energetic 20-year-old girl.")
        self._voice_prompt.setFixedHeight(60)
        form.addRow(QLabel("voice_prompt"), self._voice_prompt)

    def get_params(self) -> dict:
        return {"voice_prompt": self._voice_prompt.toPlainText().strip()}


class _CloneVoiceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QLineEdit, QPushButton, QHBoxLayout, QTextEdit
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setSpacing(6)
        form.setContentsMargins(0, 4, 0, 0)

        # 参考音频路径
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        self._audio_path = QLineEdit()
        self._audio_path.setPlaceholderText("reference_audio_path")
        browse = QPushButton("...")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse)
        h.addWidget(self._audio_path)
        h.addWidget(browse)
        form.addRow(QLabel("参考音频"), row)

        # 参考文本
        self._ref_text = QTextEdit()
        self._ref_text.setPlaceholderText("Refer to the text content corresponding to the audio.")
        self._ref_text.setFixedHeight(52)
        form.addRow(QLabel("reference_text"), self._ref_text)

    def _browse(self):
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "choose audio", "", "audio file (*.wav *.mp3 *.flac)")
        if path:
            self._audio_path.setText(path)

    def get_params(self) -> dict:
        return {
            "reference_audio_path": self._audio_path.text().strip(),
            "reference_text":       self._ref_text.toPlainText().strip(),
        }


class Qwen3TTSPanel(BaseParamPanel):
    _MODES = ["custom", "design", "clone"]
    _LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish"]

    def __init__(self, parent=None):
        super().__init__(title="Qwen3-TTS panel", parent=parent)

    def _build_widgets(self):
        # 合成模式
        self._mode = QComboBox()
        self._mode.addItems(self._MODES)
        self._add_row("mode", self._mode)

        # 语言
        self._language = QComboBox()
        self._language.addItems(self._LANGUAGES)
        self._add_row("language", self._language)

        # 各模式参数区（stack 切换）
        self._stack = QStackedWidget()
        self._custom_w = _CustomVoiceWidget()
        self._design_w = _DesignVoiceWidget()
        self._clone_w  = _CloneVoiceWidget()
        self._stack.addWidget(self._custom_w)   # index 0 → custom
        self._stack.addWidget(self._design_w)   # index 1 → design
        self._stack.addWidget(self._clone_w)    # index 2 → clone
        self._form.addRow(self._stack)

        self._mode.currentIndexChanged.connect(self._stack.setCurrentIndex)

    def get_params(self) -> dict:
        mode = self._mode.currentText()
        idx  = self._MODES.index(mode)
        sub_params = [self._custom_w, self._design_w, self._clone_w][idx].get_params()
        return {
            "mode":     mode,
            "language": self._language.currentText(),
            **sub_params,
        }
