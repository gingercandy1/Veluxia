from PySide6.QtWidgets import QCheckBox, QComboBox

from src.app.stt_languages import fill_language_combo
from src.app.ui.param.panel_base import BaseParamPanel
from src.shared.enum_type import FactoryType


class TranscriptionPanel(BaseParamPanel):
    """上传音频转写的参数。麦克风口述不走这里，它的语言在设置页里配。"""
    dynamic = True
    type = FactoryType.Transcription

    def __init__(self, parent=None):
        super().__init__(title="transcription panel", parent=parent)

    def _build_widgets(self):
        # 指定语言比自动检测准，Auto 模型也靠它挑最合适的识别模型
        self._language = QComboBox()
        fill_language_combo(self._language)
        self._add_row(self.tr("language"), self._language)

        self._export_srt = QCheckBox(self.tr("export SRT subtitles"))
        self._export_srt.setChecked(True)
        self._add_row(self.tr("subtitles"), self._export_srt)

    def get_params(self) -> dict:
        return {
            "language": self._language.currentData(),
            "export_srt": self._export_srt.isChecked(),
        }
