from PySide6.QtWidgets import QPlainTextEdit

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class StableAudioOpenPanel(BaseParamPanel):
    names = ["Stable-Audio-Open-1.0"]
    type = FactoryType.Speech

    def __init__(self, parent=None):
        super().__init__(title="Stable Audio Open 参数", parent=parent)

    def _build_widgets(self):
        # 时长（秒），模型上限约 47s
        self._duration = self._labeled_slider(1, 47, 5, decimals=1, step=0.5)
        self._add_row(self.tr("duration (sec)"), self._duration)

        self._steps = self._labeled_slider(1, 100, 8)
        self._add_row(self.tr("num_inference_steps"), self._steps)

        self._negative_prompt = QPlainTextEdit()
        self._negative_prompt.setFixedHeight(60)
        self._add_row(self.tr("negative prompt"), self._negative_prompt)

        self._seed = self._labeled_slider(0, 2147483647, 0)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "duration":            self._duration.value(),
            "num_inference_steps": self._steps.value(),
            "negative_prompt":     self._negative_prompt.toPlainText().strip(),
            "seed":                self._seed.value(),
        }
