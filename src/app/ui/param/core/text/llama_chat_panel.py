from PySide6.QtWidgets import QCheckBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class LlamaChatPanel(BaseParamPanel):
    dynamic = True
    type = FactoryType.Text

    def __init__(self, parent=None):
        super().__init__(title="Llama panel", parent=parent)

    def _build_widgets(self):
        # 最大输出 token
        self._max_tokens = self._labeled_slider(128, 8192, 4096, step=128)
        self._add_row("Max Token", self._max_tokens)

        # 上下文长度
        self._n_ctx = self._labeled_slider(512, 16384, 4096, step=512)
        self._add_row("Context Length", self._n_ctx)

        # Temperature（手动覆盖，0 = 由意图识别自动决定）
        self._temperature = self._labeled_slider(0.0, 1.0, 0.0, decimals=2, step=0.05)
        self._add_row("Temperature", self._temperature)

        # Top-p
        self._top_p = self._labeled_slider(0.1, 1.0, 0.9, decimals=2, step=0.05)
        self._add_row("Top-p", self._top_p)

        # Repeat penalty
        self._repeat_penalty = self._labeled_slider(1.0, 1.5, 1.05, decimals=2, step=0.01)
        self._add_row("Repeat Penalty", self._repeat_penalty)

    def get_params(self) -> dict:
        return {
            "max_tokens":     self._max_tokens.value(),
            "temperature":    self._temperature.value(),
            "top_p":          self._top_p.value(),
            "repeat_penalty": self._repeat_penalty.value(),
        }
