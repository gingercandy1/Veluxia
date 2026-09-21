from PySide6.QtWidgets import QComboBox, QPlainTextEdit

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class SDXLPanel(BaseParamPanel):
    names = ["SDXL"]
    type = FactoryType.Image
    _TILE_MODES = ["off", "both", "horizontal", "vertical"]
    _DEFAULT_NEGATIVE_PROMPT = (
        "blurry, low quality, low detail, noisy, cluttered, jpeg artifacts, "
        "deformed, watermark, text, signature")

    def __init__(self, parent=None):
        super().__init__(title="SDXL 参数", parent=parent)

    def _build_widgets(self):
        self._number = self._labeled_slider(1, 10, 1)
        self._add_row(self.tr("number"), self._number)

        self._width = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("width"), self._width)

        self._height = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("height"), self._height)

        self._steps = self._labeled_slider(1, 100, 30)
        self._add_row(self.tr("num_inference_steps"), self._steps)

        self._guidance = self._labeled_slider(0.5, 20.0, 5.0, decimals=1, step=0.5)
        self._add_row(self.tr("guidance_scale"), self._guidance)

        self._negative_prompt = QPlainTextEdit(self._DEFAULT_NEGATIVE_PROMPT)
        self._negative_prompt.setFixedHeight(72)
        self._add_row(self.tr("negative prompt"), self._negative_prompt)

        self._tile_mode = QComboBox()
        for mode, label in zip(self._TILE_MODES, (
                self.tr("off"), self.tr("both axes"), self.tr("horizontal"), self.tr("vertical"))):
            self._tile_mode.addItem(label, mode)
        self._add_row(self.tr("seamless tiling"), self._tile_mode)

        self._seed = self._labeled_slider(0, 2147483647, 0)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "number":              self._number.value(),
            "width":               self._width.value(),
            "height":              self._height.value(),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "negative_prompt":     self._negative_prompt.toPlainText().strip(),
            "tile_mode":           self._tile_mode.currentData(),
            "seed":                self._seed.value(),
        }
