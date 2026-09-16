from PySide6.QtWidgets import QComboBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class FluxSchnellPanel(BaseParamPanel):
    names = ["Flux.1-schnell"]
    type = FactoryType.Image
    _PREPROCESS_MODES = ["resize", "crop", "pad"]

    def __init__(self, parent=None):
        super().__init__(title="Flux.1-schnell 参数", parent=parent)

    def _build_widgets(self):
        # 宽度
        self._number = self._labeled_slider(0, 10, 2, step=64)
        self._add_row(self.tr("number"), self._number)

        # 宽度
        self._width = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("width"), self._width)

        # 高度
        self._height = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("height"), self._height)

        # 推理步数（schnell 推荐 4 步）
        self._steps = self._labeled_slider(1, 20, 4)
        self._add_row(self.tr("num_inference_steps"), self._steps)

        # CFG（schnell 推荐 1.0，无需调高）
        self._guidance = self._labeled_slider(0.5, 10.0, 1.0, decimals=1, step=0.5)
        self._add_row(self.tr("guidance_scale"), self._guidance)

        # 参考图预处理模式（图生图时生效）
        self._preprocess = QComboBox()
        self._preprocess.addItems(self._PREPROCESS_MODES)
        self._add_row(self.tr("preprocess"), self._preprocess)

        # 随机种子（每次加随机偏移，保证多样性）
        self._seed = self._labeled_slider(0, 2147483647, 0)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "number":              self._number.value(),
            "width":               self._width.value(),
            "height":              self._height.value(),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "preprocess":          self._preprocess.currentText(),
            "seed":                self._seed.value(),
        }