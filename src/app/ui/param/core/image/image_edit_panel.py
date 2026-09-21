from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class BgRemovalPanel(BaseParamPanel):
    """去背景没有可调参数：输入图来自附件，面板只是让模型切换时有界面可载入。"""
    names = ["rembg-u2net", "rembg-birefnet"]
    type = FactoryType.Image

    def __init__(self, parent=None):
        super().__init__(title="background removal", parent=parent)

    def _build_widgets(self):
        pass

    def get_params(self) -> dict:
        return {}


class UpscalePanel(BaseParamPanel):
    names = ["RealESRGAN-anime-6B", "RealESRGAN-x4plus"]
    type = FactoryType.Image

    def __init__(self, parent=None):
        super().__init__(title="upscale", parent=parent)

    def _build_widgets(self):
        # 0 = 保持模型原生 4 倍；其它值会在放大后再缩放到目标倍率
        self._outscale = self._labeled_slider(0, 4, 0, decimals=1, step=0.5)
        self._add_row(self.tr("outscale (0 = native x4)"), self._outscale)

    def get_params(self) -> dict:
        return {"outscale": self._outscale.value()}
