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


class QwenImageEditPanel(BaseParamPanel):
    """按指令改图：参考图来自附件，输出尺寸跟随参考图比例，所以不给宽高。"""
    names = ["Qwen-Image-Edit-2509"]
    type = FactoryType.Image

    def __init__(self, parent=None):
        super().__init__(title="Qwen-Image-Edit-2509", parent=parent)

    def _build_widgets(self):
        self._number = self._labeled_slider(1, 10, 1)
        self._add_row(self.tr("number"), self._number)
        # Lightning LoRA 是 4 步蒸馏
        self._steps = self._labeled_slider(1, 20, 4)
        self._add_row(self.tr("num_inference_steps"), self._steps)
        self._seed = self._labeled_slider(0, 2147483647, 0)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "number":              self._number.value(),
            "num_inference_steps": self._steps.value(),
            "seed":                self._seed.value(),
        }
