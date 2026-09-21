from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class _TextToImagePanel(BaseParamPanel):
    """只有张数 / 宽高 / 步数 / CFG / 种子的文生图面板；各模型只声明名字和取值范围。

    不带 names，所以基类本身不会注册到 WidgetFactory。
    """
    type = FactoryType.Image
    _TITLE = ""
    # (最小值, 最大值, 默认值)
    _STEPS = (1, 50, 30)
    _GUIDANCE = (0.5, 10.0, 5.0)

    def __init__(self, parent=None):
        super().__init__(title=self._TITLE, parent=parent)

    def _build_widgets(self):
        # 出图张数：占位槽数和后端循环次数都取这个值，下限 1 避免 0 张
        self._number = self._labeled_slider(1, 10, 1)
        self._add_row(self.tr("number"), self._number)

        self._width = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("width"), self._width)

        self._height = self._labeled_slider(256, 2048, 1024, step=64)
        self._add_row(self.tr("height"), self._height)

        self._steps = self._labeled_slider(*self._STEPS)
        self._add_row(self.tr("num_inference_steps"), self._steps)

        self._guidance = self._labeled_slider(*self._GUIDANCE, decimals=1, step=0.5)
        self._add_row(self.tr("guidance_scale"), self._guidance)

        self._seed = self._labeled_slider(0, 2147483647, 0)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "number":              self._number.value(),
            "width":               self._width.value(),
            "height":              self._height.value(),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "seed":                self._seed.value(),
        }


class SD35MediumPanel(_TextToImagePanel):
    names = ["SD3.5-Medium"]
    _TITLE = "SD3.5-Medium 参数"
    _STEPS = (1, 100, 40)
    _GUIDANCE = (0.5, 12.0, 4.5)


class ZImageTurboPanel(_TextToImagePanel):
    names = ["Z-Image-Turbo"]
    _TITLE = "Z-Image-Turbo 参数"
    # Turbo 是蒸馏模型：官方 8 NFE（约 9 步），guidance 必须为 0
    _STEPS = (1, 30, 9)
    _GUIDANCE = (0.0, 5.0, 0.0)


class QwenImageLightningPanel(_TextToImagePanel):
    names = ["Qwen-Image-Lightning"]
    _TITLE = "Qwen-Image-Lightning 参数"
    # Lightning LoRA 是少步蒸馏，后端把 guidance_scale 当作 true_cfg_scale 使用
    _STEPS = (1, 20, 8)
    _GUIDANCE = (1.0, 10.0, 1.0)
