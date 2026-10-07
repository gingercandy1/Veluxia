from PySide6.QtWidgets import QComboBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel, LabeledSlider


class Wan2VideoPanel(BaseParamPanel):
    names = ["Wan2.2-TI2V", "Wan2.2-TI2V-Turbo", "Wan2.1-I2V-14B"]
    type = FactoryType.Animation

    _RESOLUTIONS = ["low (320×320)", "medium (480×480)", "high (704×704)"]
    _RESOLUTION_MAP = {"low (320×320)": "low", "medium (480×480)": "medium",
                       "high (704×704)": "high"}

    # 合法帧数：4N+1
    _VALID_FRAMES = [17, 25, 33, 49, 65, 81]

    # 步数蒸馏 LoRA 的模型要 4 步、CFG 1；沿用 25 步 / CFG 5 会把蒸馏效果反过来。
    # 用户保存过的参数在 bind_model 之后恢复，会盖过这里的默认值
    _DISTILLED = {"Wan2.1-I2V-14B": (4, 1.0)}

    def __init__(self, parent=None):
        super().__init__(title="Wan2.2-TI2V panel", parent=parent)

    def bind_model(self, model_name: str):
        super().bind_model(model_name)
        if model_name in self._DISTILLED:
            steps, guidance = self._DISTILLED[model_name]
            self._steps.setValue(steps)
            self._guidance.setValue(guidance)

    def _build_widgets(self):
        # 分辨率（后端按档位取宽高）
        self._resolution = QComboBox()
        self._resolution.addItems(self._RESOLUTIONS)
        self._add_row(self.tr("resolution"), self._resolution)

        # 帧数（从合法列表里选）
        self._num_frames = QComboBox()
        self._num_frames.addItems([str(f) for f in self._VALID_FRAMES])
        self._num_frames.setCurrentText("25")
        self._add_row(self.tr("num_frames"), self._num_frames)

        # 推理步数（推荐 20-30）
        self._steps = self._labeled_slider(1, 50, 25)
        self._add_row(self.tr("num_inference_steps"), self._steps)

        # CFG（推荐 3.0-7.0，越高运动越明显）
        self._guidance = self._labeled_slider(1.0, 15.0, 5.0, decimals=1, step=0.5)
        self._add_row(self.tr("guidance_scale"), self._guidance)

        # 主体缩小的比例，头顶留空白；起跳、伸展这类会出框的动作需要（0 = 不留白）
        self._headroom = self._labeled_slider(0.0, 0.6, 0.0, decimals=2, step=0.05)
        self._add_row(self.tr("headroom"), self._headroom)

        # 随机种子
        self._seed = self._labeled_slider(0, 2147483647, 42)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "resolution":          self._RESOLUTION_MAP[self._resolution.currentText()],
            "num_frames":          int(self._num_frames.currentText()),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "headroom":            self._headroom.value(),
            "seed":                self._seed.value(),
        }
