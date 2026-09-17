from PySide6.QtWidgets import QComboBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class LTX2VideoPanel(BaseParamPanel):
    name = ["LTX-2.3", "LTX-2.5"]
    type = FactoryType.Animation

    _RESOLUTIONS = ["low (512×320)", "medium (768×512)", "high (960×640)"]
    _RESOLUTION_MAP = {"low (512×320)": "low", "medium (768×512)": "medium", "high (960×640)": "high"}

    # 合法帧数列表（8N+1）
    _VALID_FRAMES = [25, 33, 41, 49, 57, 65, 73, 81, 97, 121]

    def __init__(self, parent=None):
        super().__init__(title="LTX-2 panel", parent=parent)

    def _build_widgets(self):
        # 分辨率（LTX-2 系列显存占用远高于旧版 LTX-Video，8GB 显卡建议保持 low）
        self._resolution = QComboBox()
        self._resolution.addItems(self._RESOLUTIONS)
        self._resolution.setCurrentText("low (512×320)")
        self._resolution.setToolTip(
            "LTX-2.3/2.5 显存占用较大：8GB 显卡建议 low，12GB 可尝试 medium，high 建议 16GB+。"
        )
        self._add_row("resolution", self._resolution)

        # 帧数（从合法列表里选，帧数越多显存/耗时越高）
        self._num_frames = QComboBox()
        self._num_frames.addItems([str(f) for f in self._VALID_FRAMES])
        self._num_frames.setCurrentText("81")
        self._add_row("num_frames", self._num_frames)

        # 帧率
        self._frame_rate = self._labeled_slider(8, 30, 24, decimals=0, step=1)
        self._add_row("frame_rate", self._frame_rate)

        # 推理步数
        self._steps = self._labeled_slider(1, 50, 30)
        self._add_row("num_inference_steps", self._steps)

        # CFG（推荐 3.0）
        self._guidance = self._labeled_slider(0.5, 10.0, 3.0, decimals=1, step=0.1)
        self._add_row("guidance_scale", self._guidance)

        # 随机种子
        self._seed = self._labeled_slider(0, 2147483647, 42)
        self._add_row("seed", self._seed)

    def get_params(self) -> dict:
        return {
            "resolution":          self._RESOLUTION_MAP[self._resolution.currentText()],
            "num_frames":          int(self._num_frames.currentText()),
            "frame_rate":          float(self._frame_rate.value()),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "seed":                self._seed.value(),
        }
