from PySide6.QtWidgets import QComboBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class LTXVideoPanel(BaseParamPanel):
    name = ["LTX-Video"]
    type = FactoryType.Animation

    _RESOLUTIONS = ["low (512×320)", "medium (608×384)", "high (704×480)"]
    _RESOLUTION_MAP = {"low (512×320)": "low", "medium (608×384)": "medium", "high (704×480)": "high"}

    # 合法帧数列表
    _VALID_FRAMES = [25, 33, 41, 49, 57, 65, 73, 81, 97, 121]

    def __init__(self, parent=None):
        super().__init__(title="LTX-Video panel", parent=parent)

    def _build_widgets(self):
        # 分辨率
        self._resolution = QComboBox()
        self._resolution.addItems(self._RESOLUTIONS)
        self._add_row("resolution", self._resolution)

        # 帧数（从合法列表里选）
        self._num_frames = QComboBox()
        self._num_frames.addItems([str(f) for f in self._VALID_FRAMES])
        self._num_frames.setCurrentText("121")
        self._add_row("num_frames", self._num_frames)

        # 推理步数（distilled 推荐 4-8）
        self._steps = self._labeled_slider(1, 30, 8)
        self._add_row("num_inference_steps", self._steps)

        # CFG（distilled 推荐 1.0）
        self._guidance = self._labeled_slider(0.5, 10.0, 1.0, decimals=1, step=0.1)
        self._add_row("guidance_scale", self._guidance)

        # VAE 解码时步
        self._decode_timestep = self._labeled_slider(0.01, 0.2, 0.05, decimals=3, step=0.005)
        self._add_row("decode_timestep", self._decode_timestep)

        # VAE 解码噪声
        self._decode_noise = self._labeled_slider(0.001, 0.1, 0.025, decimals=3, step=0.001)
        self._add_row("decode_noise_scale", self._decode_noise)

        # 随机种子
        self._seed = self._labeled_slider(0, 2147483647, 42)
        self._add_row("seed", self._seed)

    def get_params(self) -> dict:
        return {
            "resolution":          self._RESOLUTION_MAP[self._resolution.currentText()],
            "num_frames":          int(self._num_frames.currentText()),
            "num_inference_steps": self._steps.value(),
            "guidance_scale":      self._guidance.value(),
            "decode_timestep":     self._decode_timestep.value(),
            "decode_noise_scale":  self._decode_noise.value(),
            "seed":                self._seed.value(),
        }
