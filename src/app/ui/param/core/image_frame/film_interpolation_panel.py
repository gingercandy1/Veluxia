from PySide6.QtWidgets import QComboBox, QCheckBox

from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel, LabeledSlider



class FilmInterpolationPanel(BaseParamPanel):
    names = ["FILM", "Rife"]
    type = FactoryType.ImageFrame

    def __init__(self, parent=None):
        super().__init__(title="interpolation panel", parent=parent)

    def _build_widgets(self):
        # 插值后端
        self._backend = QComboBox()
        self._backend.addItems(self.names)
        self._add_row(self.tr("backend"), self._backend)

        # 插值次数（每次 2^N 倍，times=1 → 3帧，times=2 → 5帧，times=3 → 9帧）
        self._times = self._labeled_slider(1, 5, 1)
        self._add_row(self.tr("times_to_interpolate"), self._times)

        # 输出宽高
        self._width = self._labeled_slider(256, 3840, 1024, step=64)
        self._add_row(self.tr("width"), self._width)

        self._height = self._labeled_slider(256, 2160, 1024, step=64)
        self._add_row(self.tr("height"), self._height)

        # 输出帧率（导出视频时生效）
        self._fps = self._labeled_slider(8, 120, 24)
        self._add_row(self.tr("fps"), self._fps)

        # RIFE scale（仅 rife 后端生效，降低可减少大动作伪影）
        self._rife_scale = self._labeled_slider(0.25, 2.0, 1.0, decimals=2, step=0.25)
        self._add_row(self.tr("RIFE Scale"), self._rife_scale)

        # 是否导出视频
        self._export_video = QCheckBox("export MP4")
        self._export_video.setChecked(True)
        self._add_row(self.tr("export_video"), self._export_video)

    def get_params(self) -> dict:
        return {
            "backend":              self._backend.currentText(),
            "times_to_interpolate": self._times.value(),
            "width":                self._width.value(),
            "height":               self._height.value(),
            "fps":                  self._fps.value(),
            "rife_scale":           self._rife_scale.value(),
            "export_video":         self._export_video.isChecked(),
        }
