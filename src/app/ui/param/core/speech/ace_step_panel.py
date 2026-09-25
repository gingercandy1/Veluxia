from src.shared.enum_type import FactoryType
from src.app.ui.param.panel_base import BaseParamPanel


class AceStepMusicPanel(BaseParamPanel):
    names = ["Ace-Step1.5"]
    type = FactoryType.Speech

    def __init__(self, parent=None):
        super().__init__(title="Ace-Step panel", parent=parent)

    def _build_widgets(self):
        # 时长（秒）
        self._duration = self._labeled_slider(5, 120, 10)
        self._add_row(self.tr("duration (sec)"), self._duration)

        # 随机种子
        self._seed = self._labeled_slider(0, 2147483647, 42)
        self._add_row(self.tr("seed"), self._seed)

    def get_params(self) -> dict:
        return {
            "duration":   self._duration.value(),
            "seed":       self._seed.value(),
        }
