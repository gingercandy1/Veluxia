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
        self._add_row("duration (sec)", self._duration)

        # 批次大小
        self._batch_size = self._labeled_slider(1, 8, 4)
        self._add_row("batch_size", self._batch_size)

        # 随机种子
        self._seed = self._labeled_slider(0, 2147483647, 42)
        self._add_row("seed", self._seed)

    def get_params(self) -> dict:
        return {
            "duration":   self._duration.value(),
            "batch_size": self._batch_size.value(),
            "seed":       self._seed.value(),
        }
