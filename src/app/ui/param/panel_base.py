import enum
import json
import os
from abc import abstractmethod
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel,
    QSlider, QSpinBox, QDoubleSpinBox, QFormLayout, QSizePolicy,
)

from src.shared.enum_type import FactoryType
from src.shared.settings import PROJECT_ROOT
from ui.base.widget import BaseWidget
from ui.param.param_factory import WidgetFactory


class LabeledSlider(QWidget):
    """滑块 + 数值联动控件"""

    def __init__(
        self,
        minimum: float,
        maximum: float,
        default: float,
        decimals: int = 0,
        step: float = 1,
        parent=None,
    ):
        super().__init__(parent)
        self._decimals = decimals
        self._factor = 10 ** decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(int(minimum * self._factor))
        self._slider.setMaximum(int(maximum * self._factor))
        self._slider.setSingleStep(int(step * self._factor))

        if decimals == 0:
            self._spin = QSpinBox()
            self._spin.setMinimum(int(minimum))
            self._spin.setMaximum(int(maximum))
            self._spin.setSingleStep(int(step))
            self._spin.setFixedWidth(90)
        else:
            self._spin = QDoubleSpinBox()
            self._spin.setMinimum(minimum)
            self._spin.setMaximum(maximum)
            self._spin.setSingleStep(step)
            self._spin.setDecimals(decimals)
            self._spin.setFixedWidth(90)

        self.setValue(default)

        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

    def _on_slider(self, v: int):
        real = v / self._factor
        self._spin.blockSignals(True)
        self._spin.setValue(real if self._decimals else v)
        self._spin.blockSignals(False)

    def _on_spin(self, v):
        self._slider.blockSignals(True)
        self._slider.setValue(int(v * self._factor))
        self._slider.blockSignals(False)

    def value(self):
        return self._spin.value()

    def setValue(self, v):
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setValue(int(v * self._factor))
        if self._decimals:
            self._spin.setValue(float(v))
        else:
            self._spin.setValue(int(v))
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)


class BaseParamPanel(BaseWidget):
    """
    参数面板基类。
    子类实现 _build_widgets() 和 get_params()。
    """
    names: list = []
    type: enum.Enum = None
    dynamic: bool = False
    config: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        try:
            if not cls.config:
                config_path = os.path.join(PROJECT_ROOT, "models.json")
                with open(config_path, "r") as f:
                    config = json.load(f)
                cls._config = config
        except Exception as e:
            print("Models read error:", e)
            return

        #静态名称
        if cls.names and not cls.dynamic:
            for name in cls.names:
                WidgetFactory.register_widget(cls.type, name, cls)

        # 动态读取名称
        if cls.dynamic and cls.type is not None:
            cls._register_dynamic()

    @classmethod
    def _register_dynamic(cls):
        """从配置文件动态注册所有模型名"""
        type_id = FactoryType.convert_to_text(cls.type)
        type_dict = cls._config.get(type_id, {})
        for name in type_dict.keys():
            WidgetFactory.register_widget(cls.type, name, cls)
        cls.names = list(type_dict.keys())

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("base_param_panel")
        self._title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(20, 0, 20, 0)
        self._form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._form.setSpacing(15)
        self._build_widgets()

    def _add_row(self, label: str, widget: QWidget):
        self._form.addRow(QLabel(label), widget)

    def _labeled_slider(self, mn, mx, default, decimals=0, step=1) -> LabeledSlider:
        return LabeledSlider(mn, mx, default, decimals, step)

    @abstractmethod
    def _build_widgets(self): ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]: ...
