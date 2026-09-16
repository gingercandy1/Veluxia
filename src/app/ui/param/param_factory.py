from src.shared.enum_type import FactoryType


class WidgetFactory:
    """生成器工厂，方便后续扩展模型"""
    _widget = {
        FactoryType.Text: {},
        FactoryType.Image: {},
        FactoryType.ImageFrame: {},
        FactoryType.Animation: {},
        FactoryType.Speech: {},
    }

    @classmethod
    def register_widget(cls, ty, name: str, widget_cls):
        cls._widget[ty].update({name: widget_cls})

    @classmethod
    def build_widget(cls, ty, name: str) :
        if name not in cls._widget.get(ty):
            raise ValueError(f"未知的界面: {name}")
        return cls._widget[ty][name]()

