from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from ui.base.widget import BaseWidget
from ui.param.panel_base import BaseParamPanel


class ParamDrawer(BaseWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("param_drawer")
        self._setup_ui()
        self.param_widget = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # 滚动区域（参数多时可滚动）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setObjectName("param_scroll_area")

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(8)
        self.scroll_area.setWidget(self.content)

        layout.addWidget(self.scroll_area)

    def load_schema(self, widget):
        """根据schema加载控件，旧控件自动清除"""
        # 清除旧控件
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            _item_widget = item.widget()
            if _item_widget:
                _item_widget.deleteLater()

        self.param_widget = widget
        self.content_layout.addWidget(widget)

    def get_params(self):
        if isinstance(self.param_widget, BaseParamPanel):
            return self.param_widget.get_params()
        else:
            return {}


