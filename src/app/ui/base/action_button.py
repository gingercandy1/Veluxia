import sys

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Property, QSize, QRect
from PySide6.QtGui import (
    QPainter, QColor, QIcon, QPixmap, QPainterPath,
    QBrush, QFont
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QPushButton, QApplication, QWidget, QHBoxLayout, QAbstractButton


class ActionButton(QPushButton):
    """自定义 SVG 图标按钮 - 使用 paintEvent 绘制，更灵活且性能更好"""

    def __init__(self,
                 svg_str: str,
                 text: str = "",
                 tooltip: str = "",
                 width=26, height=26,
                 icon_size_width=18,
                 icon_size_height=18,
                 is_circle=False,
                 parent=None):
        super().__init__(parent)

        self._svg_str = svg_str
        self._text = text
        self.is_circle = is_circle

        # 颜色定义
        self._normal_bg = QColor(40, 40, 48, 0)
        self._hover_bg  = QColor(30, 30, 30, 240)
        self._press_bg  = QColor(40, 40, 40, 255)
        self._border_color = QColor(80, 80, 90, 180)
        self._text_color = QColor(176, 176, 184)

        self._current_bg = self._normal_bg

        self.setFixedSize(width, height)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(tooltip)

        # 图标缓存
        self._icon = self._svg_to_icon(svg_str)
        self._icon_size = QSize(icon_size_width, icon_size_height)

        # 背景动画
        self._bg_anim = QPropertyAnimation(self, b"bgColor")
        self._bg_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._bg_anim.setDuration(180)

        if self._text:
            if width > height:
                icon_y = (self.height() - self._icon_size.height()) // 2
                icon_x = icon_y
            else:
                icon_x = (self.width() - self._icon_size.width()) // 2
                icon_y = icon_x

            self.icon_rect = QRect(
                icon_x, icon_y,
                self._icon_size.width(),
                self._icon_size.height()
            )
            self.text_rect = self.rect().adjusted(self._icon_size.width() + icon_x + 9, 0, -8, 0)  # 留出图标空间
        else:
            icon_x = (self.width() - self._icon_size.width()) // 2
            icon_y = (self.height() - self._icon_size.height()) // 2
            self.icon_rect = QRect(
                icon_x, icon_y,
                self._icon_size.width(),
                self._icon_size.height()
            )

    def _get_bg(self) -> QColor:
        return self._current_bg

    def _set_bg(self, color: QColor):
        self._current_bg = color
        self.update()

    bgColor = Property(QColor, _get_bg, _set_bg)

    def _svg_to_icon(self, svg_path: str, size: int = 64) -> QIcon:
        renderer = QSvgRenderer(svg_path)
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter)
        painter.end()

        return QIcon(pixmap)

    def set_color(self, background_color, hover_color):
        self._normal_bg = background_color
        self._hover_bg = hover_color
        self._current_bg = background_color
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        if self.is_circle:
            # 1. 绘制圆角背景
            path = QPainterPath()
            path.addRoundedRect(rect.adjusted(1, 1, -1, -1), 50, 50)
            painter.fillPath(path, QBrush(self._current_bg))
        else:
            # 1. 绘制圆角背景
            path = QPainterPath()
            path.addRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)
            painter.fillPath(path, QBrush(self._current_bg))

        # 2. 绘制边框
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        # 3. 绘制图标和文字
        if self._icon:
            self._icon.paint(painter, self.icon_rect)

        # 4. 绘制文字
        if self._text:
            painter.setPen(self._text_color)
            font = painter.font()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(self.text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._text)

    def enterEvent(self, event):
        self._animate_bg(self._hover_bg, 160)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_bg(self._normal_bg, 200)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._animate_bg(self._press_bg, 80)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.underMouse():
            self._animate_bg(self._hover_bg, 120)
        else:
            self._animate_bg(self._normal_bg, 120)
        super().mouseReleaseEvent(event)

    def _animate_bg(self, target_color: QColor, duration: int = 180):
        self._bg_anim.stop()
        self._bg_anim.setDuration(duration)
        self._bg_anim.setStartValue(self._current_bg)
        self._bg_anim.setEndValue(target_color)
        self._bg_anim.start()



class ExpandButton(QAbstractButton):
    """
    仿 Claude 风格的展开/收起按钮。
    - 圆角胶囊形，高级灰底色
    - hover 时背景渐亮（150ms ease-out）
    - press 时短暂压暗（80ms）
    """

    _BG_NORMAL = QColor("#2a2b32")
    _BG_HOVER = QColor("#3d3e47")
    _BG_PRESS = QColor("#1e1f24")
    _FG = QColor("#9b9ba4")
    _FG_HOVER = QColor("#d1d1d8")
    _RADIUS = 20

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._bg_color = QColor(self._BG_NORMAL)
        self._fg_color = QColor(self._FG)

        self.setFixedHeight(36)
        self.setMinimumWidth(130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

        font = QFont()
        font.setFamilies(["Noto Sans SC", "Microsoft YaHei", "sans-serif"])
        font.setPointSize(12)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.4)
        self.setFont(font)

        self._bg_anim = QPropertyAnimation(self, b"bgColor")
        self._bg_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._fg_anim = QPropertyAnimation(self, b"fgColor")
        self._fg_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ── Qt Property ───────────────────────────────────────────────────────────
    def _get_bg(self) -> QColor: return self._bg_color
    def _set_bg(self, c: QColor): self._bg_color = c; self.update()
    def _get_fg(self) -> QColor: return self._fg_color
    def _set_fg(self, c: QColor): self._fg_color = c; self.update()
    bgColor = Property(QColor, _get_bg, _set_bg)
    fgColor = Property(QColor, _get_fg, _set_fg)

    # ── 动画触发 ──────────────────────────────────────────────────────────────
    def _animate(self, bg: QColor, fg: QColor, ms: int):
        for anim, cur, target in (
                (self._bg_anim, self._bg_color, bg),
                (self._fg_anim, self._fg_color, fg),
        ):
            anim.stop()
            anim.setDuration(ms)
            anim.setStartValue(QColor(cur))
            anim.setEndValue(target)
            anim.start()

    def enterEvent(self, e):
        self._animate(self._BG_HOVER, self._FG_HOVER, 150)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._animate(self._BG_NORMAL, self._FG, 200)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        self._animate(self._BG_PRESS, self._FG, 80)
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        self._animate(self._BG_HOVER, self._FG_HOVER, 120)
        super().mouseReleaseEvent(e)

    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()

        path = QPainterPath()
        path.addRoundedRect(0, 0, w, h, self._RADIUS, self._RADIUS)
        p.fillPath(path, self._bg_color)

        p.setPen(self._fg_color)
        p.setFont(self.font())
        text = ("收起  ▲" if self._expanded else "查看更多  ▼")
        fm = p.fontMetrics()
        x = (w - fm.horizontalAdvance(text)) // 2
        y = (h + fm.ascent() - fm.descent()) // 2
        p.drawText(x, y, text)
        p.end()



if __name__ == '__main__':
    def _copy_svg() -> str:
        return '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#b0b0b8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>'''



    app = QApplication(sys.argv)

    w = QWidget()
    button = ActionButton(svg_str=_copy_svg())


    l = QHBoxLayout(w)
    l.addWidget(button)
    w.resize(100, 100)

    w.show()


    app.exec()

