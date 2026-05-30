from pathlib import Path

from PySide6.QtCore import QUrl, Qt
from PySide6.QtGui import QPixmap, QDesktopServices, QPainterPath, QColor, QPainter
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSizePolicy,
)

from src.app.ui.base.widget import BaseWidget


class ImageWidget(QWidget):
    """圆角图片，QPainter 绘制，带投影和悬停高光。"""
    MAX_WIDTH = 280
    MAX_HEIGHT = 180
    RADIUS = 10
    SHADOW_CLR = QColor(0, 0, 0, 60)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("image_widget")
        self._pixmap = None
        self._error = ""
        self._load(path)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover)

    def _load(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            self._error = f"⚠️ 图片载入失败：{Path(path).name}"
            self.setFixedSize(240, 48)
            return
        scaled = pix.scaled(
            self.MAX_WIDTH, self.MAX_HEIGHT,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._pixmap = scaled
        self.setFixedSize(scaled.width() + 4, scaled.height() + 4)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._error:
            p.setPen(QColor("#9b9ba4"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._error)
            return

        if not self._pixmap:
            return

        w, h = self._pixmap.width(), self._pixmap.height()

        # 阴影
        shadow = QPainterPath()
        shadow.addRoundedRect(3, 3, w, h, self.RADIUS, self.RADIUS)
        p.fillPath(shadow, self.SHADOW_CLR)

        # 圆角裁剪后绘制图片
        clip = QPainterPath()
        clip.addRoundedRect(0, 0, w, h, self.RADIUS, self.RADIUS)
        p.setClipPath(clip)
        p.drawPixmap(0, 0, self._pixmap)
        p.setClipping(False)

        # 悬停高光边框
        if self.underMouse():
            border = QPainterPath()
            border.addRoundedRect(0.5, 0.5, w - 1, h - 1, self.RADIUS, self.RADIUS)
            p.setPen(QColor(255, 255, 255, 40))
            p.drawPath(border)

        p.end()

    def enterEvent(self, e):
        self.update(); super().enterEvent(e)

    def leaveEvent(self, e):
        self.update(); super().leaveEvent(e)


class VideoWidget(QWidget):
    """简易视频播放器，支持进度条拖拽"""

    FIXED_W, FIXED_H = 480, 270

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("video_widget")
        self.setFixedSize(self.FIXED_W, self.FIXED_H)

        self._current_path = path
        self._setup_player(path)
        self._build_ui()

    # ====================== 初始化播放器 ======================
    def _setup_player(self, path: str):
        self._video = QVideoWidget()
        self._video.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._audio = QAudioOutput()
        self._player = QMediaPlayer()

        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video)
        self._player.setSource(QUrl.fromLocalFile(path))

        # 连接信号
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.errorOccurred.connect(self._on_error)  # 方便调试

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # 视频显示区域
        main_layout.addWidget(self._video)

        # 控制栏
        control_layout = self._build_control_bar()
        main_layout.addLayout(control_layout)

    def _build_control_bar(self) -> QHBoxLayout:
        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(8, 4, 8, 4)

        # 播放/暂停按钮
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(32, 32)
        self._play_btn.clicked.connect(self._toggle_play)

        # 文件名
        self._name_lbl = QLabel(Path(self._current_path).name)
        self._name_lbl.setStyleSheet("color: rgba(220,220,220,0.7); font-size: 11px;")

        # 进度条
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)  # 初始时长未知
        self._slider.sliderMoved.connect(self._seek)  # 拖拽时跳转
        # self._slider.sliderReleased.connect(...)  # 如果需要更精细控制

        # 时间显示
        self._time_lbl = QLabel("00:00 / 00:00")
        self._time_lbl.setStyleSheet("color: rgba(200,200,200,0.8); font-size: 10px; min-width: 90px;")

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._name_lbl)
        ctrl.addWidget(self._slider, stretch=1)
        ctrl.addWidget(self._time_lbl)

        return ctrl

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸")
        else:
            self._play_btn.setText("▶")

    def _on_duration_changed(self, duration: int):
        """总时长变化时更新滑块范围（单位：毫秒）"""
        self._slider.setRange(0, duration)

    def _on_position_changed(self, position: int):
        """播放位置变化时更新滑块和时间显示"""
        if not self._slider.isSliderDown():  # 正在拖拽时不更新，避免冲突
            self._slider.setValue(position)

        self._update_time_label(position)

    def _seek(self, position: int):
        """拖拽进度条时跳转"""
        self._player.setPosition(position)

    def _update_time_label(self, position: int):
        duration = self._player.duration()
        current = self._format_time(position)
        total = self._format_time(duration) if duration > 0 else "00:00"
        self._time_lbl.setText(f"{current} / {total}")

    @staticmethod
    def _format_time(ms: int) -> str:
        if ms < 0:
            return "00:00"
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _on_error(self, error, error_string):
        print(f"Video Error: {error} - {error_string}")
        self._time_lbl.setText("播放错误")

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()


class AudioWidget(QWidget):
    """音频播放器组件 - 支持音乐播放、进度条、时间显示、波形风格图标"""

    FIXED_HEIGHT = 92

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("audio_widget")
        self._current_path = path
        self._filename = Path(path).name

        self._setup_player(path)
        self._build_ui()

    def _setup_player(self, path: str):
        self._audio_output = QAudioOutput()
        self._player = QMediaPlayer()

        self._player.setAudioOutput(self._audio_output)
        self._player.setSource(QUrl.fromLocalFile(path))

        # 信号连接
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.errorOccurred.connect(self._on_error)

    def _build_ui(self):
        self.setFixedHeight(self.FIXED_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # 上半部分：图标 + 文件名
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        self._icon_lbl = QLabel("🎵")
        self._icon_lbl.setFixedSize(48, 48)
        self._icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_lbl.setStyleSheet("""
            font-size: 28px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
        """)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)

        self._name_lbl = QLabel(self._filename)
        self._name_lbl.setStyleSheet("font-size: 13px; color: #E0E0E0;")

        self._time_lbl = QLabel("00:00 / 00:00")
        self._time_lbl.setStyleSheet("font-size: 10px; color: rgba(200,200,200,0.75);")

        info_layout.addWidget(self._name_lbl)
        info_layout.addWidget(self._time_lbl)

        top_layout.addWidget(self._icon_lbl)
        top_layout.addLayout(info_layout, stretch=1)

        # 播放按钮
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(42, 42)
        self._play_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_btn.clicked.connect(self._toggle_play)

        top_layout.addWidget(self._play_btn)

        # 进度条
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderMoved.connect(self._seek)

        # 组装
        layout.addLayout(top_layout)
        layout.addWidget(self._slider)

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("⏸")
            self._icon_lbl.setStyleSheet("""
                font-size: 28px;
                background: rgba(74, 222, 128, 0.2);
                border-radius: 10px;
                color: #4ade80;
            """)
        else:
            self._play_btn.setText("▶")
            self._icon_lbl.setStyleSheet("""
                font-size: 28px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 10px;
            """)

    def _on_duration_changed(self, duration: int):
        self._slider.setRange(0, duration)

    def _on_position_changed(self, position: int):
        if not self._slider.isSliderDown():
            self._slider.setValue(position)
        self._update_time_label(position)

    def _seek(self, position: int):
        self._player.setPosition(position)

    def _update_time_label(self, position: int):
        duration = self._player.duration()
        current = self._format_time(position)
        total = self._format_time(duration) if duration > 0 else "00:00"
        self._time_lbl.setText(f"{current} / {total}")

    @staticmethod
    def _format_time(ms: int) -> str:
        if ms < 0:
            return "00:00"
        seconds = ms // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _on_error(self, error, error_string):
        print(f"Audio Error: {error} - {error_string}")
        self._time_lbl.setText("播放错误")

    def play(self):
        self._player.play()

    def pause(self):
        self._player.pause()

    def stop(self):
        self._player.stop()

    def set_volume(self, volume: int):
        """0 ~ 100"""
        self._audio_output.setVolume(volume / 100.0)


class FileWidget(BaseWidget):
    """通用文件附件展示（不可直接预览的文件类型）。"""

    FIXED_WIDTH = 80
    FIXED_HEIGHT = 50

    _EXT_ICON: dict[str, str] = {
        ".glb": "🧊", ".obj": "🧊", ".fbx": "🧊", ".ply": "🧊",
        ".mp3": "🎵", ".wav": "🎵", ".flac": "🎵",
        ".pdf": "📄", ".txt": "📄", ".md": "📄",
        ".zip": "📦", ".tar": "📦",
    }

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("file_widget")
        self._path = path
        self._build_ui()
        self.setFixedSize(self.FIXED_WIDTH, self.FIXED_HEIGHT)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        layout.addWidget(self._build_icon())
        layout.addLayout(self._build_info())
        layout.addWidget(self._build_open_btn())

    def _build_icon(self) -> QLabel:
        ext  = Path(self._path).suffix.lower()
        icon = self._EXT_ICON.get(ext, "📎")
        lbl  = QLabel(icon)
        lbl.setFixedSize(32, 32)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "font-size:20px; background:rgba(255,255,255,0.05); border-radius:6px;"
        )
        return lbl

    def _build_info(self) -> QVBoxLayout:
        info = QVBoxLayout()
        name_lbl = QLabel(Path(self._path).name)
        name_lbl.setStyleSheet("font-size:13px; color:#DCDCDC;")
        size_lbl = QLabel(self._human_size())
        size_lbl.setStyleSheet("font-size:11px; color:rgba(220,220,220,0.45);")
        info.addWidget(name_lbl)
        info.addWidget(size_lbl)
        return info

    def _build_open_btn(self) -> QPushButton:
        btn = QPushButton("打开")
        btn.setFixedSize(45, 28)
        btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self._path))
        )
        return btn

    def _human_size(self) -> str:
        try:
            size = Path(self._path).stat().st_size
            for unit in ("B", "KB", "MB", "GB"):
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
        except OSError:
            pass
        return ""


if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    w = AudioWidget(r"C:\Users\HP\AppData\Local\Temp\out_animation\video.gif")
    w.show()
    app.exec()