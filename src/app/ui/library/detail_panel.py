import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.ui.base.widget import BaseWidget
from src.app.ui.library.activity import LoadingStrip
from src.app.ui.library.cards import repolish
from src.app.ui.library.pack_status import (
    item_title,
    media_kind,
    media_text,
    status_text,
    step_chain,
)
from src.app.ui.message.message_widgets import AudioWidget
from src.shared.schemas import (
    DIALOGUE_EMOTIONS,
    CollectionItem,
    PackResponse,
    StepState,
    TemplateInfo,
    TemplateStepInfo,
)

PANEL_WIDTH = 340
PREVIEW_WIDTH = PANEL_WIDTH - 32


@dataclass(frozen=True)
class AssetKey:
    """当前选中的素材：产物内容变了（重跑、修改剧本）签名就变，预览要重新加载。"""
    pack_id: str
    item_id: str
    step_id: str
    signature: str


def asset_key(pack_id: str, item: CollectionItem, step_id: str) -> AssetKey:
    state = item.steps.get(step_id) or StepState()
    return AssetKey(pack_id, item.id, step_id,
                    f"{state.status}|{'|'.join(state.outputs)}|{state.meta.get('elapsed', '')}")


def parse_script_lines(text: str) -> list[dict]:
    data = json.loads(text)
    return list(data.get("lines", [])) if isinstance(data, dict) else list(data)


class ScriptEditor(QWidget):
    """剧本审阅：一句一行，说话人和情绪用下拉，台词直接改。没改动时原样确认，不重写文件。"""
    approve_requested = Signal(object)  # 修改后的剧本 JSON；None 表示原样确认

    def __init__(self, text: str, speakers: list[str], approved: bool, parent=None):
        super().__init__(parent)
        self._speakers = speakers
        self._original = parse_script_lines(text)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.table = QTableWidget(0, 3)
        self.table.setObjectName("script_table")
        self.table.setHorizontalHeaderLabels([self.tr("Speaker"), self.tr("Emotion"), self.tr("Line")])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setWordWrap(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for line in self._original:
            self._add_row(line)
        self.table.setMinimumHeight(220)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        add_btn = QPushButton(self.tr("Add line"))
        remove_btn = QPushButton(self.tr("Remove line"))
        add_btn.clicked.connect(lambda: self._add_row({}))
        remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(add_btn)
        row.addWidget(remove_btn)
        row.addStretch()
        layout.addLayout(row)

        self.approve_btn = QPushButton(
            self.tr("Update script and voice again") if approved else self.tr("Confirm and voice"))
        self.approve_btn.setObjectName("library_primary_btn")
        self.approve_btn.clicked.connect(self._on_approve)
        layout.addWidget(self.approve_btn)

    def lines(self) -> list[dict]:
        result = []
        for row in range(self.table.rowCount()):
            text_cell = self.table.item(row, 2)
            result.append({
                "speaker": self.table.cellWidget(row, 0).currentText().strip(),
                "emotion": self.table.cellWidget(row, 1).currentText().strip(),
                "text": text_cell.text().strip() if text_cell else "",
            })
        return result

    def _add_row(self, line: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        speaker = QComboBox()
        speaker.addItems(self._speakers)
        name = line.get("speaker", "")
        # 模型可能写出不在角色表里的说话人：保留原值让用户看到并改掉，确认时后端会报错
        if name and name not in self._speakers:
            speaker.addItem(name)
        speaker.setCurrentText(name)
        emotion = QComboBox()
        emotion.addItems(DIALOGUE_EMOTIONS)
        emotion.setCurrentText(line.get("emotion", "") or DIALOGUE_EMOTIONS[0])
        self.table.setCellWidget(row, 0, speaker)
        self.table.setCellWidget(row, 1, emotion)
        self.table.setItem(row, 2, QTableWidgetItem(line.get("text", "")))

    def _remove_selected(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)

    def _on_approve(self):
        lines = self.lines()
        original = [{"speaker": line.get("speaker", ""), "emotion": line.get("emotion", ""),
                     "text": line.get("text", "")} for line in self._original]
        content = None if lines == original else json.dumps({"lines": lines}, ensure_ascii=False)
        self.approve_requested.emit(content)


class DetailPanel(BaseWidget):
    """素材详情侧栏：预览、提示词与字段、执行流程（每一步的模型、参数、耗时，可单步重做）。"""
    reset_requested = Signal(str, str)              # item id, step id
    approve_requested = Signal(str, str, object)    # item id, step id, 修改后的内容或 None
    fields_update_requested = Signal(str, dict)     # item id, 要改的字段（如事后补做分层）

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("library_detail")
        self.setFixedWidth(PANEL_WIDTH)
        self.key: AssetKey | None = None
        self._item_id = ""
        self._step: TemplateStepInfo | None = None
        self._speakers: list[str] = []
        self._approved = False
        self._busy = False
        self._redo_buttons: list[QPushButton] = []
        # (应用按钮, 下拉, 当前值)；_options_key 没变就不重建，免得轮询刷新把用户正在选的值冲掉
        self._option_rows: list[tuple[QPushButton, QComboBox, str]] = []
        self._options_key: tuple | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("library_scroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        body = QWidget()
        body.setObjectName("library_scroll_content")
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(10)
        scroll.setWidget(body)

        self.title_label = QLabel(self.tr("Select an asset to see how it was made"))
        self.title_label.setObjectName("detail_title")
        self.title_label.setWordWrap(True)
        self._layout.addWidget(self.title_label)
        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("detail_muted")
        self._layout.addWidget(self.subtitle_label)

        self._preview = QVBoxLayout()
        self._preview.setSpacing(6)
        self._layout.addLayout(self._preview)

        self.prompt_label = self._section_text()
        self.fields_label = self._section_text()
        self._prompt_heading = self._heading(self.tr("Prompt"))
        self._layout.addWidget(self._prompt_heading)
        self._layout.addWidget(self.prompt_label)
        self._layout.addWidget(self.fields_label)
        # 事后可改的选项（如给已生成的整图补做分层）：只在模板封面（整图）上出现
        self._options = QVBoxLayout()
        self._options.setSpacing(6)
        self._layout.addLayout(self._options)

        self._flow_heading = self._heading(self.tr("Flow"))
        self._layout.addWidget(self._flow_heading)
        self._flow = QVBoxLayout()
        self._flow.setSpacing(4)
        self._layout.addLayout(self._flow)
        self._layout.addStretch()
        self._set_sections_visible(False)

    # ---- 对外 ----
    def show_asset(self, pack: PackResponse, template: TemplateInfo, item: CollectionItem,
                   step: TemplateStepInfo, busy: bool) -> bool:
        """刷新详情；返回 True 表示换了素材或产物变了，调用方需要重新加载预览。"""
        key = asset_key(pack.manifest.id, item, step.id)
        changed = key != self.key
        self.key, self._item_id, self._step, self._busy = key, item.id, step, busy
        self._speakers = [member.name for member in pack.manifest.cast]
        self._approved = (item.steps.get(step.id) or StepState()).approved

        self.title_label.setText(item_title(item))
        self.subtitle_label.setText(
            f"{step.label or step.id}  ·  {media_text(media_kind(step.type))}")
        self.prompt_label.setText(item.prompt)
        self.fields_label.setText(self._fields_text(item, template))
        self.fields_label.setVisible(bool(self.fields_label.text()))
        self._set_sections_visible(True)
        self._rebuild_flow(item, step_chain(template, step.id))
        self._rebuild_options(template, item, step)
        if changed:
            self._clear_preview()
            state = item.steps.get(step.id) or StepState()
            if state.status != "done":
                self._add_preview(self._muted(self.tr("Not generated yet")))
            else:
                # 预览要从后端取，取到之前先占位，set_image / set_audio / set_text 会替换掉
                strip = LoadingStrip(height=3)
                strip.set_loading(True)
                self._add_preview(strip)
                self._add_preview(self._muted(self.tr("Loading preview…")))
        return changed

    def clear(self):
        self.key = None
        self.title_label.setText(self.tr("Select an asset to see how it was made"))
        self.subtitle_label.clear()
        self._clear_preview()
        self._clear_layout(self._flow)
        self._clear_layout(self._options)
        self._redo_buttons = []
        self._option_rows = []
        self._options_key = None
        self._set_sections_visible(False)

    def set_busy(self, busy: bool):
        self._busy = busy
        for button in self._redo_buttons:
            button.setEnabled(not busy)
        self._refresh_option_buttons()

    def _refresh_option_buttons(self):
        for button, combo, current in self._option_rows:
            button.setEnabled(not self._busy and combo.currentData() != current)

    def set_image(self, key: AssetKey, pixmap: QPixmap):
        if key != self.key:
            return
        self._clear_preview()
        label = QLabel()
        label.setObjectName("detail_image")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setPixmap(pixmap.scaledToWidth(min(PREVIEW_WIDTH, pixmap.width()),
                                             Qt.TransformationMode.SmoothTransformation))
        label.setToolTip(self.tr("Double-click the card to enlarge"))
        self._add_preview(label)
        if self._step is not None and pixmap.width():
            self._add_preview(self._muted(f"{pixmap.width()} × {pixmap.height()}"))

    def set_images(self, key: AssetKey, pixmaps: list[QPixmap]):
        """一个产物里有多张图（如分层的各个图层）：从第一张到最后一张竖着排，第一张是最底层。"""
        if key != self.key:
            return
        self._clear_preview()
        last = len(pixmaps)
        for index, pixmap in enumerate(pixmaps, start=1):
            caption = (self.tr("{0} (bottom)") if index == 1 else
                       self.tr("{0} (top)") if index == last else "{0}").format(index)
            self._add_preview(self._muted(caption))
            label = QLabel()
            label.setObjectName("detail_image")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(pixmap.scaledToWidth(min(PREVIEW_WIDTH, pixmap.width()),
                                                 Qt.TransformationMode.SmoothTransformation))
            self._add_preview(label)
        if pixmaps and pixmaps[0].width():
            self._add_preview(self._muted(f"{pixmaps[0].width()} × {pixmaps[0].height()}"))

    def set_audio(self, key: AssetKey, tracks: list[tuple[str, Path]]):
        """tracks：(说明文字, 本地文件)。对话配音每句一条，说明是"说话人：台词"。"""
        if key != self.key:
            return
        self._clear_preview()
        for caption, path in tracks:
            if caption:
                text = QLabel(caption)
                text.setObjectName("detail_text")
                text.setWordWrap(True)
                self._add_preview(text)
            self._add_preview(AudioWidget(str(path)))

    def set_text(self, key: AssetKey, text: str):
        if key != self.key or self._step is None:
            return
        self._clear_preview()
        if self._step.review:
            try:
                editor = ScriptEditor(text, self._speakers, self._approved)
            except (ValueError, AttributeError) as exc:
                self._add_preview(self._error(self.tr("Cannot read script: {0}").format(exc)))
                return
            editor.approve_btn.setEnabled(not self._busy)
            editor.approve_requested.connect(
                lambda content: self.approve_requested.emit(self._item_id, self._step.id, content))
            self._add_preview(editor)
            return
        viewer = QPlainTextEdit(text)
        viewer.setObjectName("library_text_edit")
        viewer.setReadOnly(True)
        viewer.setMinimumHeight(200)
        self._add_preview(viewer)

    def set_preview_error(self, key: AssetKey, message: str):
        if key == self.key:
            self._clear_preview()
            self._add_preview(self._error(message))

    # ---- 事后可改的选项 ----
    def _rebuild_options(self, template: TemplateInfo, item: CollectionItem,
                         step: TemplateStepInfo):
        """模板里决定某些步骤做不做的下拉字段（如分层），在封面素材上给出修改入口。

        改了之后后端只把依赖这个字段的步骤标记为待重做，已生成的整图不动。
        """
        conditional = {s.when for s in template.step_details if s.when}
        options_key = (self.key.pack_id if self.key else "", item.id, step.id,
                       tuple((f, item.fields.get(f, "")) for f in sorted(conditional)))
        if options_key == self._options_key:
            return
        self._options_key = options_key
        self._clear_layout(self._options)
        self._option_rows = []
        if step.id != template.cover:
            return
        for spec in template.fields:
            if spec.id not in conditional or not spec.options:
                continue
            current = item.fields.get(spec.id) or spec.default
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            label = QLabel(spec.label or spec.id)
            label.setObjectName("detail_muted")
            combo = QComboBox()
            for option in spec.options:
                combo.addItem(option.label or option.value, option.value)
            combo.setCurrentIndex(max(combo.findData(current), 0))
            apply_btn = QPushButton(self.tr("Apply"))
            apply_btn.setObjectName("library_primary_btn")
            apply_btn.setToolTip(self.tr(
                "Redo only the steps that depend on this option; the image itself is kept."))
            combo.currentIndexChanged.connect(self._refresh_option_buttons)
            apply_btn.clicked.connect(
                lambda _=False, c=combo, field_id=spec.id, item_id=item.id:
                self.fields_update_requested.emit(item_id, {field_id: c.currentData()}))
            layout.addWidget(label)
            layout.addWidget(combo, 1)
            layout.addWidget(apply_btn)
            self._options.addWidget(row)
            self._option_rows.append((apply_btn, combo, current))
        self._refresh_option_buttons()

    # ---- 执行流程 ----
    def _rebuild_flow(self, item: CollectionItem, chain: list[TemplateStepInfo]):
        self._clear_layout(self._flow)
        self._redo_buttons = []
        for index, step in enumerate(chain, start=1):
            if index > 1:
                arrow = QLabel("↓")
                arrow.setObjectName("flow_arrow")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._flow.addWidget(arrow)
            self._flow.addWidget(self._flow_step(index, item, step))

    def _flow_step(self, index: int, item: CollectionItem, step: TemplateStepInfo) -> QFrame:
        state = item.steps.get(step.id) or StepState()
        frame = QFrame()
        frame.setObjectName("flow_step")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        header = QHBoxLayout()
        name = QLabel(f"{index}. {step.label or step.id}")
        name.setObjectName("flow_step_name")
        status = "review" if step.review and state.status == "done" and not state.approved \
            else state.status
        badge = QLabel(status_text(status))
        badge.setObjectName("card_badge")
        badge.setProperty("status", status)
        repolish(badge)
        header.addWidget(name)
        header.addStretch()
        header.addWidget(badge)
        redo = QPushButton(self.tr("Redo"))
        redo.setObjectName("flow_redo_btn")
        redo.setToolTip(self.tr("Regenerate this step and everything after it"))
        redo.setEnabled(not self._busy)
        redo.clicked.connect(lambda: self.reset_requested.emit(item.id, step.id))
        self._redo_buttons.append(redo)
        header.addWidget(redo)
        layout.addLayout(header)

        details = [part for part in (
            state.meta.get("model", ""),
            "{} × {}".format(*state.meta["size"]) if state.meta.get("size") else "",
            self.tr("{0}s").format(state.meta["elapsed"]) if "elapsed" in state.meta else "",
        ) if part]
        if details:
            layout.addWidget(self._muted("  ·  ".join(details)))
        if state.meta.get("prompt"):
            prompt = self._muted(state.meta["prompt"])
            prompt.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(prompt)
        params = state.meta.get("params") or {}
        if params:
            layout.addWidget(self._muted(", ".join(f"{k}={v}" for k, v in params.items())))
        if state.error:
            layout.addWidget(self._error(state.error))
        return frame

    # ---- 工具 ----
    @staticmethod
    def _fields_text(item: CollectionItem, template: TemplateInfo) -> str:
        lines = []
        for spec in template.fields:
            value = item.fields.get(spec.id, "")
            if not value:
                continue
            # 下拉字段存的是拼进提示词的英文值，给人看的是选项名
            labels = {option.value: option.label for option in spec.options}
            lines.append(f"{spec.label or spec.id}：{labels.get(value) or value}")
        return "\n".join(lines)

    def _set_sections_visible(self, visible: bool):
        for widget in (self._prompt_heading, self.prompt_label, self._flow_heading):
            widget.setVisible(visible)
        if not visible:
            self.fields_label.hide()

    def _heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("detail_heading")
        return label

    def _section_text(self) -> QLabel:
        label = QLabel()
        label.setObjectName("detail_text")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _muted(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("detail_muted")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _error(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("library_error")
        label.setWordWrap(True)
        return label

    def _add_preview(self, widget: QWidget):
        self._preview.addWidget(widget)

    def _clear_preview(self):
        self._clear_layout(self._preview)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
