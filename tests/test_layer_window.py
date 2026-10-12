"""图层窗口控件契约，不连接后端。"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtTest import QTest

from src import resources  # noqa: F401  注册 :/svg 图标，眼睛按钮要用
from src.app.ui.library.layer_window import LayerNode, LayerWindow

SEG0 = "segment:0+1200"
SEG1 = "segment:1200+800"
CTRL = Qt.KeyboardModifier.ControlModifier
SHIFT = Qt.KeyboardModifier.ShiftModifier
NONE = Qt.KeyboardModifier.NoModifier


def _nodes():
    return [
        LayerNode(SEG0, "Segment 1", badge="3", version_count=2, current_version=-1, children=[
            LayerNode(f"layer:0+1200:L1", "Layer 1"),
            LayerNode(f"layer:0+1200:L2", "Layer 2"),
            LayerNode(f"layer:0+1200:M1", "Merged 1"),
        ]),
        LayerNode(SEG1, "Segment 2", children=[LayerNode("layer:1200+800:L1", "Layer 1")]),
        LayerNode("screen:01", "Screen 1"),
    ]


def _window(qapp):
    win = LayerWindow()
    win.resize(400, 600)
    win.set_nodes(_nodes())
    return win


def _click(win, key, modifier=NONE):
    QTest.mouseClick(win.row(key), Qt.MouseButton.LeftButton, modifier)


def test_tree_has_rows_children_and_folds(qapp):
    win = _window(qapp)
    child = "layer:0+1200:L1"
    assert win.row(SEG0) is not None and win.row(child) is not None
    assert win.row(child).isVisibleTo(win)
    win.row(SEG0).fold_btn.click()
    assert win.is_collapsed(SEG0) and not win.row(child).isVisibleTo(win)
    assert win.row(SEG1).isVisibleTo(win)
    win.row(SEG0).fold_btn.click()
    assert not win.is_collapsed(SEG0) and win.row(child).isVisibleTo(win)
    # 叶子没有折叠效果
    win.toggle_fold("screen:01")
    assert not win.is_collapsed("screen:01")
    # 徽标
    assert win.row(SEG0).badge.text() == "3"
    assert win.row("screen:01").badge.isHidden()


def test_click_ctrl_shift_and_ctrl_a_selection(qapp):
    win = _window(qapp)
    changes = []
    win.selection_changed.connect(changes.append)

    _click(win, SEG1)
    assert win.selected_keys() == [SEG1] and changes[-1] == [SEG1]
    _click(win, "screen:01", CTRL)
    assert win.selected_keys() == [SEG1, "screen:01"]
    _click(win, SEG1, CTRL)
    assert win.selected_keys() == ["screen:01"]

    # Shift 按显示顺序连选：锚点是上一次点的行
    _click(win, "layer:0+1200:L1")
    _click(win, "layer:0+1200:M1", SHIFT)
    assert win.selected_keys() == ["layer:0+1200:L1", "layer:0+1200:L2", "layer:0+1200:M1"]

    count = len(changes)
    QTest.keyClick(win, Qt.Key.Key_A, CTRL)
    assert len(win.selected_keys()) == 7 and len(changes) == count + 1

    win.select([SEG0, "nope"])
    assert win.selected_keys() == [SEG0]
    assert win.row(SEG0).property("selected") is True


def test_shift_range_skips_rows_hidden_by_folding(qapp):
    win = _window(qapp)
    win.toggle_fold(SEG0)
    _click(win, SEG0)
    _click(win, SEG1, SHIFT)
    assert win.selected_keys() == [SEG0, SEG1]


def test_eye_toggles_visibility_and_emits(qapp):
    win = _window(qapp)
    seen = []
    win.visibility_changed.connect(seen.append)
    assert len(win.visible_keys()) == 7
    win.row(SEG1).eye.click()
    assert SEG1 not in win.visible_keys() and SEG1 not in seen[-1]
    assert len(seen[-1]) == 6
    win.row(SEG1).eye.click()
    assert SEG1 in win.visible_keys() and len(seen) == 2


def test_node_visible_false_initial_and_state_survives_rebuild(qapp):
    win = LayerWindow()
    win.set_nodes([LayerNode("screen:01", "A", visible=False), LayerNode("screen:02", "B")])
    assert win.visible_keys() == ["screen:02"]
    win.row("screen:02").eye.click()
    win.set_nodes([LayerNode("screen:01", "A", visible=False), LayerNode("screen:02", "B"),
                   LayerNode("screen:03", "C")])
    assert win.visible_keys() == ["screen:03"]


def test_rebuild_keeps_selection_and_folds_and_drops_gone_nodes(qapp):
    win = _window(qapp)
    win.toggle_fold(SEG0)
    win.select([SEG1, "screen:01"])
    changes = []
    win.selection_changed.connect(changes.append)

    win.set_nodes(_nodes())
    assert win.selected_keys() == [SEG1, "screen:01"] and changes == []
    assert win.is_collapsed(SEG0)
    assert win.row(SEG1).property("selected") is True

    win.set_nodes(_nodes()[:1])
    assert win.selected_keys() == [] and changes == [[]]


def test_version_control_emits_target_index(qapp):
    win = _window(qapp)
    got = []
    win.version_requested.connect(lambda key, index: got.append((key, index)))
    version = win.row(SEG0).version
    assert version.label.text() == "orig"
    assert not version.prev_btn.isEnabled() and version.next_btn.isEnabled()
    version.next_btn.click()
    assert got == [(SEG0, 0)]

    nodes = _nodes()
    nodes[0].current_version = 1
    win.set_nodes(nodes)
    version = win.row(SEG0).version
    assert version.label.text() == "2/2"
    assert version.prev_btn.isEnabled() and not version.next_btn.isEnabled()
    version.prev_btn.click()
    assert got[-1] == (SEG0, 0)
    nodes[0].current_version = 0
    win.set_nodes(nodes)
    win.row(SEG0).version.prev_btn.click()
    assert got[-1] == (SEG0, -1)
    # 没有版本的节点不显示控件
    assert win.row(SEG1).version is None


def test_send_enabled_only_with_single_selection_and_idle(qapp):
    win = _window(qapp)
    sent = []
    win.edit_requested.connect(lambda key, prompt: sent.append((key, prompt)))
    assert not win.send_btn.isEnabled()
    _click(win, SEG0)
    assert win.send_btn.isEnabled()
    _click(win, SEG1, CTRL)
    assert not win.send_btn.isEnabled()
    _click(win, SEG1)

    win.send_btn.click()          # 空提示词不发
    assert sent == []
    win.prompt_edit.setPlainText("  make the ground steeper ")
    win.send_btn.click()
    assert sent == [(SEG1, "make the ground steeper")]
    assert win.prompt_edit.toPlainText() == ""

    win.prompt_edit.setPlainText("again")
    QTest.keyClick(win.prompt_edit, Qt.Key.Key_Return, CTRL)
    assert sent[-1] == (SEG1, "again") and win.prompt_edit.toPlainText() == ""

    win.set_idle(False)
    assert not win.send_btn.isEnabled()
    win.prompt_edit.setPlainText("x")
    QTest.keyClick(win.prompt_edit, Qt.Key.Key_Return, CTRL)
    assert len(sent) == 2
    win.set_idle(True)
    assert win.send_btn.isEnabled()


def test_plain_enter_in_prompt_inserts_newline(qapp):
    win = _window(qapp)
    _click(win, SEG0)
    win.prompt_edit.setPlainText("a")
    QTest.keyClick(win.prompt_edit, Qt.Key.Key_Return)
    assert "\n" in win.prompt_edit.toPlainText()


def test_ctrl_a_in_prompt_does_not_select_nodes(qapp):
    win = _window(qapp)
    win.show()
    win.prompt_edit.setFocus()
    win.prompt_edit.setPlainText("abc")
    QTest.keyClick(win.prompt_edit, Qt.Key.Key_A, CTRL)
    assert win.selected_keys() == []
    win.hide()


def test_tool_buttons_text_state_and_signal(qapp):
    win = _window(qapp)
    names = ["extend", "seams", "segment", "layer"]
    assert list(win.tool_buttons) == names
    assert [win.tool_buttons[n].text() for n in names] == ["Extend", "Seams", "Segment", "Layer"]
    got = []
    win.tool_clicked.connect(got.append)
    for name in names:
        win.tool_buttons[name].click()
    assert got == names

    win.set_tool_state("segment", running=True, enabled=True, tooltip="busy")
    assert win.tool_buttons["segment"].text() == "Stop segment"
    assert win.tool_buttons["segment"].toolTip() == "busy"
    win.set_tool_state("layer", running=False, enabled=False, tooltip="Select segments first")
    assert not win.tool_buttons["layer"].isEnabled()
    win.tool_buttons["layer"].click()
    assert got[-1] == "layer"      # 禁用的按钮 click() 不触发
    win.set_tool_state("segment", running=False, enabled=True)
    assert win.tool_buttons["segment"].text() == "Segment"
    win.set_tool_state("extend", True, True)
    win.set_tool_state("seams", True, True)
    win.set_tool_state("layer", True, True)
    assert [win.tool_buttons[n].text() for n in ("extend", "seams", "layer")] == [
        "Stop extend", "Stop seams", "Stop layer"]


def test_merge_and_delete_rules(qapp):
    win = _window(qapp)
    merged, deleted = [], []
    win.merge_requested.connect(merged.append)
    win.delete_requested.connect(deleted.append)
    assert not win.merge_btn.isEnabled() and not win.delete_btn.isEnabled()

    win.select(["layer:0+1200:L1", "layer:0+1200:L2"])
    assert win.merge_btn.isEnabled() and not win.delete_btn.isEnabled()
    win.merge_btn.click()
    assert merged == [["layer:0+1200:L1", "layer:0+1200:L2"]]

    # 跨段、或含非层节点都不能合并
    win.select(["layer:0+1200:L1", "layer:1200+800:L1"])
    assert not win.merge_btn.isEnabled()
    win.select(["layer:0+1200:L1", SEG0])
    assert not win.merge_btn.isEnabled()

    win.select(["layer:0+1200:L1"])
    assert not win.delete_btn.isEnabled()
    win.select(["layer:0+1200:M1"])
    assert win.delete_btn.isEnabled() and not win.merge_btn.isEnabled()
    win.delete_btn.click()
    assert deleted == ["layer:0+1200:M1"]

    win.set_idle(False)
    assert not win.delete_btn.isEnabled()
    win.select(["layer:0+1200:L1", "layer:0+1200:L2"])
    assert not win.merge_btn.isEnabled()


def test_idle_disables_version_control_and_double_click_opens(qapp):
    win = _window(qapp)
    win.set_idle(False)
    assert not win.row(SEG0).version.isEnabled()
    win.set_idle(True)
    assert win.row(SEG0).version.isEnabled()
    opened = []
    win.node_opened.connect(opened.append)
    QTest.mouseDClick(win.row(SEG1), Qt.MouseButton.LeftButton)
    assert opened == [SEG1]


def test_set_images_sets_thumbnails(qapp):
    win = _window(qapp)
    image = QImage(100, 50, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    win.set_images({SEG1: image})
    assert win.row(SEG1).thumb.pixmap() is not None and not win.row(SEG1).thumb.pixmap().isNull()
    assert win.row(SEG0).thumb.pixmap().isNull()
    # 重建后缩略图仍在
    win.set_nodes(_nodes())
    assert not win.row(SEG1).thumb.pixmap().isNull()
