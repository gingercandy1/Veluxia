"""聊天历史持久化的契约测试：重复保存不产生重复项、思考过程和中途停止的附件能存能恢复。

Seam：ChatSessionManager（SQLite 落在 tmp_path），AssistantMessageBubble 的
get_persisted_content / 从历史 dict 重建。不需要后端和模型。
"""
import sqlite3

from src.app.ui.chat.chat_session_manager import ChatSessionManager


def _assistant(message_id: str, text: str, extra: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "message_id": message_id,
        "time": "12:00",
        "model_type": "text",
        "model_name": "Qwen3-8B",
        "content": {"content": text, "attachments": [], "extra": extra or {}},
    }


def _db_rows(tmp_path, session_id):
    conn = sqlite3.connect(tmp_path / "chat_history.db")
    try:
        return conn.execute("SELECT message_id, content FROM messages WHERE session_id = ?",
                            (session_id,)).fetchall()
    finally:
        conn.close()


def test_saving_same_message_twice_keeps_one_entry(tmp_path):
    manager = ChatSessionManager(str(tmp_path))
    session_id = manager.create_new_session()

    manager.add_message(_assistant("a1", "partial"))
    manager.add_message(_assistant("a1", "final"))

    history = manager.get_history()
    assert [m["message_id"] for m in history] == ["a1"]
    assert history[0]["content"]["content"] == "final"
    assert _db_rows(tmp_path, session_id) == [("a1", "final")]


def test_thinking_survives_reload(tmp_path):
    manager = ChatSessionManager(str(tmp_path))
    session_id = manager.create_new_session()
    manager.add_message(_assistant("a1", "answer", {"thinking": "step 1, step 2"}))

    reloaded = ChatSessionManager(str(tmp_path))
    reloaded.switch_session(session_id)
    assert reloaded.get_history()[0]["content"]["extra"] == {"thinking": "step 1, step 2"}


def test_deleting_session_removes_its_messages(tmp_path):
    manager = ChatSessionManager(str(tmp_path))
    session_id = manager.create_new_session()
    manager.add_message(_assistant("a1", "hello"))

    manager.delete_session(session_id)

    assert _db_rows(tmp_path, session_id) == []


def test_assistant_bubble_persists_and_restores_thinking(qapp):
    from src.app.ui.message.message_bubble import AssistantMessageBubble

    live = AssistantMessageBubble("", "12:00", "a1")
    live.append_thinking("step 1, ")
    live.append_thinking("step 2")
    live.append_output("answer")
    saved = live.get_persisted_content()
    assert saved["extra"] == {"thinking": "step 1, step 2"}

    restored = AssistantMessageBubble(saved, "12:00", "a1")
    assert restored.thinking_block.isVisibleTo(restored)
    assert restored.get_persisted_content() == saved


def test_stopped_media_bubble_keeps_already_shown_images(qapp, tmp_path):
    from src.app.ui.message.message_bubble import AssistantMessageBubble

    bubble = AssistantMessageBubble("", "12:00", "a1", model_type="image", item_count=3)
    first = str(tmp_path / "first.png")
    bubble.add_partial_attachment(first)

    assert bubble.get_persisted_content()["attachments"] == [first]
    assert bubble.get_persisted_content()["extra"] == {}
