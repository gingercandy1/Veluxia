import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from src.app.ui.setting.page.log_page import log_error, log_warning


class HistoryManager:
    """
    将聊天记录持久化到本地 JSON 文件。
    每个 session_id 对应一个文件。
    """
    def __init__(self, save_dir: str = "./config/chat_history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.save_dir / "chat_history.db"
        self._current_session_id = ""
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id  TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    model_name  TEXT,
                    model_type  TEXT DEFAULT 'text',
                    content     TEXT,
                    attachments TEXT DEFAULT '[]',
                    extra       TEXT DEFAULT '{}',
                    time        TEXT,
                    created_at  TEXT NOT NULL
                );

                -- 按 session 查消息的主要查询路径
                CREATE INDEX IF NOT EXISTS idx_msg_session
                    ON messages(session_id, created_at);

                -- 为将来关键词搜索预留（后续可替换为 FTS5 虚拟表）
                CREATE INDEX IF NOT EXISTS idx_msg_content
                    ON messages(content);
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            log_error(f"数据库操作失败: {e}")
            raise
        finally:
            conn.close()

    def set_current_session(self, session_id: str):
        self._current_session_id = session_id

    def get_current_session(self) -> str:
        return self._current_session_id

    def create_new_session(self, name: str = None) -> str:
        """创建新会话"""
        if not name:
            name = f"会话 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        session_id = f"session_{int(datetime.now().timestamp())}"
        now = datetime.now().isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, name, now, now),
            )
        return session_id

    def list_sessions(self) -> list[dict]:
        """返回所有会话，按最近更新排序，包含消息数量。"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.session_id,
                       s.title,
                       s.created_at,
                       s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def rename_session(self, session_id: str, new_title: str):
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                (new_title, datetime.now().isoformat(), session_id),
            )

    def delete(self, session_id: str):
        """删除会话及其所有消息（CASCADE 自动处理）。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def append_message(self, session_id: str, message: dict):
        """追加单条消息，比全量 save 高效，add_message 场景专用。"""
        with self._connect() as conn:
            self._insert_message(conn, session_id, message)
            self._touch_session(conn, session_id)

    def save(self, session_id: str, messages: list[dict]):
        """全量替换某个 session 的所有消息（用于 truncate 等场景）。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            for msg in messages:
                self._insert_message(conn, session_id, msg)
            self._touch_session(conn, session_id)

    def load(self, session_id: str) -> list[dict]:
        """按时间顺序加载某个 session 的所有消息。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def update(self, session_id: str, message_id: str, content: str):
        """更新某条消息的文本内容。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE messages SET content = ? WHERE message_id = ? AND session_id = ?",
                (content, message_id, session_id),
            )
            self._touch_session(conn, session_id)

    def delete_messages_after(self, session_id: str, keep_ids: list[str]):
        """
        保留 keep_ids 中的消息，删除该 session 内其余消息。
        用于 truncate_after 场景，避免全量重写。
        """
        if not keep_ids:
            with self._connect() as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                self._touch_session(conn, session_id)
            return

        placeholders = ",".join("?" * len(keep_ids))
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM messages WHERE session_id = ? AND message_id NOT IN ({placeholders})",
                [session_id] + keep_ids,
            )
            self._touch_session(conn, session_id)

    def search_messages(self, keyword: str) -> list[dict]:
        """
        搜索包含关键词的消息。
        返回结果包含 session_id / session_title，供 UI 跳转定位使用。

        升级路径：
            将来数据量大时，把 messages 中的 content 迁移到 FTS5 虚拟表，
            将 LIKE 查询改为 FTS MATCH 即可，返回结构不变。
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT m.*, s.title AS session_title
                FROM messages m
                JOIN sessions s ON m.session_id = s.session_id
                WHERE m.content LIKE ?
                ORDER BY m.created_at DESC
            """, (f"%{keyword}%",)).fetchall()

        results = []
        for row in rows:
            msg = self._row_to_message(row)
            msg["session_id"] = row["session_id"]
            msg["session_title"] = row["session_title"]
            results.append(msg)
        return results

    def _insert_message(self, conn, session_id: str, msg: dict):
        content_obj = msg.get("content", {})

        if isinstance(content_obj, str):
            content_text = content_obj
            attachments_str = "[]"
            extra_str = "{}"
        else:
            content_text = content_obj.get("content", "")
            attachments_str = json.dumps(
                content_obj.get("attachments", []), ensure_ascii=False
            )
            extra_str = json.dumps(
                content_obj.get("extra", {}), ensure_ascii=False
            )

        message_id = msg.get("message_id", "")
        role = msg.get("role", "")
        model_name = msg.get("model_name", "")
        model_type = msg.get("model_type", "text")
        time = msg.get("time", "")
        created_at = msg.get("created_at", datetime.now().isoformat())

        conn.execute("""
            INSERT OR REPLACE INTO messages
                (message_id, session_id, role, model_name, model_type,
                 content, attachments, extra, time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (message_id,
              session_id,
              role,
              model_name,
              model_type,
              content_text,
              attachments_str,
              extra_str,
              time,
              created_at))

    def _row_to_message(self, row) -> dict:
        """将数据库行还原为与原 JSON 格式完全一致的 message dict。"""
        return {
            "message_id": row["message_id"],
            "role": row["role"],
            "model_name": row["model_name"],
            "model_type": row["model_type"],
            "time": row["time"],
            "created_at": row["created_at"],
            "content": {
                "content": row["content"],
                "attachments": json.loads(row["attachments"] or "[]"),
                "extra": json.loads(row["extra"] or "{}"),
                "model_name": row["model_name"],
            },
        }

    def _touch_session(self, conn, session_id: str):
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id),
        )


class ChatSessionManager(QObject):
    """聊天会话管理器 """

    session_changed = Signal()
    history_updated = Signal()
    session_list_changed = Signal()

    def __init__(self, save_dir: str = "./config/chat_history"):
        super().__init__()
        self._history_mgr = HistoryManager(save_dir)
        self._current_session_id: str = ""
        self._history: list[dict] = []

    def create_new_session(self, title: str = None) -> str:
        new_id = self._history_mgr.create_new_session(title)
        self.switch_session(new_id)
        return new_id

    def switch_session(self, session_id: str):
        self._history_mgr.set_current_session(session_id)
        self._load_history(session_id)
        self.session_changed.emit()

    def get_current_session_id(self) -> str:
        return self._current_session_id

    def list_sessions(self) -> list[dict]:
        return self._history_mgr.list_sessions()

    def rename_session(self, session_id: str, new_title: str):
        self._history_mgr.rename_session(session_id, new_title)
        self.session_list_changed.emit()

    def delete_session(self, session_id: str):
        self._history_mgr.delete(session_id)
        if session_id == self._current_session_id:
            self.create_new_session()
        self.session_list_changed.emit()

    # 历史数据操作
    def add_message(self, message: dict):
        self._history.append(message)
        self._history_mgr.append_message(self._current_session_id, message)
        self.history_updated.emit()

    def update_message_content(self, message_id: str, new_content: str):
        self._history_mgr.update(self._current_session_id, message_id, new_content)
        # 同步更新内存中的 _history
        for msg in self._history:
            if msg.get("message_id") == message_id:
                if isinstance(msg.get("content"), dict):
                    msg["content"]["content"] = new_content
                else:
                    msg["content"] = new_content
                break
        self.history_updated.emit()

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def clear_current_session(self):
        self._history.clear()
        self._history_mgr.save(self._current_session_id, [])

    def prepare_for_retry(self, message_id: str) -> tuple[Optional[dict], bool]:
        """
        为重试做准备：查找消息索引，并删除后面的 assistant 回复（如果存在）
        返回 (user_message, index)
        """
        history = self.get_history()
        current_index = -1
        current_msg = None

        for index, msg in enumerate(history):
            if msg.get("message_id") == message_id:
                current_msg = msg
                current_index = index
                break

        if current_index == -1:
            return None, False

        # 删除后面的 assistant 回复
        if current_index + 1 < len(history):
            log_warning(f"重试：删除第 {current_index + 1} 条之后的所有消息")
            self.truncate_after(current_index)

        return current_msg, True

    def truncate_after(self, index: int):
        """截断指定索引之后的所有消息"""
        if index + 1 >= len(self._history):
            return

        self._history = self._history[:index + 1]
        keep_ids = [m["message_id"] for m in self._history]
        self._history_mgr.delete_messages_after(self._current_session_id, keep_ids)
        self.history_updated.emit()

    def _load_history(self, session_id: str):
        self._history = self._history_mgr.load(session_id)