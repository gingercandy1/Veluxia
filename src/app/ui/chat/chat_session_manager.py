import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ui.setting.page.log_page import log_error, log_warning


class HistoryManager:
    """
    将聊天记录持久化到本地 JSON 文件。
    每个 session_id 对应一个文件。
    """
    def __init__(self, save_dir: str = "./config/chat_history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._current_session_id = ""

    def set_current_session(self, session_id: str):
        self._current_session_id = session_id

    def get_current_session(self) -> str:
        return self._current_session_id

    def create_new_session(self, name: str = None) -> str:
        """创建新会话"""
        if not name:
            name = f"会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"

        session_id = f"session_{int(datetime.now().timestamp())}"

        # 创建空会话文件并写入元数据
        metadata = {
            "session_id": session_id,
            "title": name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        self._path(session_id).write_text(
            json.dumps([metadata], ensure_ascii=False, indent=2),  # 第一条记录存元数据
            encoding="utf-8"
        )

        self.save(session_id, [metadata])
        return session_id

    def list_sessions(self) -> list[dict]:
        """返回带标题的会话列表"""
        sessions = []
        for p in sorted(self.save_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data and isinstance(data[0], dict) and "title" in data[0]:
                    sessions.append({
                        "session_id": p.stem,
                        "title": data[0]["title"],
                        "created_at": data[0].get("created_at"),
                        "updated_at": data[0].get("updated_at"),
                        "message_count": len(data) - 1
                    })
            except:
                continue
        return sessions

    def rename_session(self, session_id: str, new_title: str):
        messages = self.load(session_id)
        if messages and isinstance(messages[0], dict):
            messages[0]["title"] = new_title
            messages[0]["updated_at"] = datetime.now().isoformat()
            self.save(session_id, messages)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.save_dir / f"{safe}.json"

    def save(self, session_id: str, messages: list[dict]):
        try:
            self._path(session_id).write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            log_error(f"⚠️ 历史保存失败: {e}")

    def load(self, session_id: str) -> list[dict]:
        p = self._path(session_id)
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    def delete(self, session_id: str):
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    def update(self, session_id: str, message_id: str, content: str):
        messages = self.load(session_id)
        for message in messages:
            if message["message_id"] == message_id:
                message["content"] = content

        self.save(session_id, messages)



class ChatSessionManager(QObject):
    """聊天会话管理器 """

    session_changed = Signal(str)
    history_updated = Signal()
    session_list_changed = Signal()

    def __init__(self, save_dir: str = "./config/chat_history"):
        super().__init__()
        self._history_mgr = HistoryManager(save_dir)
        self._current_session_id: str = ""
        self._history: list[dict] = []

    # 会话管理
    def create_new_session(self, title: str = None) -> str:
        new_id = self._history_mgr.create_new_session(title)
        self.switch_session(new_id)
        return new_id

    def switch_session(self, session_id: str):
        if session_id == self._current_session_id:
            return

        self._current_session_id = session_id
        self._history_mgr.set_current_session(session_id)
        self._load_current_history()
        self.session_changed.emit(session_id)

    def get_current_session_id(self) -> str:
        return self._current_session_id

    def list_sessions(self) -> list[dict]:
        return self._history_mgr.list_sessions()

    def rename_session(self, session_id: str, new_title: str):
        self._history_mgr.rename_session(session_id, new_title)
        self.session_list_changed.emit()

    # 历史数据操作
    def add_message(self, message: dict):
        """添加一条消息（自动更新元数据）"""
        self._history.append(message)
        self._update_metadata()
        self._save()
        self.history_updated.emit()

    def update_message_content(self, message_id: str, new_content: str):
        self._history_mgr.update(self._current_session_id, message_id, new_content)
        # 同步更新内存中的 _history
        for msg in self._history:
            if msg.get("message_id") == message_id:
                msg["content"] = new_content
                break
        self._update_metadata()
        self.history_updated.emit()

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def clear_current_session(self):
        self._history.clear()
        self._update_metadata()
        self._save()

    def delete_session(self, session_id: str):
        self._history_mgr.delete(session_id)
        if session_id == self._current_session_id:
            self.create_new_session()
        self.session_list_changed.emit()

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
        self._update_metadata()
        self._save()
        self.history_updated.emit()

    def _remove_message_by_index(self, index: int):
        """内部删除指定索引的消息"""
        if 0 <= index < len(self._history):
            del self._history[index]
            self._update_metadata()
            self._save()
            self.history_updated.emit()

    def _load_current_history(self):
        raw_data = self._history_mgr.load(self._current_session_id)
        if raw_data and isinstance(raw_data[0], dict) and "title" in raw_data[0]:
            self._history = raw_data[1:]  # 去掉元数据
        else:
            self._history = raw_data

    def _save(self):
        # 保存时把元数据 + 消息一起存
        full_data = [self._get_current_metadata()] + self._history
        self._history_mgr.save(self._current_session_id, full_data)

    def _get_current_metadata(self) -> dict:
        sessions = self.list_sessions()
        for s in sessions:
            if s["session_id"] == self._current_session_id:
                return {
                    "session_id": self._current_session_id,
                    "title": s["title"],
                    "created_at": s.get("created_at"),
                    "updated_at": datetime.now().isoformat(),
                    "message_count": len(self._history)
                }
        # fallback
        return {
            "session_id": self._current_session_id,
            "title": "新会话",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "message_count": len(self._history)
        }

    def _update_metadata(self):
        # 每次修改后更新 updated_at 和 message_count
        pass