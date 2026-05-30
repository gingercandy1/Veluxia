import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from core.preloader import preloader
from src.backend.core.model_utils import huggingface_token
from src.shared.settings import PROJECT_ROOT


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: str(datetime.now()))
    embedding: Optional[np.ndarray] = None   # 懒计算


@dataclass
class TopicSegment:
    """单一话题段：一组连续的、主题相近的消息"""
    segment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: List[Message] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None    # 段质心（所有消息 embedding 均值）
    summary: Optional[str] = None            # 段被压缩后的摘要
    is_compressed: bool = False


class EmbedModel:
    _instances = {}  # 简单缓存不同模型

    def __new__(cls, embed_model: str = "Qwen/Qwen3-Embedding-0.6B", embed_device: str = "cpu"):
        key = (embed_model, embed_device)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            cls._instances[key].initialized = False
        return cls._instances[key]

    def __init__(self, embed_model: str = "Qwen/Qwen3-Embedding-0.6B", embed_device: str = "cpu"):
        if getattr(self, 'initialized', False):
            return
        self.initialized = True

        self.model_dir = Path(os.path.join(PROJECT_ROOT, "models", embed_model.split("/")[-1]))
        if not self.model_dir.exists():
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=embed_model,
                              local_dir=str(self.model_dir),
                              token=huggingface_token)

        self._embed_model = self.sentence_transformers.SentenceTransformer(model_name_or_path=str(self.model_dir), device=embed_device)

    @property
    def embed_model(self):
       return self._embed_model

    @property
    def sentence_transformers(self):
        return preloader.get("sentence_transformers")

    def encode(self, text: str) -> np.ndarray:
        return np.array(self._embed_model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def get_embedding_dimension(self) -> int | None:
        return self._embed_model.get_embedding_dimension()



class SessionStore:
    """
    负责将会话数据持久化到 DB，以及从中恢复。
    """
    _instances: Dict[tuple, "SessionStore"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        embed_model,
        storage_dir: str="config/memory_store/qdrant",
        collection_name: str = "sessions"
    ):
        key = (storage_dir, collection_name)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(embed_model, storage_dir, collection_name)
            return cls._instances[key]

    def __init__(self,
                 embed_model,
                 storage_dir: str,
                 collection_name: str = "sessions"):
        self.storage_dir = Path(os.path.join(PROJECT_ROOT, storage_dir))
        os.makedirs(self.storage_dir, exist_ok=True)
        self.collection_name = collection_name
        self._embed_model = embed_model

        self._client = self.qdrant_client.QdrantClient(path=str(self.storage_dir))
        self._ensure_collection()

    @property
    def qdrant_client(self):
        return preloader.get("qdrant_client")

    def _ensure_collection(self):
        dim = self._embed_model.get_embedding_dimension()

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.qdrant_client.models.VectorParams(
                    size=dim,
                    distance=self.qdrant_client.models.Distance.COSINE,
                ),
            )

    def delete_session(self, user_id: str, session_id: str) -> int:
        """删除某个会话的全部消息，返回删除数量"""
        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.qdrant_client.models.Filter(must=[
                self.qdrant_client.models.FieldCondition(
                    key="user_id",
                    match=self.qdrant_client.models.MatchValue(value=user_id)
                ),
                self.qdrant_client.models.FieldCondition(
                    key="session_id",
                    match=self.qdrant_client.models.MatchValue(value=session_id)
                ),
            ]),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            return 0
        ids = [p.id for p in results]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self.qdrant_client.models.PointIdsList(points=ids),
        )
        return len(ids)

    def delete_segment(self, session_id: str, segment_id: str) -> int:
        """删除某个话题段"""
        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.qdrant_client.models.Filter(must=[
                self.qdrant_client.models.FieldCondition(
                    key="session_id",
                    match=self.qdrant_client.models.MatchValue(value=session_id)
                ),
                self.qdrant_client.models.FieldCondition(
                    key="segment_id",
                    match=self.qdrant_client.models.MatchValue(value=segment_id)
                ),
            ]),
            limit=10000,
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            return 0
        ids = [p.id for p in results]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self.qdrant_client.models.PointIdsList(points=ids),
        )
        return len(ids)

    def delete_all_user_data(self, user_id: str) -> int:
        """删除某个用户的全部数据"""
        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.qdrant_client.models.Filter(must=[
                self.qdrant_client.models.FieldCondition(
                    key="user_id",
                    match=self.qdrant_client.models.MatchValue(value=user_id)
                ),
            ]),
            limit=100000,
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            return 0
        ids = [p.id for p in results]
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=self.qdrant_client.models.PointIdsList(points=ids),
        )
        return len(ids)

    def save_message(
        self,
        msg: Message,
        seg: TopicSegment,
        msg_index: int,
        user_id: str,
        session_id: str,
    ) -> None:
        # 用 msg_index 做穩定的 int id（qdrant point id 必須是 uint64 或 UUID）
        point_id = self._make_point_id(session_id, msg_index)

        self._client.upsert(
            collection_name=self.collection_name,
            points=[self.qdrant_client.models.PointStruct(
                id=point_id,
                vector=msg.embedding.tolist(),
                payload={
                    "user_id": user_id,
                    "session_id": session_id,
                    "segment_id": seg.segment_id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "msg_index": msg_index,
                    "segment_summary": seg.summary or "",
                    "is_compressed": seg.is_compressed,
                },
            )],
        )

    def update_segment_summary(
        self,
        seg: TopicSegment,
        session_id: str,
    ) -> None:
        results, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.qdrant_client.models.Filter(must=[
                self.qdrant_client.models.FieldCondition(key="session_id", match=self.qdrant_client.models.MatchValue(value=session_id)),
                self.qdrant_client.models.FieldCondition(key="segment_id", match=self.qdrant_client.models.MatchValue(value=seg.segment_id)),
            ]),
            limit=1000,
            with_payload=False,
            with_vectors=False,
        )
        if not results:
            return

        ids = [p.id for p in results]
        self._client.set_payload(
            collection_name=self.collection_name,
            payload={
                "segment_summary": seg.summary or "",
                "is_compressed": seg.is_compressed,
            },
            points=ids,
        )

    def load_segments(
        self,
        user_id: str,
        session_id: str,
    ) -> List[TopicSegment]:
        all_points = []
        offset = None

        while True:
            results, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self.qdrant_client.models.Filter(must=[
                    self.qdrant_client.models.FieldCondition(key="user_id",    match=self.qdrant_client.models.MatchValue(value=user_id)),
                    self.qdrant_client.models.FieldCondition(key="session_id", match=self.qdrant_client.models.MatchValue(value=session_id)),
                ]),
                limit=500,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            all_points.extend(results)
            if next_offset is None:
                break
            offset = next_offset

        if not all_points:
            return []

        all_points.sort(key=lambda p: p.payload.get("msg_index", 0))
        seg_map: Dict[str, TopicSegment] = {}
        for point in all_points:
            pl = point.payload
            seg_id = pl["segment_id"]

            if seg_id not in seg_map:
                seg = TopicSegment(segment_id=seg_id)
                seg.summary = pl.get("segment_summary") or None
                seg.is_compressed = bool(pl.get("is_compressed", False))
                seg_map[seg_id] = seg

            seg_map[seg_id].messages.append(Message(
                role=pl["role"],
                content=pl["content"],
                timestamp=pl["timestamp"],
                embedding=np.array(point.vector, dtype=np.float32),
            ))

        return sorted(
            seg_map.values(),
            key=lambda s: s.messages[0].timestamp if s.messages else "",
        )

    @staticmethod
    def _make_point_id(session_id: str, msg_index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}_{msg_index}"))



class ConversationMemory:
    DEFAULT_USER = "default"
    DEFAULT_SESSION = "default"
    _instances: Dict[tuple, "ConversationMemory"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, user_id: str, session_id: str, summarizer_fn=None):
        key = (user_id, session_id)
        with cls._lock:
            if key not in cls._instances:
                embed_model = EmbedModel()
                store = SessionStore.get_instance(embed_model=embed_model)
                cls._instances[key] = cls(
                    embed_model=embed_model,
                    user_id=user_id,
                    session_id=session_id,
                    store=store,
                    summarizer_fn=summarizer_fn
                )
            return cls._instances[key]

    def switch_session(self, user_id: str, session_id: str):
        with self._lock:
            self.user_id = user_id
            self.session_id = session_id
            self.segments = self._init_segments()

    def __init__(
            self,
            embed_model,
            user_id: str = "default",
            session_id: str = "default",
            topic_shift_threshold: float = 0.35,  # 余弦距离超过此值 = 话题切换
            max_segment_size: int = 12,  # 单段消息数超限触发摘要
            recent_window: int = 6,  # 当前段无条件保留的最近消息数
            recency_weight: float = 0.4,  # 混合评分中 recency 权重 α
            summarizer_fn=None,  # 可注入 LLM 摘要函数
            store: Optional[SessionStore] = None,   # 可选注入，None 则纯内存
    ):
        self._embed_model = embed_model
        self.user_id = user_id
        self.session_id = session_id
        self.threshold = topic_shift_threshold
        self.max_segment_size = max_segment_size
        self.recent_window = recent_window
        self.alpha = recency_weight
        self._summarize = summarizer_fn or self._default_summarizer
        self._store = store

        self.segments: List[TopicSegment] = self._init_segments()
        print("✅ bge-m3 记忆系统初始化完成")

    def _init_segments(self) -> List[TopicSegment]:
        if self._store:
            segments = self._store.load_segments(self.user_id, self.session_id)
            if segments:
                for seg in segments:
                    self._update_centroid(seg)
                print(f"✅ 已恢复会话 [{self.session_id}]，"
                      f"共 {sum(len(s.messages) for s in segments)} 条消息")
                return segments
        return [TopicSegment()]

    def clear_session(self):
        """清空当前会话，内存和向量库同步清除"""
        if self._store:
            self._store.delete_session(self.user_id, self.session_id)

        # 内存也重置
        self.segments = [TopicSegment()]
        print(f"✅ 会话 [{self.session_id}] 已清空")

    def remove_segment(self, segment_id: str):
        """删除指定话题段"""
        if self._store:
            self._store.delete_segment(self.session_id, segment_id)
        self.segments = [s for s in self.segments if s.segment_id != segment_id]
        if not self.segments:
            self.segments = [TopicSegment()]

    @staticmethod
    def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(1.0 - np.dot(a, b) / denom) if denom > 1e-9 else 1.0

    def _update_centroid(self, seg: TopicSegment):
        embs = [m.embedding for m in seg.messages if m.embedding is not None]
        if embs:
            seg.centroid = np.mean(embs, axis=0)

    #  话题漂移检测
    def _is_topic_shift(self, emb: np.ndarray) -> bool:
        seg = self.current_segment
        if seg.centroid is None or len(seg.messages) < 2:
            return False
        return self._cosine_distance(seg.centroid, emb) > self.threshold

    #  核心写入
    @property
    def current_segment(self) -> TopicSegment:
        return self.segments[-1]

    def add(self, role: str, content: str):
        emb = self._embed_model.encode(content)
        msg = Message(role=role, content=content, embedding=emb)

        if self._is_topic_shift(emb):
            self.segments.append(TopicSegment())

        seg = self.current_segment
        msg_index = sum(len(s.messages) for s in self.segments)
        seg.messages.append(msg)
        self._update_centroid(seg)

        if self._store:
            self._store.save_message(msg, seg, msg_index, self.user_id, self.session_id)

        if len(seg.messages) > self.max_segment_size:
            self._compress_segment(seg)

    def add_turn(self, user_msg: str, bot_reply: str):
        self.add("user", user_msg)
        self.add("assistant", bot_reply)

    #  渐进摘要
    def _compress_segment(self, seg: TopicSegment):
        keep_idx = len(seg.messages) - self.recent_window
        old_messages = seg.messages[:keep_idx]
        seg.messages = seg.messages[keep_idx:]

        old_text = "\n".join(f"[{m.role}]: {m.content}" for m in old_messages)
        if seg.summary:
            old_text = f"[Existing summary]\n{seg.summary}\n\n[New content]\n{old_text}"

        seg.summary = self._summarize(old_text)
        seg.is_compressed = True

        if self._store:
            self._store.update_segment_summary(seg, self.session_id)

    def _default_summarizer(self, text: str) -> str:
        lines = [l for l in text.split("\n") if l.strip()]
        return f"[Summary of {len(lines)} messages:{text}"

    def build_memory_context(self, query: str, long_term_limit: int = 4) -> str:
        lines: List[str] = []

        recent = self.current_segment.messages[-self.recent_window:]
        if recent:
            lines.append("[Recent conversation]")
            for m in recent:
                lines.append(f"  {'User' if m.role == 'user' else 'Assistant'}: {m.content}")
            lines.append("")

        if self.current_segment.summary:
            lines.append("[Current topic summary]")
            lines.append(f"  {self.current_segment.summary}")
            lines.append("")

        other_segs =  [s for s in self.segments[:-1] if s.centroid is not None]
        if other_segs:
            q_emb = self._embed_model.encode(query)
            scored = sorted(
                [(s, 1 - self._cosine_distance(s.centroid, q_emb))
                 for s in other_segs if s.centroid is not None],
                key=lambda x: x[1],
                reverse=True,
            )
            relevant = [(s, score) for s, score in scored
                        if score > 0.6][:long_term_limit]

            if relevant:
                lines.append("[Related past topics]")
                for seg, score in relevant:
                    summary = seg.summary or " | ".join(
                        f"{m.role}: {m.content[:40]}"
                        for m in seg.messages[:2]
                    )
                    lines.append(f"  {summary[:200]}")
                lines.append("")

        return "\n".join(lines)

    def stats(self) -> Dict:
        return {
            "session_id": self.session_id,
            "total_segments": len(self.segments),
            "current_segment_messages": len(self.current_segment.messages),
            "compressed_segments": sum(1 for s in self.segments if s.is_compressed),
            "has_store": self._store is not None,
        }