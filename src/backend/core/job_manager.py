import asyncio
import threading
import time
import uuid
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

from src.backend.core.exceptions import GenerationCancelled


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status = JobStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.updated_at = time.time()
        # 生成器在自己的推理循环里轮询这个 event 来响应取消请求。
        self.cancel_event = threading.Event()


class JobManager:
    """
    通用后台任务管理器：生成任务耗时不固定（从几秒到几十分钟都有可能），
    不能再依赖一次性阻塞 HTTP 请求 + 客户端超时的模式。

    提交后立即返回 job_id；真正的生成逻辑在独立线程里跑（内部用 asyncio.run
    驱动那些"名义上是 async def、实际是同步阻塞 GPU 调用"的 generate() 协程），
    不占用 FastAPI 主事件循环，前端可以随时轮询 /status 拿最新进度而不被卡住。
    """

    _jobs: Dict[str, Job] = {}
    _max_age_seconds = 3600

    @classmethod
    def submit(cls, coro_factory: Callable[[Job], Awaitable[Any]]) -> Job:
        cls._cleanup()

        job = Job(str(uuid.uuid4()))
        cls._jobs[job.id] = job

        def _run_in_thread():
            if job.cancel_event.is_set():
                # 还没排上执行就被取消了（PENDING 阶段），不必再跑一遍生成。
                job.status = JobStatus.CANCELLED
                job.updated_at = time.time()
                return
            job.status = JobStatus.RUNNING
            try:
                result = asyncio.run(coro_factory(job))
                if job.cancel_event.is_set():
                    # 取消请求在生成器完全跑完之后才被响应（比如语音模型没有
                    # 可中断的检查点）：结果照样算出来了，但既然用户已经点了
                    # 停止，就不把它当成"成功"返回，直接丢弃。
                    job.status = JobStatus.CANCELLED
                else:
                    job.result = result
                    job.status = JobStatus.DONE
            except GenerationCancelled:
                job.status = JobStatus.CANCELLED
            except Exception as exc:
                # FastAPI/Starlette 的 HTTPException 不把 detail 塞进 args，
                # str(exc) 拿不到真正的错误信息，要单独取 .detail。
                job.error = getattr(exc, "detail", None) or str(exc) or exc.__class__.__name__
                job.status = JobStatus.ERROR
            finally:
                job.updated_at = time.time()

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _run_in_thread)
        return job

    @classmethod
    def get(cls, job_id: str) -> Optional[Job]:
        return cls._jobs.get(job_id)

    @classmethod
    def cancel(cls, job_id: str) -> bool:
        """
        请求取消一个任务。只是设置标记——真正的中断发生在生成器自己检查
        cancel_event 的地方（diffusers 去噪步回调 / 显式 check_cancelled()）。
        对已经跑到没有检查点的库（比如某些语音模型），后端会继续算完，
        但结果会被前端丢弃、轮询会立刻拿到 cancelled 状态而不用干等。
        """
        job = cls._jobs.get(job_id)
        if job is None:
            return False
        if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            return False
        job.cancel_event.set()
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
        else:
            job.status = JobStatus.CANCELLING
        job.updated_at = time.time()
        return True

    @classmethod
    def _cleanup(cls):
        now = time.time()
        stale = [
            jid for jid, job in cls._jobs.items()
            if job.status in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED)
            and now - job.updated_at > cls._max_age_seconds
        ]
        for jid in stale:
            cls._jobs.pop(jid, None)
