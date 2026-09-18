import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class Job:
    def __init__(self, job_id: str):
        self.id = job_id
        self.status = JobStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.updated_at = time.time()


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
    def submit(cls, coro_factory: Callable[[], Awaitable[Any]]) -> Job:
        cls._cleanup()

        job = Job(str(uuid.uuid4()))
        cls._jobs[job.id] = job

        def _run_in_thread():
            job.status = JobStatus.RUNNING
            try:
                job.result = asyncio.run(coro_factory())
                job.status = JobStatus.DONE
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
    def _cleanup(cls):
        now = time.time()
        stale = [
            jid for jid, job in cls._jobs.items()
            if job.status in (JobStatus.DONE, JobStatus.ERROR)
            and now - job.updated_at > cls._max_age_seconds
        ]
        for jid in stale:
            cls._jobs.pop(jid, None)
