import json
import threading
import time
import httpx
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Callable

from PySide6.QtCore import QCoreApplication

from src.shared.schemas import BaseResponse, ImageResponse, TextResponse, AnimationResponse, SpeechResponse, \
    ModelInfoResponse, TranslateResponse, RefineResponse, SpriteSheetResponse, TranscriptionResponse, \
    CreatePackRequest, PackListResponse, PackResponse, TemplateListResponse, \
    ApproveStepRequest, ResetStepRequest, DraftItemsRequest, DraftItemsResponse

_DEFAULT_LIMITS = httpx.Limits(
    max_connections=10,
    max_keepalive_connections=5,
    keepalive_expiry=30.0,
)


class BackendStatus(str, Enum):
    OK = "ok"
    UNREACHABLE = "unreachable"
    UNAUTHORIZED = "unauthorized"


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def probe_backend(base_url: str, token: str = "", timeout: float = 5.0) -> BackendStatus:
    """区分"连不上"和"token 错"，前者等一会儿可能恢复，后者要提示用户改设置。

    /health 不需要 token，只能证明进程活着；所以探测需要鉴权的 /ready。
    """
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/ready",
                         headers=auth_headers(token), timeout=timeout)
    except httpx.HTTPError:
        return BackendStatus.UNREACHABLE
    if resp.status_code == 401:
        return BackendStatus.UNAUTHORIZED
    return BackendStatus.OK if resp.is_success else BackendStatus.UNREACHABLE


class ApiClient:
    """
    封装对 Backend FastAPI 的所有 HTTP 调用。
    实例化一次后可在整个 UI 生命周期内复用。
    所有调用均为同步阻塞，应在 Qt Worker Thread 内使用。
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        timeout: int = 300,
    ) -> None:
        if hasattr(self, "initialize"):
            return

        self.initialize = True
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = httpx.Client(timeout=timeout, limits=_DEFAULT_LIMITS)
        self._model_info = {}

    def close(self):
        self._session.close()

    def configure(self, base_url: str, token: str = "") -> None:
        """地址和 token 由启动流程按设置决定（本机 / 远程），单例构造时还不知道。"""
        self.base_url = base_url.rstrip("/")
        self._session.headers.pop("Authorization", None)
        self._session.headers.update(auth_headers(token))

    def _post(self, path: str, payload: Dict[str, Any], response_cls=BaseResponse) -> BaseResponse:
        try:
            resp = self._session.post(
                f"{self.base_url}{path}",
                json=payload,
                timeout=self._timeout ,
            )
            resp.raise_for_status()
            return response_cls.model_validate(resp.json())  # 直接反序列化
        except httpx.ConnectError:
            return response_cls.from_error(QCoreApplication.translate("ApiClient",
                "Cannot connect to the backend. Please check that the service is running."))
        except httpx.ConnectTimeout:
            return response_cls.from_error(
                QCoreApplication.translate("ApiClient", "Connection timed out."))
        except httpx.ReadTimeout:
            return response_cls.from_error(QCoreApplication.translate("ApiClient",
                "The backend timed out; the task may still be running."))
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return response_cls.from_error(detail)
        except Exception as exc:
            return response_cls.from_error(str(exc))

    def _get(self, path: str, response_cls=BaseResponse) -> BaseResponse:
        try:
            resp  = self._session.get(f"{self.base_url}{path}", timeout=30)
            resp.raise_for_status()
            return response_cls.model_validate(resp.json())
        except Exception as exc:  # noqa: BLE001
            return response_cls.from_error(str(exc))

    def _delete(self, path: str, response_cls=BaseResponse) -> BaseResponse:
        try:
            resp = self._session.request("DELETE", f"{self.base_url}{path}", timeout=30)
            resp.raise_for_status()
            return response_cls.model_validate(resp.json())
        except httpx.ConnectError:
            return response_cls.from_error(
                QCoreApplication.translate("ApiClient", "Cannot connect to the backend."))
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return response_cls.from_error(detail)
        except Exception as exc:
            return response_cls.from_error(str(exc))

    def health(self, retries: int = 1) -> bool:
        for _ in range(max(retries, 1)):
            try:
                resp = self._session.get(f"{self.base_url}/health", timeout=5)
                if resp.is_success:
                    return True
            except Exception:
                pass
        return False

    def ready(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/ready", timeout=5)
            if resp.is_success:
                return bool(resp.json().get("ready"))
        except Exception:
            pass
        return False

    def generate_image(self, req, stop_event: Optional[threading.Event] = None, on_partial=None):
        payload = req.to_api_payload()
        return self._submit_and_poll(
            "/image/generate", payload, ImageResponse, stop_event=stop_event,
            on_partial=on_partial, poll_interval=1.0 if on_partial else 2.0,
        )

    def generate_text(self, req):
        payload = req.to_api_payload()
        return self._post("/text/generate", payload, TextResponse)

    def generate_animation(self, req, stop_event: Optional[threading.Event] = None):
        payload = req.to_api_payload()
        return self._submit_and_poll("/animation/generate", payload, AnimationResponse, stop_event=stop_event)

    def generate_speech(self, req, stop_event: Optional[threading.Event] = None):
        payload = req.to_api_payload()
        return self._submit_and_poll("/speech/generate", payload, SpeechResponse, stop_event=stop_event)

    def transcribe(self, req, stop_event: Optional[threading.Event] = None):
        payload = req.to_api_payload()
        return self._submit_and_poll("/transcription/generate", payload, TranscriptionResponse,
                                     stop_event=stop_event)

    def cancel_job(self, path_prefix: str, job_id: str) -> None:
        """通知后端取消一个已提交的任务；这是尽力而为，不等待/不关心结果——
        前端反正马上就要放弃这次结果了，等待取消确认只会拖慢"点停止就立刻停"的观感。"""
        try:
            self._session.post(f"{self.base_url}{path_prefix}/cancel/{job_id}", timeout=10)
        except Exception:
            pass

    @staticmethod
    def _interruptible_sleep(seconds: float, stop_event: Optional[threading.Event]):
        if stop_event is None:
            time.sleep(seconds)
            return
        stop_event.wait(timeout=seconds)

    def _submit_and_poll(
        self,
        path_prefix: str,
        payload: Dict[str, Any],
        response_cls,
        poll_interval: float = 2.0,
        poll_timeout: float = 60.0,
        max_consecutive_connect_failures: int = 5,
        max_total_wait_seconds: float = 2 * 3600,
        stop_event: Optional[threading.Event] = None,
        on_partial=None,
    ) -> BaseResponse:
        """
        图片/动画/语音生成耗时不固定（从几秒到几十分钟不等），不能再用一次性
        阻塞 HTTP 请求 + 固定超时的模式（超时不代表生成失败，只是客户端等不及了，
        但那样会导致 loading 状态被过早收起，而后端其实还在继续跑）。

        改为：先提交任务拿 job_id（这一步很快，用短超时即可），然后轮询任务状态，
        不管后端实际跑多久，前端只在拿到真正的 done/error 时才结束等待。

        注意：加载模型/推理这类同步 CPU 密集操作是在后端线程池里跑的，Python 的
        GIL 会导致它偶尔把 FastAPI 主事件循环饿一下，使某次轮询请求读超时——这不
        代表任务失败或后端挂了。只有真正的 ConnectError（连接不上，说明后端进程
        可能已经崩溃/退出）才计入失败次数；其它异常（读超时等）无限重试，只用一个
        很宽松的总等待时长兜底，防止真出问题时无限等下去。
        """
        try:
            resp = self._session.post(
                f"{self.base_url}{path_prefix}/submit",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
        except httpx.ConnectError:
            return response_cls.from_error(QCoreApplication.translate("ApiClient",
                "Cannot connect to the backend. Please check that the service is running."))
        except Exception as exc:
            return response_cls.from_error(str(exc))

        connect_failures = 0
        seen_partial = 0
        start = time.monotonic()
        while True:
            self._interruptible_sleep(poll_interval, stop_event)
            if stop_event is not None and stop_event.is_set():
                self.cancel_job(path_prefix, job_id)
                return response_cls.from_error(
                    QCoreApplication.translate("ApiClient", "Generation stopped."))
            if time.monotonic() - start > max_total_wait_seconds:
                return response_cls.from_error(QCoreApplication.translate("ApiClient",
                    "Timed out waiting for the generation task. Please check the backend log."))

            try:
                resp = self._session.get(
                    f"{self.base_url}{path_prefix}/status/{job_id}",
                    timeout=poll_timeout,
                )
                resp.raise_for_status()
                status_json = resp.json()
            except httpx.ConnectError:
                connect_failures += 1
                if connect_failures >= max_consecutive_connect_failures:
                    return response_cls.from_error(QCoreApplication.translate("ApiClient",
                        "Lost connection to the backend; the generation result is unknown."))
                continue
            except Exception:
                # 读超时/临时解析失败等：后端大概率还活着，只是这次响应慢，继续等。
                continue

            connect_failures = 0
            status = status_json.get("status")
            partial = status_json.get("partial") or []
            if on_partial is not None and len(partial) > seen_partial:
                new_items, seen_partial = partial[seen_partial:], len(partial)
                on_partial(new_items)
            if status == "done":
                return response_cls.model_validate(status_json.get("result") or {})
            if status == "error":
                return response_cls.from_error(
                    status_json.get("error")
                    or QCoreApplication.translate("ApiClient", "Generation failed."))
            # pending / running：继续等待

    def stream_text(self, req, stop_event: Optional[threading.Event] = None) -> Generator[Dict[str, Any], None, None]:
        max_retries = 3
        payload = req.to_api_payload()

        attempt = 0
        while attempt <= max_retries:
            try:
                with self._session.stream(
                    "POST",
                    f"{self.base_url}/text/stream",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines():
                        if stop_event is not None and stop_event.is_set():
                            # 文本流没有 job_id 可取消，前端能做的只是主动断开连接——
                            # 退出 with 块会关闭这次 HTTP 流，后端那次请求随之中止。
                            yield {"type": "cancelled"}
                            return
                        if not raw:
                            continue
                        if not raw.startswith("data: "):
                            continue
                        data = raw[6:]
                        if data == "[DONE]":
                            return
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            yield {"type": "text", "text": data}
                return

            except (httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                attempt += 1
                if attempt > max_retries:
                    yield {"type": "error", "text": QCoreApplication.translate("ApiClient",
                        "Connection lost (retried {0} times): {1}").format(max_retries, exc)}
                else:
                    yield {"type": "error", "text": QCoreApplication.translate("ApiClient",
                        "Connection lost, retry {0}...").format(attempt)}

            except httpx.ConnectError as exc:
                yield {"type": "error", "text": QCoreApplication.translate("ApiClient",
                    "Cannot connect to the backend: {0}").format(exc)}
                return

            except Exception as exc:
                yield {"type": "error", "text": str(exc)}
                return

    def clear_memory(self, session_id: str, user_id: str = "default") -> BaseResponse:
        return self._delete(f"/text/memory/session/{session_id}?user_id={user_id}")

    def translate(self, req, is_default=True):
        payload = req.to_api_payload()
        if is_default:
            return self._post("/translate/generate", payload, TranslateResponse)
        else:
            return self._post("/translate/default", payload, TranslateResponse)

    def refine_prompt(self, req, mode: str) -> RefineResponse:
        payload = req.to_api_payload()
        payload["extra"]["content"] = req.prompt
        payload["extra"]["mode"] = mode
        return self._post("/prompt/refine", payload, RefineResponse)

    def export_sprites(self, frame_paths: list[str], name: str, **options) -> SpriteSheetResponse:
        """把序列帧导出为精灵图 + atlas + 编号 PNG；纯 CPU 操作，后端不占显存。"""
        payload = {
            "model_name": "SpriteSheet",
            "extra": {"frame_paths": frame_paths, "name": name, **options},
        }
        return self._post("/image_frame/export", payload, SpriteSheetResponse)

    def download_media(self, media_url: str, cache_dir: Path) -> Optional[str]:
        """
        把后端 /media/... 相对 URL 拉取到本地缓存目录，返回本地文件路径。
        用于将展示层与后端的实际存储位置解耦——即使以后 backend 独立部署/远程运行，
        前端也只需要走 HTTP 拿文件，不再假设和 backend 共享文件系统。
        已缓存过的同名文件直接复用，不重复下载。
        """
        if not media_url:
            return None
        if not media_url.startswith("/media/"):
            # 兼容旧响应里直接给本地绝对路径的情况
            return media_url

        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / Path(media_url).name
        if local_path.exists():
            return str(local_path)

        try:
            resp = self._session.get(f"{self.base_url}{media_url}", timeout=self._timeout)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            return str(local_path)
        except Exception as exc:
            print(f"⚠️ 媒体下载失败 {media_url}: {exc}")
            return None

    def get_model_info(self, factory_type_str):
        _model_info = self._get(f"/{factory_type_str}/models", response_cls=ModelInfoResponse)
        return _model_info

    # 资料库（ADR 0004）
    def list_templates(self) -> TemplateListResponse:
        return self._get("/library/templates", response_cls=TemplateListResponse)

    def list_packs(self) -> PackListResponse:
        return self._get("/library/packs", response_cls=PackListResponse)

    def get_pack(self, pack_id: str) -> PackResponse:
        return self._get(f"/library/packs/{pack_id}", response_cls=PackResponse)

    def create_pack(self, request: CreatePackRequest) -> PackResponse:
        return self._post("/library/packs", request.model_dump(), PackResponse)

    def draft_items(self, request: DraftItemsRequest) -> DraftItemsResponse:
        """首次调用要加载 LLM，较慢；仍在默认 300 秒超时之内。"""
        return self._post("/library/drafts", request.model_dump(), DraftItemsResponse)

    def delete_pack(self, pack_id: str) -> BaseResponse:
        return self._delete(f"/library/packs/{pack_id}")

    def run_pack(self, pack_id: str, stop_event: Optional[threading.Event] = None) -> PackResponse:
        """整包执行可能跑一整夜；进度以 manifest 为准，由界面另行轮询 get_pack。"""
        return self._submit_and_poll("/library/run", {"pack_id": pack_id}, PackResponse,
                                     stop_event=stop_event, max_total_wait_seconds=24 * 3600)

    def approve_step(self, pack_id: str, request: ApproveStepRequest) -> PackResponse:
        return self._post(f"/library/packs/{pack_id}/approve", request.model_dump(), PackResponse)

    def reset_step(self, pack_id: str, request: ResetStepRequest) -> PackResponse:
        return self._post(f"/library/packs/{pack_id}/reset", request.model_dump(), PackResponse)

    def fetch_media(self, media_url: str) -> bytes:
        """直接取媒体字节，不走 download_media 的按文件名缓存：
        资源包里每个条目的产物同名（如 trim.png），重跑后内容也会变，按名缓存会串图或过期。"""
        resp = self._session.get(f"{self.base_url}{media_url}", timeout=self._timeout)
        resp.raise_for_status()
        return resp.content





class ApiGuardClient:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8756",
        timeout: int = 300,
    ) -> None:
        if hasattr(self, "initialize"):
            return

        self.initialize = True
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = httpx.Client(timeout=timeout, limits=_DEFAULT_LIMITS)

    def detect_device(self):
        try:
            resp = self._session.get(f"{self.base_url}/system/detect-device", timeout=5)
            if resp.is_success:
                return resp.json()
        except Exception:
            pass
        return {}

    def health(self, retries: int = 1) -> bool:
        for _ in range(max(retries, 1)):
            try:
                resp = self._session.get(f"{self.base_url}/health", timeout=5)
                if resp.is_success:
                    return True
            except Exception:
                pass
        return False

    def install_backend(self, extra: str = "cpu", on_progress: Optional[Callable[[str], None]] = None):
        """
        触发后端安装，并实时返回安装日志
        Args:
            extra: cpu 或 cuda
            on_progress: 进度回调函数，每收到一行日志都会调用
        """
        try:
            # 1. send install request
            payload = {"extra": extra}
            resp = self._session.post(
                f"{self.base_url}/system/install-backend",
                json=payload,
                timeout=10.0
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") == "already_running":
                if on_progress:
                    on_progress("⚠️ " + QCoreApplication.translate(
                        "ApiClient", "An install task is already running...") + "\n")
                return result

            if on_progress:
                on_progress("✅ " + QCoreApplication.translate(
                    "ApiClient", "Install request accepted: {0}").format(result.get("message")) + "\n")

            # 2. check installation detail information
            self._poll_install_status(on_progress)
            return result

        except httpx.ConnectError:
            error_msg = QCoreApplication.translate("ApiClient",
                "Cannot connect to the backend. Please check that the service is running.")
            if on_progress:
                on_progress(f"❌ {error_msg}\n")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = str(e)
            if on_progress:
                on_progress("❌ " + QCoreApplication.translate(
                    "ApiClient", "Request failed: {0}").format(error_msg) + "\n")
            raise

    def _poll_install_status(self, on_progress: Optional[Callable[[str], None]] = None):
        """轮询安装进度"""
        if not on_progress:
            return

        max_wait = 1800  # 最长等待30分钟
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                resp = self._session.get(
                    f"{self.base_url}/system/install-status",
                    timeout=8.0
                )
                resp.raise_for_status()
                data = resp.json()

                status = data.get("status")
                log = data.get("log", "")
                message = data.get("message", "")

                if log and on_progress:
                    on_progress(log)

                if status in ("done", "failed"):
                    if on_progress:
                        if status == "done":
                            on_progress(f"🎉 {message}\n")
                        else:
                            on_progress(f"❌ {message}\n")
                    break
                time.sleep(0.6)
            except Exception:
                time.sleep(1.0)

        else:
            if on_progress:
                on_progress("⚠️ " + QCoreApplication.translate("ApiClient",
                    "Install monitoring timed out (over 30 minutes).") + "\n")