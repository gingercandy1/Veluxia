import json
import time
import httpx
from typing import Any, Dict, Generator, Optional, Callable

from src.shared.schemas import BaseResponse, ImageResponse, TextResponse, AnimationResponse, SpeechResponse, \
    ModelInfoResponse, TranslateResponse

_DEFAULT_LIMITS = httpx.Limits(
    max_connections=10,
    max_keepalive_connections=5,
    keepalive_expiry=30.0,
)


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
            return response_cls.from_error("無法連接後端，請確認服務是否啟動")
        except httpx.ConnectTimeout:
            return response_cls.from_error("連接超時")
        except httpx.ReadTimeout:
            return response_cls.from_error("後端響應超時，任務可能仍在運行")
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
            return response_cls.from_error("無法連接後端")
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

    def generate_image(self, req):
        payload = req.to_api_payload()
        return self._post("/image/generate", payload, ImageResponse)

    def generate_text(self, req):
        payload = req.to_api_payload()
        return self._post("/text/generate", payload, TextResponse)

    def generate_animation(self, req):
        payload = req.to_api_payload()
        return self._post("/animation/generate", payload, AnimationResponse)

    def generate_speech(self, req):
        payload = req.to_api_payload()
        return self._post("/speech/generate", payload, SpeechResponse)

    def stream_text(self, req) -> Generator[Dict[str, Any], None, None]:
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
                    yield {"type": "error", "text": f"連接中斷（已重試{max_retries}次）: {exc}"}
                else:
                    yield {"type": "error", "text": f"連接中斷，重試第{attempt}次..."}

            except httpx.ConnectError as exc:
                yield {"type": "error", "text": f"無法連接後端: {exc}"}
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


    def get_model_info(self, factory_type_str):
        _model_info = self._get(f"/{factory_type_str}/models", response_cls=ModelInfoResponse)
        return _model_info





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
                    on_progress("⚠️ 安装任务已在运行中...\n")
                return result

            if on_progress:
                on_progress(f"✅ 已接受安装请求: {result.get('message')}\n")

            # 2. check installation detail information
            self._poll_install_status(on_progress)
            return result

        except httpx.ConnectError:
            error_msg = "无法连接后端，请确认服务是否启动"
            if on_progress:
                on_progress(f"❌ {error_msg}\n")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = str(e)
            if on_progress:
                on_progress(f"❌ 请求异常: {error_msg}\n")
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
                on_progress("⚠️ 安装监控超时（超过30分钟）\n")