import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import psutil
from PySide6.QtCore import QThread, Signal

from .param import GenerationRequest
from src.shared.enum_type import FactoryType
from src.app import local_backend
from src.app.client import ApiClient, ApiGuardClient, BackendStatus, probe_backend
from src.shared.schemas import (
    BaseResponse, ImageResponse, AnimationResponse, SpeechResponse, TranscriptionResponse,
)
from src.shared.settings import BACKEND_URL, PROJECT_ROOT, ConfigManager

# 前端本地媒体缓存：把后端 /media/... URL 下载到这里再展示，
# 放在项目目录下而非系统临时目录，保证聊天记录里的历史附件不会被系统清理掉。
MEDIA_CACHE_DIR = Path(PROJECT_ROOT) / "cache" / "media"


class ApiWorker(QThread):
    finished_ok = Signal(object)
    error       = Signal(str)
    cancelled   = Signal()

    # 流式文本專用信號
    thinking_chunk = Signal(str)
    text_chunk     = Signal(str)
    stream_done    = Signal(bool)

    # 多图批量生成时，每完成一张（已下载到本地）就发一次，用于提前展示
    partial_ready  = Signal(str)

    # 模型就绪前的阶段（下载中 / 加载中），带进度，用于气泡里的占位提示
    stage_changed  = Signal(dict)

    def __init__(self, client: ApiClient, request: GenerationRequest, model_type: str):
        super().__init__()
        self._client = client
        self._model_type = model_type
        self._request = request
        self._stop_event = threading.Event()
        self._downloaded: dict[str, str] = {}

    def stop(self):
        """用户点了"停止"：通知后端取消任务（图片/动画/语音），
        文本流则直接断开连接。真正的中断发生在各自的取消点上，这里只是发信号。"""
        self._stop_event.set()

    def run(self):
        try:
            if self._stop_event.is_set():
                self.cancelled.emit()
                return
            model_type = self._model_type
            # 转写模式下文本框只是备注，不用翻译/优化
            if model_type != FactoryType.Transcription:
                self._prepare_prompt()
            if model_type == FactoryType.Image:
                result = self._client.generate_image(
                    self._request, stop_event=self._stop_event, on_partial=self._on_partial)
            elif model_type == FactoryType.Animation:
                result = self._client.generate_animation(self._request, stop_event=self._stop_event)
            elif model_type == FactoryType.Speech:
                result = self._client.generate_speech(self._request, stop_event=self._stop_event)
            elif model_type == FactoryType.Transcription:
                result = self._client.transcribe(self._request, stop_event=self._stop_event)
            elif model_type == FactoryType.Text:
                result = self._run_stream()
            else:
                result = self._run_stream()

            if self._stop_event.is_set():
                self.cancelled.emit()
                return

            result = self._resolve_media(result)
            if isinstance(result, AnimationResponse) and self._request.model_params.get("export_sprites"):
                self._export_sprites(result)
            self._emit_result(result)

        except Exception as e:
            if self._stop_event.is_set():
                self.cancelled.emit()
            else:
                self.error.emit(str(e))

    def _prepare_prompt(self):
        """勾选了"优化提示词"时由优化模型直接产出英文提示词，不再走翻译；
        优化失败则退回普通翻译流程，不阻断生成。"""
        if self._request.model_params.get("refine_prompt") and self._refine():
            return
        self._translate()

    def _refine(self) -> bool:
        mode = FactoryType.convert_to_text(self._model_type)
        try:
            response = self._client.refine_prompt(self._request, mode)
            if response.ok and response.refined:
                self._request.refined = response.refined
                print(f"✨ 提示词已优化: {response.refined}")
                return True
            print(f"⚠️ 提示词优化失败，使用原始提示词：{response.error}")
        except Exception as e:
            print(f"⚠️ 提示词优化失败，使用原始提示词：{e}")
        return False

    def _translate(self):
        """翻译提示词也是一次网络请求，放在这个后台线程里做（曾经在主线程里做，
        新气泡要等它跑完才能画出来，看起来像卡住了）。翻译失败不阻断生成，
        直接退回用户原始输入。"""
        try:
            response = self._client.translate(self._request, is_default=True)
            self._request.translated = response.translate_result or ""
        except Exception as e:
            print(f"⚠️ 翻译失败，使用原始提示词：{e}")

    def _export_sprites(self, result: AnimationResponse):
        """精灵图导出失败不应让已经生成好的动画作废，错误写进响应交给界面提示。
        帧文件此时已下载到本地缓存，后端与前端同机，可直接按本地路径导出。"""
        if not result.frame_paths:
            result.error = self.tr("No frames to export")
            return
        try:
            export = self._client.export_sprites(
                result.frame_paths, name="anim",
                fps=int(self._request.model_params.get("frame_rate", 12)), trim=True)
        except Exception as e:
            result.error = self.tr("Sprite sheet export failed: {0}").format(e)
            return
        result.export_paths = [p for p in (self._download(export.sheet_path), self._download(export.atlas_path)) if p]

    def _download(self, media_url: Optional[str]) -> Optional[str]:
        if not media_url:
            return media_url
        # 提前展示的中间结果已经下载过，最终结果里复用同一个本地路径（界面靠路径去重）。
        if media_url not in self._downloaded:
            self._downloaded[media_url] = self._client.download_media(media_url, MEDIA_CACHE_DIR) or media_url
        return self._downloaded[media_url]

    def _on_partial(self, urls: list[str]):
        for url in urls:
            local = self._download(url)
            if local and not self._stop_event.is_set():
                self.partial_ready.emit(str(local))

    def _resolve_media(self, result: BaseResponse) -> BaseResponse:
        """把响应里的 /media/... URL 换成本地缓存文件路径，展示层无需关心来源。"""
        if isinstance(result, ImageResponse):
            result.paths = [p for p in (self._download(p) for p in result.paths) if p]
        elif isinstance(result, AnimationResponse):
            result.video_path = self._download(result.video_path)
            result.frame_paths = [p for p in (self._download(p) for p in result.frame_paths) if p]
        elif isinstance(result, SpeechResponse):
            result.audio_path = self._download(result.audio_path)
        elif isinstance(result, TranscriptionResponse):
            result.srt_path = self._download(result.srt_path)
        return result

    def _run_stream(self):
        for event in self._client.stream_text(self._request, stop_event=self._stop_event):
            t = event.get("type")
            if t == "stage":
                self.stage_changed.emit(event)
            elif t == "thinking":
                self.thinking_chunk.emit(event["text"])
            elif t == "text":
                self.text_chunk.emit(event["text"])
            elif t == "cancelled":
                return BaseResponse(ok=False)
            elif t == "error":
                # 交给 run() 统一经 _emit_result 发 error；这里再发一次会让气泡多出一行 unknown error
                return BaseResponse.from_error(event["text"])
            elif t == "done":
                self.stream_done.emit(True)
                break
        return BaseResponse(ok=True)

    def _emit_result(self, result: BaseResponse):
        if result.ok:
            self.finished_ok.emit(result)
        else:
            self.error.emit(result.error or "unknown error")



class ModelListWorker(QThread):
    """在线程里拉取某类型的模型列表，避免切换模式时同步 HTTP 卡住主线程。"""
    loaded = Signal(str, object)  # (type_str, ModelInfoResponse)

    def __init__(self, type_str: str):
        super().__init__()
        self._type_str = type_str

    def run(self):
        self.loaded.emit(self._type_str, ApiClient.instance().get_model_info(self._type_str))


class ClearMemoryWorker(QThread):
    """在线程里清除后端会话记忆，避免清空聊天时同步 HTTP 卡住主线程。"""

    def __init__(self, session_id: str):
        super().__init__()
        self._session_id = session_id

    def run(self):
        resp = ApiClient.instance().clear_memory(session_id=self._session_id)
        if not resp.ok:
            print(f"⚠️ 清除会话记忆失败: {resp.error}")


class LibraryTaskWorker(QThread):
    """在线程里执行一次资料库调用（ApiClient 方法都是同步阻塞的），结果经 finished_ok 回到主线程。
    资料库页面的调用种类多、处理逻辑都在页面里，用一个通用 worker 而不是每种调用写一个类。"""
    finished_ok = Signal(object)
    error       = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            self.finished_ok.emit(self._fn(*self._args, **self._kwargs))
        except Exception as e:
            self.error.emit(str(e) or e.__class__.__name__)


class BackendInstallWorker(QThread):
    """下载 / 解压 / 安装本机后端，可能持续几十分钟（torch 要联网装），日志逐行回到主线程。"""
    progress    = Signal(str)
    finished_ok = Signal()
    error       = Signal(str)

    def __init__(self, root: Path, archive: Path | None = None):
        super().__init__()
        self._root = root
        self._archive = archive

    def run(self):
        try:
            local_backend.install(self._root, self.progress.emit, archive=self._archive)
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e) or e.__class__.__name__)


class BaseProcess:
    @staticmethod
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

    @staticmethod
    def kill_port(port: int) -> None:
        """殺掉佔用指定端口的進程"""
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.net_connections():
                    if conn.laddr.port == port:
                        print(f"⚠️ 殺掉殘留進程 PID={proc.pid} 占用端口 {port}")
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


class ApiProcess(BaseProcess):
    @staticmethod
    def start_backend(port: int = 8765) -> subprocess.Popen:
        if ApiProcess.is_port_in_use(port):
            print(f"⚠️ 端口 {port} 已被佔用，嘗試清理...")
            ApiProcess.kill_port(port)
            time.sleep(1.0)

        proc = subprocess.Popen(
            [sys.executable, "-m", "src.backend.server", "--port", str(port)],
            stdout=None,
            stderr=None,
        )
        return proc

    @staticmethod
    def wait_for_backend(retries: int = 30, interval: float = 0.5) -> bool:
        for _ in range(retries):
            if ApiClient.instance().health():
                print("✅ Backend 已就绪")
                return True
            time.sleep(interval)
        print("⚠️  Backend 启动超时，继续运行（可能部分功能不可用）")
        return False

    @staticmethod
    def wait_for_models(retries: int = 240, interval: float = 1.0) -> bool:
        """/health 只代表进程活着，模型注册在后台线程异步进行，这里单独等 /ready。"""
        for _ in range(retries):
            if ApiClient.instance().ready():
                print("✅ 模型已注册完成")
                return True
            time.sleep(interval)
        print("⚠️  模型注册超时，继续运行（模型列表可能不完整）")
        return False

    @staticmethod
    def wait_backend_down(timeout: int = 15) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not ApiClient.instance().health():
                return True  # 连不上了，说明已退出
            time.sleep(0.5)  # 还活着，继续等
        return False


class ApiGuardProcess(BaseProcess):
    @staticmethod
    def start_backend(port: int = 8756) -> subprocess.Popen:
        if ApiProcess.is_port_in_use(port):
            print(f"⚠️ 端口 {port} 已被佔用，嘗試清理...")
            ApiProcess.kill_port(port)
            time.sleep(1.0)

        proc = subprocess.Popen(
            [sys.executable, "-m", "src.backend.system", "--port", str(port)],
            stdout=None,
            stderr=None,
        )
        return proc

    @staticmethod
    def wait_for_backend(retries: int = 30, interval: float = 0.5) -> bool:
        for _ in range(retries):
            if ApiGuardClient.instance().health():
                print("✅ Backend 已就绪")
                return True
            time.sleep(interval)
        print("⚠️  Backend 启动超时，继续运行（可能部分功能不可用）")
        return False

class BackendStartupWorker(QThread):
    ready   = Signal()        # 後端就緒
    timeout = Signal()        # 啟動超時
    log     = Signal(str)     # 日誌輸出
    failed  = Signal(str)     # 需要用户去设置页处理（未安装 / 地址或 token 不对）

    # 安装版后端冷启动要导入 torch 等重型依赖，比源码模式慢得多
    INSTALLED_START_RETRIES = 240
    REMOTE_CONNECT_RETRIES = 10

    def __init__(self, port: int = 8765, guard_port: int = 8756):
        super().__init__()
        self._port = port
        self._guard_port = guard_port
        self._proc = None
        self._guard_proc = None

    def close_guard(self):
        proc = self._guard_proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)  # 等待最多5秒
                print("✅ Backend 子進程已終止")
            except subprocess.TimeoutExpired:
                proc.kill()  # 強制殺掉
                print("⚠️ Backend 強制Kill")

    def close(self):
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)  # 等待最多5秒
                print("✅ Backend 子進程已終止")
            except subprocess.TimeoutExpired:
                proc.kill()  # 強制殺掉
                print("⚠️ Backend 強制Kill")

    def run(self):
        server = ConfigManager().get_section("server")
        if server.get("mode") == "remote":
            self._connect_remote(server.get("url", ""), server.get("token", ""))
            return

        ApiClient.instance().configure(BACKEND_URL)
        if local_backend.is_frozen():
            self._start_installed()
        else:
            self._start_from_source()

    def _start_from_source(self):
        """源码运行：用当前解释器拉起后端和守护进程（开发流程）。"""
        self.log.emit("⏳ " + self.tr("Starting backend..."))
        self._proc = ApiProcess.start_backend(self._port)
        main_result = ApiProcess.wait_for_backend()
        if not main_result:
            self.timeout.emit()
            return

        self._guard_proc = ApiGuardProcess.start_backend(port=self._guard_port)
        guard_result = ApiGuardProcess.wait_for_backend()
        if not guard_result:
            self.timeout.emit()
            return

        self._wait_models_and_ready()

    def _start_installed(self):
        """exe 运行：自身没有后端依赖，从安装目录拉起后端。"""
        root = local_backend.install_dir()
        if not local_backend.is_installed(root):
            self.failed.emit(self.tr(
                "The local backend is not installed. Install it or set a remote address "
                "in Settings → Backend."))
            return
        self.log.emit("⏳ " + self.tr("Starting backend..."))
        self._proc = local_backend.start(root, self._port)
        if not ApiProcess.wait_for_backend(retries=self.INSTALLED_START_RETRIES):
            self.timeout.emit()
            return
        self._wait_models_and_ready()

    def _connect_remote(self, url: str, token: str):
        if not url:
            self.failed.emit(self.tr(
                "The remote backend address is empty. Set it in Settings → Backend."))
            return
        self.log.emit("⏳ " + self.tr("Connecting to {0}...").format(url))
        status = BackendStatus.UNREACHABLE
        for _ in range(self.REMOTE_CONNECT_RETRIES):
            status = probe_backend(url, token)
            if status != BackendStatus.UNREACHABLE:
                break
            time.sleep(1.0)
        if status == BackendStatus.UNAUTHORIZED:
            self.failed.emit(self.tr(
                "The remote backend rejected the token. Check it in Settings → Backend."))
            return
        if status == BackendStatus.UNREACHABLE:
            self.failed.emit(self.tr(
                "Cannot connect to {0}. Check the address in Settings → Backend.").format(url))
            return
        ApiClient.instance().configure(url, token)
        self._wait_models_and_ready()

    def _wait_models_and_ready(self):
        self.log.emit("⏳ " + self.tr("Loading model list..."))
        ApiProcess.wait_for_models()
        self.ready.emit()


if __name__ == '__main__':
    _client = ApiClient()
    ApiProcess.start_backend()

    ApiProcess.wait_for_backend(_client, retries=20)