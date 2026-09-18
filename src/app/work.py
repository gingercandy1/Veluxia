import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import psutil
from PySide6.QtCore import QThread, Signal

from .param import GenerationRequest
from src.shared.enum_type import FactoryType
from src.app.client import ApiClient, ApiGuardClient
from src.shared.schemas import BaseResponse, ImageResponse, AnimationResponse, SpeechResponse
from src.shared.settings import PROJECT_ROOT

# 前端本地媒体缓存：把后端 /media/... URL 下载到这里再展示，
# 放在项目目录下而非系统临时目录，保证聊天记录里的历史附件不会被系统清理掉。
MEDIA_CACHE_DIR = Path(PROJECT_ROOT) / "cache" / "media"


class ApiWorker(QThread):
    finished_ok = Signal(object)
    error       = Signal(str)

    # 流式文本專用信號
    thinking_chunk = Signal(str)
    text_chunk     = Signal(str)
    stream_done    = Signal(bool)

    def __init__(self, client: ApiClient, request: GenerationRequest, model_type: str):
        super().__init__()
        self._client = client
        self._model_type = model_type
        self._request = request

    def run(self):
        try:
            self._translate()
            model_type = self._model_type
            if model_type == FactoryType.Image:
                result = self._client.generate_image(self._request)
            elif model_type == FactoryType.Animation:
                result = self._client.generate_animation(self._request)
            elif model_type == FactoryType.Speech:
                result = self._client.generate_speech(self._request)
            elif model_type == FactoryType.Text:
                result = self._run_stream()
            else:
                result = self._run_stream()
            result = self._resolve_media(result)
            self._emit_result(result)

        except Exception as e:
            self.error.emit(str(e))

    def _translate(self):
        """翻译提示词也是一次网络请求，放在这个后台线程里做（曾经在主线程里做，
        新气泡要等它跑完才能画出来，看起来像卡住了）。翻译失败不阻断生成，
        直接退回用户原始输入。"""
        try:
            response = self._client.translate(self._request, is_default=True)
            self._request.translated = response.translate_result or ""
        except Exception as e:
            print(f"⚠️ 翻译失败，使用原始提示词：{e}")

    def _download(self, media_url: Optional[str]) -> Optional[str]:
        if not media_url:
            return media_url
        return self._client.download_media(media_url, MEDIA_CACHE_DIR) or media_url

    def _resolve_media(self, result: BaseResponse) -> BaseResponse:
        """把响应里的 /media/... URL 换成本地缓存文件路径，展示层无需关心来源。"""
        if isinstance(result, ImageResponse):
            result.paths = [p for p in (self._download(p) for p in result.paths) if p]
        elif isinstance(result, AnimationResponse):
            result.video_path = self._download(result.video_path)
            result.frame_paths = [p for p in (self._download(p) for p in result.frame_paths) if p]
        elif isinstance(result, SpeechResponse):
            result.audio_path = self._download(result.audio_path)
        return result

    def _run_stream(self):
        for event in self._client.stream_text(self._request):
            t = event.get("type")
            if t == "thinking":
                self.thinking_chunk.emit(event["text"])
            elif t == "text":
                self.text_chunk.emit(event["text"])
            elif t == "error":
                self.error.emit(event["text"])
                return BaseResponse(ok=False)
            elif t == "done":
                self.stream_done.emit(True)
                break
        return BaseResponse(ok=True)

    def _emit_result(self, result: BaseResponse):
        if result.ok:
            self.finished_ok.emit(result)
        else:
            self.error.emit(result.error or "unknown error")



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
        self.log.emit("⏳ 正在啟動後端...")
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

        self.log.emit("⏳ 正在加载模型列表...")
        ApiProcess.wait_for_models()

        self.ready.emit()


if __name__ == '__main__':
    _client = ApiClient()
    ApiProcess.start_backend()

    ApiProcess.wait_for_backend(_client, retries=20)