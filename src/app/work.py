import socket
import subprocess
import sys
import time

import psutil
from PySide6.QtCore import QThread, Signal

from param import GenerationRequest
from src.shared.enum_type import FactoryType
from src.app.client import ApiClient, ApiGuardClient
from src.shared.schemas import BaseResponse


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
            self._emit_result(result)

        except Exception as e:
            self.error.emit(str(e))

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
            [sys.executable, "-m", "backend.server", "--port", str(port)],
            stdout=None,
            stderr=None,
        )
        return proc

    @staticmethod
    def wait_for_backend(retries: int = 20, interval: float = 0.5) -> bool:
        for _ in range(retries):
            if ApiClient.instance().health():
                print("✅ Backend 已就绪")
                return True
            time.sleep(interval)
        print("⚠️  Backend 启动超时，继续运行（可能部分功能不可用）")
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
            [sys.executable, "-m", "backend.system", "--port", str(port)],
            stdout=None,
            stderr=None,
        )
        return proc

    @staticmethod
    def wait_for_backend(retries: int = 20, interval: float = 0.5) -> bool:
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

        self.ready.emit()


if __name__ == '__main__':
    _client = ApiClient()
    ApiProcess.start_backend()

    ApiProcess.wait_for_backend(_client, retries=20)