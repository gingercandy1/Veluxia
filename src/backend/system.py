import argparse
import asyncio
import platform
import shutil
import subprocess
from typing import Optional, Dict

import psutil
import uvicorn
from fastapi import APIRouter, FastAPI
from starlette.middleware.cors import CORSMiddleware


class SystemRouter:
    def __init__(self, project_root: Optional[str] = None):
        from src.shared.settings import PROJECT_ROOT
        self.project_root = project_root or PROJECT_ROOT
        self.router = APIRouter(prefix="/system", tags=["system"])

        self._state = {
            "status":  "idle",  # idle | running | done | failed
            "log":     [],
            "message": "",
        }

        self._register_routes()

    def _register_routes(self):
        self.router.post("/install-backend")(self.install_backend)
        self.router.get("/install-status")(self.install_status)
        self.router.post("/shutdown")(self.shutdown)
        self.router.get("/detect-device")(self.detect_device)   # ← 新增

    async def detect_device(self) -> Dict:
        info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpus": [],
            "cpu": {
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_count_physical": psutil.cpu_count(logical=False),
                "cpu_brand": platform.processor() or "Unknown",
                "architecture": platform.machine(),
            },
            "current_device": "cpu"
        }

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                info["cuda_available"] = True
                gpus = []

                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "total_memory_gb": round(int(parts[2]) / 1024, 2),
                            "driver_version": parts[3],
                            "compute_cap": parts[4] if len(parts) > 4 else None,
                        })

                info["gpus"] = gpus
                info["gpu_count"] = len(gpus)
                info["current_device"] = f"cuda:0" if gpus else "cpu"

                try:
                    version_result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, timeout=3
                    )
                    if version_result.stdout:
                        info["cuda_version"] = version_result.stdout.strip().split()[0]
                except:
                    pass

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        return info

    async def install_backend(self, payload: dict):
        extra = payload.get("extra", "cpu")

        if self._state["status"] == "running":
            return {"status": "already_running", "message": "安装正在进行中"}

        self._reset_state()
        asyncio.create_task(self._run_install(extra))
        return {"status": "accepted", "message": f"开始安装 {extra.upper()} 后端"}

    async def install_status(self):
        log_lines = self._state["log"].copy()
        self._state["log"].clear()

        return {
            "status":  self._state["status"],
            "log":     "".join(log_lines),
            "message": self._state["message"],
        }

    async def shutdown(self):
        import os
        import signal

        async def _do_shutdown():
            await asyncio.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)

        asyncio.create_task(_do_shutdown())
        return {"status": "shutting_down"}

    async def _run_install(self, extra: str):
        uv = shutil.which("uv")
        if not uv:
            self._set_failed("找不到 uv，请先安装：pip install uv")
            return

        cmd = self._build_cmd(uv, extra)
        self._append_log(f"运行: {' '.join(cmd)}\n")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_root,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            async for raw_line in proc.stdout:
                line = raw_line.decode("utf-8", errors="replace")
                line = line.replace("\r\n", "\n").replace("\r", "\n")
                self._append_log(line)

            await proc.wait()

            if proc.returncode == 0:
                self._set_done(f"{extra.upper()} 安装成功，请重启服务")
            else:
                self._set_failed(f"安装失败（退出码 {proc.returncode}）")

        except asyncio.CancelledError:
            self._set_failed("安装任务被取消")
        except Exception as e:
            self._set_failed(f"安装出错: {e}")

    def _build_cmd(self, uv: str, extra: str) -> list[str]:
        if extra in ("cuda", "cpu"):
            return [
                uv, "pip", "install", "-e", f".[{extra}]",
                "--reinstall-package", "torch",
                "--reinstall-package", "torchvision",
                "--reinstall-package", "torchaudio",
                "--reinstall-package", "llama-cpp-python",
            ]
        return [uv, "pip", "install", "-e", f".[{extra}]"]

    def _reset_state(self):
        self._state.update({"status": "running", "log": [], "message": ""})

    def _append_log(self, line: str):
        self._state["log"].append(line)

    def _set_done(self, message: str):
        self._state.update({"status": "done", "message": message})

    def _set_failed(self, message: str):
        self._state.update({"status": "failed", "message": message})

    @property
    def is_running(self) -> bool:
        return self._state["status"] == "running"


def create_app() -> FastAPI:
    app = FastAPI()

    # 允许本地 UI 进程跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    _routers = [
        SystemRouter(),
    ]
    for r in _routers:
        app.include_router(r.router)

    # 健康检查
    @app.get("/health", tags=["System"])
    async def health() -> dict:
        return {"status": "ok"}
    return app

app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")