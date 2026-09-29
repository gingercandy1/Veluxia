"""前端拉起本机后端：端口探测不靠连接（Windows 上连空端口要约 2 秒），两个服务同时启动。"""
import socket
import subprocess
import sys
import time

import pytest

from src.app import work
from src.app.work import BackendStartupWorker, BaseProcess, wait_until_healthy


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_port_probe_sees_listeners_without_connecting():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert BaseProcess.is_port_in_use(port)
    assert not BaseProcess.is_port_in_use(_free_port())


def test_wait_asks_health_once_the_port_listens():
    calls = []
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert wait_until_healthy(port, lambda: calls.append(1) or True, timeout=2)
    assert calls == [1]


def test_wait_skips_health_while_nothing_listens():
    calls = []
    started = time.monotonic()
    assert not wait_until_healthy(_free_port(), lambda: calls.append(1) or False,
                                  timeout=0.6, interval=0.05)
    assert time.monotonic() - started < 2
    assert calls == []  # 没人监听时不去白等一次被拒的连接


def test_wait_gives_up_when_the_process_exited():
    proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    proc.wait()
    started = time.monotonic()
    assert not wait_until_healthy(_free_port(), lambda: True, timeout=30, proc=proc)
    assert time.monotonic() - started < 1


@pytest.fixture
def startup(monkeypatch):
    events = []
    monkeypatch.setattr(work.ApiProcess, "start_backend",
                        staticmethod(lambda port=8765: events.append("start backend") or "B"))
    monkeypatch.setattr(work.ApiGuardProcess, "start_backend",
                        staticmethod(lambda port=8756: events.append("start guard") or "G"))
    monkeypatch.setattr(work.ApiProcess, "wait_for_backend",
                        staticmethod(lambda **kw: events.append(f"wait {kw['proc']}") or True))
    monkeypatch.setattr(work.ApiGuardProcess, "wait_for_backend",
                        staticmethod(lambda **kw: events.append(f"wait {kw['proc']}") or True))
    worker = BackendStartupWorker()
    monkeypatch.setattr(worker, "_wait_models_and_ready", lambda: events.append("models"))
    return worker, events


def test_backend_and_guard_start_together(startup):
    worker, events = startup
    worker._start_from_source()
    assert events == ["start backend", "start guard", "wait B", "wait G", "models"]
