"""远程访问 token 鉴权的契约测试（ADR 0005）。"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.auth import install_token_auth, is_loopback_host


def _make_app(token: str) -> TestClient:
    app = FastAPI()
    install_token_auth(app)
    app.state.api_token = token

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        return {"ready": True}

    return TestClient(app)


def test_no_token_configured_allows_all():
    client = _make_app("")
    assert client.get("/ready").status_code == 200


def test_health_is_public_when_token_configured():
    client = _make_app("secret")
    assert client.get("/health").status_code == 200


def test_missing_or_wrong_token_rejected():
    client = _make_app("secret")
    assert client.get("/ready").status_code == 401
    assert client.get("/ready", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/ready", headers={"Authorization": "secret"}).status_code == 401


def test_correct_token_accepted():
    client = _make_app("secret")
    resp = client.get("/ready", headers={"Authorization": "Bearer secret"})
    assert resp.status_code == 200


def test_loopback_hosts():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("localhost")
    assert is_loopback_host("::1")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("192.168.1.10")
