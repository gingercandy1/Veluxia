"""资料库服务与 /library 路由契约（ADR 0004）。

用临时模板目录 + 假 runner，无权重、无 GPU 也可跑。
"""
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.backend.core import model_utils
from src.backend.core.collection import library, steps, template
from src.backend.core.collection.steps import StepContext, StepRunner
from src.backend.core.exceptions import GenerationCancelled
from src.backend.router.library import LibraryRouter


class _EchoRunner(StepRunner):
    """把条目提示词写成文本产物；gate 未放行时阻塞，用来模拟执行中的资源包。"""
    type_name = "test.echo"
    gate = threading.Event()
    started = threading.Event()

    def run(self, ctx: StepContext):
        type(self).started.set()
        while not type(self).gate.wait(0.02):
            if ctx.cancel_event.is_set():
                raise GenerationCancelled()
        path = ctx.out_dir / f"{ctx.step_id}.txt"
        path.write_text(ctx.prompt, encoding="utf-8")
        return [path]


@pytest.fixture
def client(tmp_path, monkeypatch):
    media = tmp_path / "media"
    templates = (tmp_path / "templates").resolve()
    templates.mkdir()
    (templates / "echo.json").write_text(json.dumps({
        "id": "echo", "version": 2, "type": "scene",
        "steps": [{"id": "write", "type": "test.echo"}],
    }), encoding="utf-8")
    monkeypatch.setattr(model_utils, "get_media_root", lambda: media)
    monkeypatch.setattr(library, "get_media_root", lambda: media)
    monkeypatch.setattr(template, "TEMPLATE_DIR", templates)
    steps.register_runner(_EchoRunner())
    _EchoRunner.gate.set()
    _EchoRunner.started.clear()

    app = FastAPI()
    app.include_router(LibraryRouter().router)
    with TestClient(app) as test_client:
        yield test_client
    _EchoRunner.gate.set()
    steps._RUNNERS.pop(_EchoRunner.type_name, None)


def _create(client, **overrides):
    body = {"name": "森林", "template": "echo", "style": {"prompt": "teal"},
            "items": [{"id": "mushroom", "prompt": "mushroom"}, {"prompt": "vine"}]}
    body.update(overrides)
    return client.post("/library/packs", json=body)


def _wait_job(client, job_id):
    for _ in range(200):
        status = client.get(f"/library/run/status/{job_id}").json()
        if status["status"] in ("done", "error", "cancelled"):
            return status
        time.sleep(0.02)
    raise AssertionError("任务没有在预期时间内结束")


def test_create_list_get_delete(client):
    created = _create(client)
    assert created.status_code == 200
    manifest = created.json()["manifest"]
    assert manifest["template_version"] == 2 and manifest["type"] == "scene"
    # 没填 id 的条目按序号生成
    assert [item["id"] for item in manifest["items"]] == ["mushroom", "02"]
    pack_id = manifest["id"]
    assert created.json()["media_base"] == f"/media/library/{pack_id}"

    listed = client.get("/library/packs").json()["packs"]
    assert [p["manifest"]["id"] for p in listed] == [pack_id]
    assert client.get(f"/library/packs/{pack_id}").json()["manifest"]["name"] == "森林"

    assert client.delete(f"/library/packs/{pack_id}").json()["ok"] is True
    assert client.get(f"/library/packs/{pack_id}").status_code == 404
    assert client.delete(f"/library/packs/{pack_id}").status_code == 404


@pytest.mark.parametrize("overrides", [
    {"template": "missing"},
    {"template": "../echo"},
    {"items": []},
    {"items": [{"id": "../x", "prompt": "a"}]},
    {"items": [{"id": "a", "prompt": "1"}, {"id": "a", "prompt": "2"}]},
])
def test_create_rejects_invalid_requests(client, overrides):
    assert _create(client, **overrides).status_code == 400


def test_pack_id_cannot_escape_library(client):
    assert client.get("/library/packs/..%5Cconfig").status_code == 400
    with pytest.raises(ValueError):
        library.pack_dir("../config")


def test_broken_manifest_is_skipped_in_list(client):
    pack_id = _create(client).json()["manifest"]["id"]
    broken = library.library_root() / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{", encoding="utf-8")
    assert [p["manifest"]["id"] for p in client.get("/library/packs").json()["packs"]] == [pack_id]


def test_run_job_writes_outputs_into_manifest(client):
    pack_id = _create(client).json()["manifest"]["id"]
    job = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    status = _wait_job(client, job["job_id"])
    assert status["status"] == "done", status["error"]

    pack = client.get(f"/library/packs/{pack_id}").json()
    assert pack["running"] is False
    state = pack["manifest"]["items"][0]["steps"]["write"]
    assert state == {"status": "done", "outputs": ["mushroom/write.txt"], "error": None}
    assert (library.pack_dir(pack_id) / "mushroom" / "write.txt").read_text(
        encoding="utf-8") == "mushroom, teal"


def test_running_pack_cannot_be_run_twice_or_deleted(client):
    pack_id = _create(client).json()["manifest"]["id"]
    _EchoRunner.gate.clear()
    first = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    assert _EchoRunner.started.wait(5)

    assert client.get(f"/library/packs/{pack_id}").json()["running"] is True
    second = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    rejected = _wait_job(client, second["job_id"])
    assert rejected["status"] == "error" and "已在执行" in rejected["error"]
    assert client.delete(f"/library/packs/{pack_id}").status_code == 409

    _EchoRunner.gate.set()
    assert _wait_job(client, first["job_id"])["status"] == "done"
    assert client.delete(f"/library/packs/{pack_id}").status_code == 200


def test_cancel_run_keeps_manifest_and_releases_pack(client):
    pack_id = _create(client).json()["manifest"]["id"]
    _EchoRunner.gate.clear()
    job = client.post("/library/run/submit", json={"pack_id": pack_id}).json()
    assert _EchoRunner.started.wait(5)

    assert client.post(f"/library/run/cancel/{job['job_id']}").json()["ok"] is True
    assert _wait_job(client, job["job_id"])["status"] == "cancelled"
    pack = client.get(f"/library/packs/{pack_id}").json()
    assert pack["running"] is False
    assert pack["manifest"]["items"][0]["steps"]["write"]["status"] == "pending"


def test_routers_without_factory_type_skip_models_route(client):
    assert client.get("/library/models").status_code == 404


def test_builtin_templates_are_listed():
    ids = [t.id for t in template.list_templates()]
    assert "scene_props" in ids
